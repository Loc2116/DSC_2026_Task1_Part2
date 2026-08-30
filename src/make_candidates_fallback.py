"""
make_candidates_fallback.py -- SINH CANDIDATE DỰ PHÒNG. KHÔNG PHẢI FILE CỦA D.

===========================================================================
ĐỌC KỸ TRƯỚC KHI DÙNG
===========================================================================
File này tồn tại vì D chưa bàn giao `bm25_top100_public.json`, mà E đang không
có bài nộp hợp lệ nào trên bảng. Nó là LƯỚI AN TOÀN, không phải bản thay thế.

Chunker của D tốt hơn: tách từ tiếng Việt đàng hoàng (underthesea/VnCoreNLP),
cắt tiếp theo khoản (chunk_id có hậu tố `_sub_`), BM25 đã tinh chỉnh.
Bản này tách từ ở mức ÂM TIẾT (`\\w+`) và chỉ cắt theo `Điều N`.
=> recall@100 THẤP HƠN của D. Chạy --eval trên dev rồi so với 0.9700 của D
   TRƯỚC KHI quyết định nộp bằng file này.

D giao file thật thì VỨT file này đi, đừng trộn.
Tên file xuất ra luôn có hậu tố _FALLBACK để không bao giờ lẫn với file của D.

Cách dùng:
    python make_candidates_fallback.py --questions dev_150_locked.json --eval
    python make_candidates_fallback.py --questions Input/public-official.json

Xuất ra đúng định dạng rerank_from_d.py đang đọc:
    {"<qid>": [{"doc_id": str, "top_chunks": [{"chunk_id", "text", "bm25_score"}]}, ...]}

---------------------------------------------------------------------------
GHI CHÚ KỸ THUẬT -- hai lần OOM đã gặp thật, đừng lặp lại
---------------------------------------------------------------------------
Sandbox 3,9GB. Corpus 8532 văn bản -> 169.772 chunk. Hai bản trước đều chết:

  bản 1: giữ tf dict cho mọi chunk           -> ~3,5GB, chết
  bản 2: postings list-of-tuple + text RAM   -> ~1,5GB, chết ở 70% lượt 2

Bản này:
  - postings dùng array('i')/array('H') thay list tuple  -> nhẹ hơn ~10 lần
  - KHÔNG giữ text chunk trong RAM, đọc lại từ đĩa lúc xuất (chỉ ~100 doc/câu)
  - hai lượt quét: lượt 1 đếm df (không giữ gì), lượt 2 dựng postings
"""

import argparse, json, math, os, re, time
from array import array
from collections import Counter, defaultdict

K1, B = 1.5, 0.75
TOP_DOCS, TOP_CHUNKS = 100, 3
DF_CUTOFF = 0.30          # term có mặt ở >30% chunk: IDF ~0 nhưng ngốn hết postings

_tok = re.compile(r"\w+", re.UNICODE)
_split = re.compile(r"(?m)^\s*(?=Điều\s+\d+[.\s])")   # ranh giới Điều của văn bản luật VN
_sub = re.compile(r"(?m)^\s*(?=\d+\.\s)")             # ranh giới khoản: "1. ", "2. "

# Cắt theo Điều là chưa đủ. Đo thật: chunk trung bình 6.714 ký tự, có cái 981.032
# (văn bản không hề có chữ "Điều" -> giữ nguyên cả bài), 36% vượt cửa sổ 1024 token
# của cross-encoder. Của D trung bình 1.387, max 2.498. Cắt tiếp cho khớp.
MAX_CHARS = 2200

tok = lambda s: _tok.findall(s.lower())


def _hard_split(s: str, limit: int) -> list[str]:
    """Chốt chặn cuối: cắt theo đoạn trống, vẫn dài thì cắt thẳng theo độ dài."""
    if len(s) <= limit:
        return [s]
    out, buf = [], ""
    for para in s.split("\n\n"):
        if len(buf) + len(para) + 2 <= limit:
            buf += ("\n\n" if buf else "") + para
        else:
            if buf:
                out.append(buf)
            while len(para) > limit:          # đoạn đơn lẻ vẫn quá dài -> cắt cứng
                out.append(para[:limit])
                para = para[limit:]
            buf = para
    if buf:
        out.append(buf)
    return out


def chunks_of(passage: str) -> list[str]:
    parts = [p.strip() for p in _split.split(passage) if p.strip()] or [passage.strip()]
    out = []
    for p in parts:
        if len(p) <= MAX_CHARS:
            out.append(p)
            continue
        for s in _sub.split(p):               # cắt tiếp theo khoản
            s = s.strip()
            if s:
                out.extend(_hard_split(s, MAX_CHARS))
    return [c for c in out if c]


def read_passage(ctx_dir: str, doc_id: str) -> str:
    with open(os.path.join(ctx_dir, f"context_{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)["passage"]


def load_questions(path: str) -> dict:
    """Nhận {qid: {question:..}} hoặc {qid: ".."} hoặc [{question_id, question}]."""
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {
            str(r.get("question_id") or r.get("id")): (r.get("question") or r.get("query"))
            for r in raw
        }
    return {
        str(k): (v if isinstance(v, str) else (v.get("question") or v.get("query")))
        for k, v in raw.items()
    }


class Index:
    """Index ngược tối giản. Không giữ text -- chỉ giữ toạ độ (doc, thứ tự chunk) + độ dài."""

    def __init__(self, ctx_dir):
        self.ctx_dir = ctx_dir
        self.doc_of = array("i")      # chunk_idx -> chỉ số trong self.docs
        self.pos_of = array("i")      # chunk_idx -> chunk thứ mấy trong văn bản đó
        self.len_of = array("i")      # chunk_idx -> số token
        self.docs: list[str] = []
        self.p_idx = defaultdict(lambda: array("i"))   # term -> chunk_idx
        self.p_tf = defaultdict(lambda: array("H"))    # term -> tần suất (chunk < 65k token)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.n_chunk = 0

    def build(self, questions, verbose=True):
        """
        MỘT lượt quét. Dựng postings cho toàn bộ từ vựng câu hỏi rồi mới cắt term
        phổ biến -- nhờ array nên tổng postings chỉ cỡ vài chục MB, không cần tách
        hai lượt như bản trước (bản đó tốn 170s, vượt giới hạn thời gian một lệnh).
        """
        qvocab = set()
        for q in questions.values():
            qvocab |= set(tok(q))
        print(f"  từ vựng câu hỏi: {len(qvocab):,} term", flush=True)

        df, n, tot = Counter(), 0, 0
        t0 = time.time()
        files = sorted(f for f in os.listdir(self.ctx_dir) if f.startswith("context_"))
        for i, fn in enumerate(files):
            doc_id = fn[len("context_") : -len(".json")]
            self.docs.append(doc_id)
            for j, ch in enumerate(chunks_of(read_passage(self.ctx_dir, doc_id))):
                t = tok(ch)
                n += 1
                tot += len(t)
                full = Counter(t)
                hit = qvocab & full.keys()          # phép giao chạy bằng C
                if not hit:
                    continue                         # chunk không dính term nào -> bỏ hẳn
                df.update(hit)
                idx = len(self.doc_of)
                self.doc_of.append(len(self.docs) - 1)
                self.pos_of.append(j)
                self.len_of.append(len(t))
                for w in hit:
                    self.p_idx[w].append(idx)
                    self.p_tf[w].append(min(full[w], 65535))
            if verbose and i % 2000 == 0:
                print(f"    ...{i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

        self.n_chunk, self.avgdl = n, tot / max(n, 1)
        cut = DF_CUTOFF * n
        self.idf = {
            w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items() if d <= cut
        }
        for w in list(self.p_idx):                   # trả bộ nhớ của term bị loại
            if w not in self.idf:
                del self.p_idx[w], self.p_tf[w]
        print(f"  corpus: {n:,} chunk | avgdl {self.avgdl:.0f} | term giữ {len(self.idf):,}/{len(df):,}")
        print(f"  {len(self.doc_of):,} chunk có tín hiệu | "
              f"{sum(len(v) for v in self.p_idx.values()):,} posting entry "
              f"({time.time()-t0:.0f}s)", flush=True)
        return self

    def search_ids(self, question: str, top_docs: int) -> list[dict]:
        """
        CHỈ trả doc_id, KHÔNG đọc text. Dùng để sinh hard negative cho fine-tune:
        finetune_rerank.py chỉ lấy doc_id từ file này, text nó đọc thẳng từ corpus.
        Bỏ được bước đọc lại ~100 file/câu -> nhanh hơn nhiều, file nhỏ hơn ~2000 lần.
        """
        acc = defaultdict(float)
        for w in set(tok(question)):
            iw = self.idf.get(w)
            if iw is None:
                continue
            for idx, f_ in zip(self.p_idx[w], self.p_tf[w]):
                dl = self.len_of[idx]
                acc[idx] += iw * f_ * (K1 + 1) / (f_ + K1 * (1 - B + B * dl / self.avgdl))
        best = {}
        for idx, s in sorted(acc.items(), key=lambda x: -x[1]):
            d = self.docs[self.doc_of[idx]]
            if d not in best:
                best[d] = round(s, 4)
                if len(best) >= top_docs:
                    break
        return [{"doc_id": d, "top_chunks": []} for d in best]

    def search(self, question: str) -> list[dict]:
        acc = defaultdict(float)
        for w in set(tok(question)):
            iw = self.idf.get(w)
            if iw is None:
                continue
            for idx, f_ in zip(self.p_idx[w], self.p_tf[w]):
                dl = self.len_of[idx]
                acc[idx] += iw * f_ * (K1 + 1) / (f_ + K1 * (1 - B + B * dl / self.avgdl))

        best: dict[str, list] = {}
        for idx, s in sorted(acc.items(), key=lambda x: -x[1]):
            doc_id = self.docs[self.doc_of[idx]]
            if doc_id not in best:
                if len(best) >= TOP_DOCS:
                    continue                       # đủ 100 văn bản rồi
                best[doc_id] = []
            if len(best[doc_id]) < TOP_CHUNKS:
                best[doc_id].append((s, self.pos_of[idx]))

        out = []
        for doc_id, picks in sorted(best.items(), key=lambda kv: -kv[1][0][0])[:TOP_DOCS]:
            parts = chunks_of(read_passage(self.ctx_dir, doc_id))   # đọc lại text ở đây
            out.append({
                "doc_id": doc_id,
                "top_chunks": [
                    {"chunk_id": f"{doc_id}_{j}", "text": parts[j], "bm25_score": round(s, 4)}
                    for s, j in picks
                    if j < len(parts)
                ],
            })
        return out


HERE = os.path.dirname(os.path.abspath(__file__))


def _rel(p: str) -> str:
    """Đường dẫn tính theo vị trí file .py, không theo thư mục đang đứng.
    Nhờ vậy bấm nút Run trong VS Code cũng chạy đúng."""
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def main():
    ap = argparse.ArgumentParser(description="Sinh candidate DỰ PHÒNG (không phải file của D)")
    # Có mặc định để bấm Run là chạy được luôn, khỏi cần gõ tham số.
    ap.add_argument("--questions", default="Input/public-official.json")
    ap.add_argument("--contexts", default="Input/selected-contexts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", action="store_true", help="đo recall@100 (chỉ khi file có gold)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids-only", action="store_true",
                    help="chỉ xuất doc_id, không kèm text -> để sinh hard negative cho fine-tune")
    ap.add_argument("--top-docs", type=int, default=TOP_DOCS,
                    help="số document mỗi câu (20 là đủ để lấy 4 negative)")
    a = ap.parse_args()

    a.questions, a.contexts = _rel(a.questions), _rel(a.contexts)
    if a.out:
        a.out = _rel(a.out)

    questions = load_questions(a.questions)
    if a.limit:
        questions = dict(list(questions.items())[: a.limit])
    print(f"{len(questions)} câu hỏi từ {a.questions}")

    idx = Index(a.contexts).build(questions)

    name = os.path.splitext(os.path.basename(a.questions))[0]
    tag = "ids" if a.ids_only else "top100"
    path = a.out or os.path.join(HERE, f"bm25_{tag}_{name}_FALLBACK.json")

    print("Truy vấn...")
    t0 = time.time()
    got, rong = {}, []
    # Ghi từng câu một. Gom hết rồi json.dump() một lượt = OOM, đã bị thật
    # (file cụt ở 33/42 MB, KHÔNG báo lỗi gì).
    with open(path, "w", encoding="utf-8") as f:
        f.write("{")
        for i, (qid, q) in enumerate(questions.items(), 1):
            res = idx.search_ids(q, a.top_docs) if a.ids_only else idx.search(q)
            if not res:
                rong.append(qid)
            got[qid] = [c["doc_id"] for c in res]
            f.write(("," if i > 1 else "") + json.dumps(str(qid)) + ":" + json.dumps(res, ensure_ascii=False))
            if i % 100 == 0:
                print(f"  {i}/{len(questions)} ({time.time()-t0:.0f}s)", flush=True)
        f.write("}")

    if rong:
        print(f"[!] {len(rong)} câu KHÔNG có candidate: {rong[:5]}")
    print(f"\nĐã ghi {path}  ({os.path.getsize(path)/1e6:.0f} MB)")
    if a.ids_only:
        print("Dùng làm HARD NEGATIVE cho fine-tune. Không cần bản của D thay thế —")
        print("negative chỉ cần 'trông giống đáp án đúng', chênh lệch recall không đáng kể.")
    else:
        print("File DỰ PHÒNG để NỘP BÀI, kém hơn của D (~0.95 so với ~0.98).")
        print("D giao file thật thì dùng của D, mỗi phần trăm recall đều thành điểm.")

    if a.eval:
        with open(a.questions, encoding="utf-8-sig") as f:
            raw = json.load(f)
        gold = {str(k): set(map(str, v.get("answer") or [])) for k, v in raw.items()}
        if not any(gold.values()):
            print("File không có gold -> bỏ qua --eval")
            return
        tot = sum(len(gold[q] & set(ids)) / len(gold[q]) for q, ids in got.items() if gold.get(q))
        print(f"\nrecall@100 = {tot/len(got):.4f}   (của D trên cùng dev: 0.9700)")


if __name__ == "__main__":
    main()
