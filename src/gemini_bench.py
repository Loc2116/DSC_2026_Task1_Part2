"""gemini_bench.py -- đo một model ngoài (Gemini...) trên CHÍNH dev300, không tốn lượt nộp.

Vì sao dev300 chứ không phải đề thi:
  - có nhãn sẵn -> tự chấm được, 0 lượt nộp, 0 giờ GPU
  - so thẳng với mốc của mình trên ĐÚNG tập câu đó (paired), không phải so hai bài rời

Vì sao KHÔNG đưa submission của mình vào:
  lượt 19/08 đưa submission_B_M20.json vào -> Gemini chép nguyên 725/725 câu, đúng cả
  thứ tự. Phép đo rỗng, mất một lượt nộp. Đầu vào ở đây CHỈ có câu hỏi + rổ ứng viên.

Dùng:
    python gemini_bench.py build              # 150 câu đầu theo qid (phần lớn là câu DỄ)
    python gemini_bench.py build --hard       # CHỈ câu bài mình đang sai <- đo cái này
    python gemini_bench.py ping               # thử khoá bằng 1 lượt gọi, làm TRƯỚC
    python gemini_bench.py run                # <- CHẠY QUA API, không phải upload tay
    python gemini_bench.py score out_*.md     # chấm bảng Gemini trả về, so với mình
    python gemini_bench.py build2 out_*.md    # LƯỢT 2: đọc kỹ lại đúng 5 văn bản lượt 1 chọn
    python gemini_bench.py score2 out_*.md -- out2_*.md    # so 3 cách: mình / 4+1 lượt1 / 4+1 lượt2
    python gemini_bench.py selftest
"""
import json, os, re, sys, glob, time, random, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

N_Q       = 150     # số câu lấy từ dev300 (150 câu đầu theo qid). Đọc kết quả bằng bảng
                    # đối chiếu từng câu ở cuối, đừng đọc bằng chênh lệch điểm.
N_CAND    = 50      # ứng viên/câu, xếp theo BM25. Trần rổ 50 = 0.9700 vs mình 0.9100
                    # -> 6 điểm dư địa. Rổ 30 chỉ cho 2,3 điểm, quá chật để đo.
EXCERPT   = 900     # ký tự trích mỗi văn bản
HEAD      = 260     # ký tự đầu văn bản (chứa tên + số hiệu)
EXC2_K    = 2       # lượt 2: số mảnh trích mỗi văn bản (đọc kỹ hơn lượt 1)
Q_PER_BATCH2 = 15   # lượt 2 chỉ 5 văn bản/câu nên nhét được nhiều câu hơn
MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
WORKERS   = 4       # gọi song song. Tăng nếu không dính giới hạn tốc độ
Q_PER_BATCH = 20    # ~1,2M ký tự/batch. 150 câu -> 8 batch; giảm N_Q nếu ngại việc tay

ROOT    = os.path.dirname(os.path.abspath(__file__))
CTX     = os.path.join(ROOT, "Input", "selected-contexts")
DEV     = os.path.join(ROOT, "dev_300_locked.json")
SCORES  = os.path.join(ROOT, "Ketqua_E", "scores_dev300_bm25pick_merge1800_M20_K20.json")
OUT     = os.path.join(ROOT, "Ketqua_E", "gemini")

PROMPT = """Bạn là trợ lý tra cứu văn bản pháp luật Việt Nam.

File đính kèm là JSON: mỗi phần tử có `qid`, `question`, và `candidates` — danh sách văn
bản ứng viên, mỗi văn bản gồm `doc_id`, `head` (phần đầu văn bản, chứa tên và số hiệu) và
`excerpt` (đoạn trong văn bản khớp câu hỏi nhất).

Với MỖI câu hỏi, chọn đúng 5 `doc_id` có khả năng chứa câu trả lời cao nhất, xếp theo độ
tin cậy giảm dần.

Ràng buộc bắt buộc:
- Chỉ được chọn `doc_id` có trong `candidates` của chính câu hỏi đó. Không bịa id.
- Đúng 5 id, không trùng nhau, không bỏ sót câu nào.
- Xử lý hết mọi câu trong file. Nếu dài quá thì cứ xuất tiếp, đừng cắt giữa chừng.

Chỉ trả về bảng markdown, không giải thích:

| qid | answer |
| --- | --- |
| 12345 | 111, 222, 333, 444, 555 |
"""


PROMPT2 = """Bạn là trợ lý tra cứu văn bản pháp luật Việt Nam.

File đính kèm là JSON: mỗi phần tử có `qid`, `question`, và `candidates` — ĐÚNG 5 văn bản
đã qua vòng sàng lọc trước, lần này kèm đoạn trích DÀI HƠN để bạn đọc kỹ.

Nhiệm vụ: với mỗi câu hỏi, đọc kỹ cả 5 văn bản rồi XẾP LẠI THỨ TỰ theo khả năng chứa câu
trả lời, chắc chắn nhất lên đầu. Hãy cân nhắc văn bản nào thực sự có nội dung trả lời được
câu hỏi, chứ không chỉ nhắc tới cùng chủ đề.

Ràng buộc bắt buộc:
- Trả lại ĐÚNG 5 doc_id đã cho, không thêm, không bớt, không thay. Chỉ đổi thứ tự.
- Xử lý hết mọi câu trong file, không bỏ sót.

Chỉ trả về bảng markdown, không giải thích:

| qid | answer |
| --- | --- |
| 12345 | 111, 222, 333, 444, 555 |
"""


def _load():
    sys.path.insert(0, ROOT)
    import deep_chunk as DC
    DC.MERGE_CHARS = 1800                      # cùng cấu hình bài chốt
    return DC, json.load(open(DEV, encoding="utf-8")), json.load(open(SCORES, encoding="utf-8"))


def _ours(S, q, n=2):
    """top-5 của cấu hình chốt: blend_bm25_first(n=2) rồi lấp bằng max(ce, ce_deep)."""
    e = S[q]
    ce = lambda v: max(v.get("ce", -9), v.get("ce_deep", -9))
    out = list(dict.fromkeys(sorted(e, key=lambda d: -e[d]["bm25"])[:n]))
    for d in sorted(e, key=lambda d: -ce(e[d])):
        if len(out) >= 5: break
        if d not in out: out.append(d)
    return out[:5]


def build(hard=False):
    """hard=True: CHỈ lấy câu bài mình đang SAI và gold còn nằm trong rổ ứng viên.

    Lấy mẫu theo thứ tự qid thì gần như toàn câu dễ -- 20 câu đầu cả hai bên đều
    đúng hết, 0 câu phân giải được. Câu khó chỉ chiếm ~1/11 nên muốn đo phải nhắm
    thẳng vào chúng. Mẫu này LỆCH (chọn theo lỗi của mình) nên KHÔNG dùng để so
    điểm tuyệt đối -- nó trả lời đúng một câu: còn dư địa nào cho bộ xếp hạng khác không.
    """
    DC, dev, S = _load()
    gold = lambda q: {str(x) for x in dev[q]["answer"]}
    if hard:
        qids = [q for q in sorted(dev)
                if not (gold(q) & set(_ours(S, q)))
                and (gold(q) & set(sorted(S[q], key=lambda d: -S[q][d]["bm25"])[:N_CAND]))]
        print(f"chế độ HARD: {len(qids)} câu bài mình đang sai mà gold vẫn trong rổ {N_CAND}")
    else:
        qids = sorted(dev)[:N_Q]
    os.makedirs(OUT, exist_ok=True)
    batches, cur = [], []
    for qid in qids:
        q = dev[qid]["question"]
        cands = sorted(S[qid], key=lambda d: -S[qid][d]["bm25"])[:N_CAND]
        items = []
        for d in cands:
            txt = DC.read_passage(CTX, d)
            exc = DC.pick_chunks(q, CTX, d, k=1)
            items.append({"doc_id": d,
                          "head": txt[:HEAD],
                          "excerpt": (exc[0] if exc else txt)[:EXCERPT]})
        cur.append({"qid": qid, "question": q, "candidates": items})
        if len(cur) == Q_PER_BATCH:
            batches.append(cur); cur = []
    if cur:
        batches.append(cur)

    for i, b in enumerate(batches, 1):
        p = os.path.join(OUT, f"batch_{i:02d}.json")
        json.dump(b, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{p}  {len(b)} câu  {os.path.getsize(p)/1e6:.2f} MB")
    open(os.path.join(OUT, "PROMPT.txt"), "w", encoding="utf-8").write(PROMPT)

    gold = {q: {str(x) for x in dev[q]["answer"]} for q in qids}
    hit = sum(len(gold[q] & set(sorted(S[q], key=lambda d: -S[q][d]["bm25"])[:N_CAND]))
              / len(gold[q]) for q in qids) / len(qids)
    print(f"\nPROMPT.txt đã ghi. TRẦN của rổ {N_CAND} ứng viên trên {N_Q} câu này: {hit:.4f}")
    print("Không ai vượt được trần đó, kể cả chọn hoàn hảo.")


def build2(paths):
    """Lượt 2: chỉ 5 văn bản lượt 1 đã chọn, trích dài hơn, bắt xếp lại thứ tự.

    Vì sao có lượt này: đo trên 15 câu khó, gold nằm trong top-5 của lượt 1 ở 10 ca
    nhưng chỉ được xếp HẠNG 1 ở 5 ca. Tức lượt 1 tìm đúng chỗ, sắp thứ tự kém.
    Chọn hoàn hảo trong 5 gợi ý đó = 0.9600 ước dev300 (so với 0.9333 nếu lấy hạng 1).
    Cùng khuôn "đọc sâu" đã ăn +2,67 điểm ngày 17/08, chỉ đổi bộ chấm.
    """
    DC, dev, S = _load()
    pred = {}
    for p in paths:
        pred.update(parse(open(p, encoding="utf-8").read()))
    qs = [q for q in pred if q in dev]
    if not qs:
        sys.exit("không đọc được câu nào từ output lượt 1")
    os.makedirs(OUT, exist_ok=True)
    batches, cur = [], []
    for qid in qs:
        q = dev[qid]["question"]
        items = []
        for d in pred[qid]:
            txt = DC.read_passage(CTX, d)
            items.append({"doc_id": d, "head": txt[:HEAD],
                          "excerpt": "\n\n[...]\n\n".join(DC.pick_chunks(q, CTX, d, k=EXC2_K))})
        cur.append({"qid": qid, "question": q, "candidates": items})
        if len(cur) == Q_PER_BATCH2:
            batches.append(cur); cur = []
    if cur:
        batches.append(cur)
    for i, b in enumerate(batches, 1):
        p = os.path.join(OUT, f"round2_{i:02d}.json")
        json.dump(b, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{p}  {len(b)} câu  {os.path.getsize(p)/1e6:.2f} MB")
    open(os.path.join(OUT, "PROMPT2.txt"), "w", encoding="utf-8").write(PROMPT2)
    print(f"\nPROMPT2.txt đã ghi. {len(qs)} câu, {len(batches)} lô.")


def merge41(ourlist, glist):
    """Giữ 4 suất của mình, suất 5 lấy lựa chọn đầu của model ngoài chưa có trong 4."""
    o = list(ourlist[:4])
    for d in glist:
        if d not in o:
            o.append(d); break
    return (o + list(ourlist[4:]))[:5]


def score2(p1, p2):
    """So ba cách trên cùng tập câu: chỉ mình · 4+1 lượt 1 · 4+1 lượt 2."""
    DC, dev, S = _load()
    r1 = {}; r2 = {}
    for p in p1: r1.update(parse(open(p, encoding="utf-8").read()))
    for p in p2: r2.update(parse(open(p, encoding="utf-8").read()))
    qs = [q for q in r2 if q in dev and q in r1]
    if not qs:
        sys.exit("không có câu nào có đủ cả hai lượt")
    ce = lambda v: max(v.get("ce", -9), v.get("ce_deep", -9))
    def ours(q, n=2):
        e = S[q]
        out = list(dict.fromkeys(sorted(e, key=lambda d: -e[d]["bm25"])[:n]))
        for d in sorted(e, key=lambda d: -ce(e[d])):
            if len(out) >= 5: break
            if d not in out: out.append(d)
        return out[:5]
    gold = lambda q: {str(x) for x in dev[q]["answer"]}
    rec = lambda f: sum(len(gold(q) & set(f(q))) / len(dev[q]["answer"]) for q in qs) / len(qs)
    bad = [q for q in qs if set(r2[q]) != set(r1[q])]
    print(f"chấm trên {len(qs)} câu")
    if bad: print(f"  !! {len(bad)} câu lượt 2 đổi TẬP id (đáng lẽ chỉ đổi thứ tự): {bad[:5]}")
    a = rec(ours); b = rec(lambda q: merge41(ours(q), r1[q])); c = rec(lambda q: merge41(ours(q), r2[q]))
    print(f"\n  chỉ bài mình          : {a:.4f}")
    print(f"  4+1, hạng 1 của lượt 1: {b:.4f}   ({100*(b-a):+.2f})")
    print(f"  4+1, hạng 1 của lượt 2: {c:.4f}   ({100*(c-a):+.2f})")
    moved = sum(1 for q in qs if r2[q][0] != r1[q][0])
    print(f"\n  lượt 2 đổi lựa chọn đầu ở {moved}/{len(qs)} câu")


def _api(text, key, tries=5):
    """Gọi Gemini REST bằng urllib -- KHÔNG cần cài SDK. temperature=0 cho đỡ dao động.

    Khoá phải đi trong HEADER `x-goog-api-key`, KHÔNG phải `?key=` trên URL.
    Khoá đời mới của AI Studio bắt đầu bằng `AQ.` (loại auth key, gắn service account);
    kiểu này chỉ nhận qua header. Bản `?key=` cũ trả 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.
    """
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"
           f":generateContent")
    body = json.dumps({"contents": [{"parts": [{"text": text}]}],
                       "generationConfig": {"temperature": 0}}).encode()
    hdr = {"Content-Type": "application/json", "x-goog-api-key": key}
    for i in range(tries):
        try:
            req = urllib.request.Request(url, body, hdr)
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:400]
            if e.code in (400, 401, 403):        # sai khoá/sai model -> thử lại vô ích
                raise SystemExit(f"HTTP {e.code} — dừng luôn, thử lại không giúp:\n{msg}")
            if i == tries - 1:
                raise SystemExit(f"HTTP {e.code} sau {tries} lần thử:\n{msg}")
            time.sleep(2 ** i + random.random())      # 429/503 thì lùi dần rồi thử lại
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i + random.random())


def ping():
    """Thử khoá bằng đúng MỘT lượt gọi rẻ tiền, trước khi chạy cả trăm câu."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("thiếu biến môi trường GEMINI_API_KEY")
    print(f"model = {MODEL} · khoá bắt đầu bằng {key[:4]}... dài {len(key)}")
    print("trả lời:", _api("Trả lời đúng một từ: OK", key).strip()[:80])
    print("=> khoá dùng được.")


def run(paths, prompt, prefix):
    """Chạy qua API, MỖI CÂU MỘT LƯỢT GỌI.

    Một câu một lượt là cố ý: không còn chuyện model lặng lẽ bỏ bớt câu khi lô quá to
    (lỗi đã làm mất 275 câu ở lượt nộp 19/08, và làm chat chỉ trả 5/20 câu).
    Ghi kết quả ngay sau mỗi câu + bỏ qua câu đã có -> đứt giữa chừng thì chạy lại là tiếp.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("thiếu biến môi trường GEMINI_API_KEY")
    lock = threading.Lock()
    for p in paths:
        out = os.path.join(OUT, prefix + os.path.basename(p).split("_")[-1].replace(".json", ".md"))
        done = parse(open(out, encoding="utf-8").read()) if os.path.exists(out) else {}
        todo = [it for it in json.load(open(p, encoding="utf-8")) if it["qid"] not in done]
        if not todo:
            print(f"{os.path.basename(p)}: đã xong đủ {len(done)} câu, bỏ qua"); continue
        if not done:
            open(out, "w", encoding="utf-8").write("| qid | answer |\n| --- | --- |\n")
        print(f"{os.path.basename(p)}: {len(todo)} câu cần chạy (đã có {len(done)})")
        n = [0]

        def one(it):
            txt = prompt + "\n\nDỮ LIỆU:\n" + json.dumps([it], ensure_ascii=False)
            ids = parse(_api(txt, key)).get(it["qid"], [])
            valid = {c["doc_id"] for c in it["candidates"]}
            ids = [d for d in ids if d in valid][:5]
            with lock:
                n[0] += 1
                if ids:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(f"| {it['qid']} | {', '.join(ids)} |\n")
                print(f"  [{n[0]}/{len(todo)}] {it['qid']}: {len(ids)} id"
                      + ("  <-- RỖNG/không hợp lệ" if len(ids) < 5 else ""))

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(one, todo))
        print(f"-> {out}")


def parse(md: str) -> dict:
    """Bảng markdown -> {qid: [5 id]}. Bỏ dòng tiêu đề và dòng gạch."""
    out = {}
    for line in md.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 2 or not re.fullmatch(r"\d+", c[0]):
            continue
        ids = re.findall(r"\d+", c[1])
        if ids:
            out[c[0]] = list(dict.fromkeys(ids))[:5]
    return out


def _recall(pred: dict, gold: dict) -> float:
    return sum(len(gold[q] & set(pred.get(q, []))) / len(gold[q]) for q in gold) / len(gold)


def score(paths):
    DC, dev, S = _load()
    pred = {}
    for p in paths:
        pred.update(parse(open(p, encoding="utf-8").read()))
    qs = [q for q in pred if q in dev]
    if not qs:
        sys.exit("không parse được câu nào khớp dev300")
    gold = {q: {str(x) for x in dev[q]["answer"]} for q in qs}

    ce = lambda v: max(v.get("ce", -9), v.get("ce_deep", -9))
    def ours(q, n=2):                                    # blend_bm25_first, cấu hình chốt
        e = S[q]
        bm = sorted(e, key=lambda d: -e[d]["bm25"])
        out = list(dict.fromkeys(bm[:n]))
        for d in sorted(e, key=lambda d: -ce(e[d])):
            if len(out) >= 5: break
            if d not in out: out.append(d)
        return out[:5]

    O = {q: ours(q) for q in qs}
    bad = [q for q in qs if len(pred[q]) != 5]
    off = [q for q in qs if any(d not in S[q] for d in pred[q])]
    cop = sum(1 for q in qs if pred[q] == O[q])
    g, o = _recall(pred, gold), _recall(O, gold)
    print(f"chấm trên {len(qs)} câu  (thiếu {len(dev)-len(qs)} câu so với dev300)")
    if bad: print(f"  !! {len(bad)} câu không đủ 5 id: {bad[:5]}")
    if off: print(f"  !! {len(off)} câu có id NGOÀI rổ ứng viên (model bịa): {off[:5]}")
    print(f"  trùng y hệt bài mình (cả thứ tự): {cop}/{len(qs)}"
          + ("   <-- NGHI CHÉP, kiểm lại đầu vào" if cop > 0.9*len(qs) else ""))
    print(f"\n  Gemini  : {g:.4f}")
    print(f"  bài mình: {o:.4f}")
    print(f"  chênh   : {100*(g-o):+.2f} điểm")
    # đối chiếu từng câu -- đọc thẳng, không cần công thức nhiễu
    gw = [q for q in qs if (gold[q] & set(pred[q])) and not (gold[q] & set(O[q]))]
    ow = [q for q in qs if (gold[q] & set(O[q])) and not (gold[q] & set(pred[q]))]
    print(f"\n  Gemini đúng / mình sai : {len(gw)} câu")
    print(f"  mình đúng / Gemini sai : {len(ow)} câu")
    print(f"  hai bên như nhau       : {len(qs)-len(gw)-len(ow)} câu")
    print("\nĐọc: chênh lệch chỉ đáng tin khi hai cột lệch nhau rõ. Lệch 1-3 câu là hoà.")


def selftest():
    md = """rác đầu file
| qid | answer |
| --- | --- |
| 111 | 1, 2, 3, 4, 5 |
| 222 | 9,9, 8, 7, 6, 5 |
"""
    p = parse(md)
    assert p == {"111": ["1", "2", "3", "4", "5"], "222": ["9", "8", "7", "6", "5"]}, p
    assert _recall({"a": ["1", "2"]}, {"a": {"1"}}) == 1.0
    assert _recall({"a": ["3"]}, {"a": {"1"}}) == 0.0
    assert _recall({"a": ["1"], "b": ["9"]}, {"a": {"1", "2"}, "b": {"9"}}) == 0.75
    print("selftest OK — parser bỏ được dòng rác/tiêu đề, khử trùng id, recall đúng")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "build": build(hard="--hard" in sys.argv)
    elif cmd == "build2": build2(sys.argv[2:] or sorted(glob.glob("out_*.md")))
    elif cmd == "ping": ping()
    elif cmd == "run":  run(sys.argv[2:] or sorted(glob.glob(os.path.join(OUT, "batch_*.json"))), PROMPT, "out_")
    elif cmd == "run2": run(sys.argv[2:] or sorted(glob.glob(os.path.join(OUT, "round2_*.json"))), PROMPT2, "out2_")
    elif cmd == "score": score(sys.argv[2:] or sorted(glob.glob("out_*.md")))
    elif cmd == "score2":
        a = sys.argv[2:]; i = a.index("--") if "--" in a else len(a)
        score2(a[:i], a[i+1:])
    else: selftest()
