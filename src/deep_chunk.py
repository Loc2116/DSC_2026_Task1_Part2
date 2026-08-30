"""
deep_chunk.py -- E5: chấm sâu vùng ranh giới top-M.

===========================================================================
Vì sao có file này
===========================================================================
Chẩn đoán 13/08 trên dev300 (dùng scores_dev300_ft.json, 314 gold / 300 câu):

    gold đã nằm top-5      275  (87,6%)   <- đã thắng
    gold ở CE hạng 6-10     17  ( 5,4%)   <- MỤC TIÊU của file này
    gold ở CE hạng 11-20      6  ( 1,9%)
    gold ở CE hạng 21-50      7  ( 2,2%)
    gold ở CE hạng 51+        2  ( 0,6%)
    gold ngoài top-100 BM25   7  ( 2,2%)  <- E không chạm tới được, D chịu

CẢ 17 văn bản hụt ở hạng 6-10 đều dài hơn 3 đoạn (trung vị 125, max 744).
D chỉ đưa 3 đoạn mỗi văn bản -> cross-encoder chấm văn bản ĐÚNG bằng bằng
chứng SAI, nên nó rơi xuống hạng 6-10. Fine-tune không cứu được (đã thử,
±1,5 câu) vì vấn đề không nằm ở năng lực model.

===========================================================================
Cách làm
===========================================================================
Tầng 1: giữ nguyên, D đưa gì chấm nấy -> có bảng xếp hạng 100 văn bản.
Tầng 2: với M văn bản đứng đầu, tự băm lại TOÀN VĂN, chọn K đoạn giống câu
        hỏi nhất, chấm lại. Điểm sâu = đoạn cao nhất trong K đoạn đó.

Vì sao chỉ M văn bản: chấm hết chunk của 100 văn bản tốn 13x chi phí hiện
tại (~23h cho đề thi -- đã đếm trên CPU ngày 13/08). M=10, K=20 chỉ tốn
thêm ~200 đoạn/câu so với 157 đoạn hiện có, tức ~2,3x.

Vì sao M > 5: recall@5 là TẬP HỢP. Đảo thứ tự bên trong top-5 không đổi
điểm một li nào. Muốn ăn thêm thì phải kéo văn bản từ hạng 6-10 LÊN top-5,
nên vùng cần chấm sâu phải phủ cả 1..10 để so sánh công bằng.

===========================================================================
KHÔNG ghi đè "ce" -- đây là chỗ quan trọng nhất
===========================================================================
Điểm sâu lưu vào khoá RIÊNG `ce_deep`, `ce` giữ nguyên. Nhờ vậy MỘT lượt GPU
cho ra ba biến thể, đo offline trên CPU không cần chạy lại:

    A "base"    dùng ce                        <- chính là baseline 0.8883
    B "max"     dùng max(ce, ce_deep)          <- thêm bằng chứng, không mất
    C "replace" dùng ce_deep nếu có, không thì ce

Đúng quy tắc 2 và 5 trong CLAUDE.md: mỗi lượt GPU phải sinh scores dùng lại
được, và gộp hết vào một phiên.

===========================================================================
17/08 -- HAI CẦN GẠT MỚI, đo trên `oracle_chunks_dev300.json`, 0 giờ GPU
===========================================================================
Lượt oracle chấm CE trên TOÀN BỘ đoạn của 24 gold hụt top-5. Từ đó quét được
mọi hàm chọn đoạn trên CPU. Kết quả (M=20, K=20, chỉ đổi cách chọn/băm):

    hàm chọn              chunks_of gốc   gộp 1800 ký tự
    count (cũ)               0.9083          0.9183
    idf                      0.9150          0.9183
    idf/sqrt(len)            0.9117          0.9217
    bm25 mức đoạn  <- chốt   0.9217          0.9250
    lai count+bm25           0.9183          0.9250
    CHỌN HOÀN HẢO (trần)     0.9250          0.9317

1. `count` cũ đếm số TERM RIÊNG BIỆT trùng nhau -- không IDF, không chuẩn hoá
   độ dài. Nó thưởng đoạn dài và coi "của" ngang "lữ hành". Thay bằng BM25
   mức đoạn (idf + tf bão hoà + chuẩn hoá độ dài) ăn +1,34 điểm, 0 dependency.
2. `MERGE_CHARS` gộp đoạn liền kề. `chunks_of` cắt theo Điều nên đoạn trung
   bình chỉ ~650 ký tự, 125-212 đoạn/văn bản -> K=20 chỉ phủ 9-16%. Gộp lên
   1800 thì phủ ~43%, thêm +0,33 nữa.

Lai count+bm25 THUA bm25 thuần -> bài học `max > replace` KHÔNG chuyển sang
đây. Chọn đoạn phải dứt khoát, đừng hedge.

Khoảng cách còn lại tới trần chỉ +0,67 = 2 câu -> model embedding tranh nhau
đúng 2 câu đó. ĐỪNG thêm dependency 2GB cho chừng đó.
"""

import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_candidates_fallback import chunks_of, read_passage, tok  # dùng lại, không chép

DEFAULT_M, DEFAULT_K = 10, 20

# 0 = giữ nguyên chunks_of (cắt theo Điều). >0 = gộp đoạn liền kề tới ~N ký tự.
# Đặt ở notebook: `DC.MERGE_CHARS = 1800` TRƯỚC khi gọi bất kỳ hàm nào (nó đổi cache).
MERGE_CHARS = 0
K1, B = 1.5, 0.75                                # tham số BM25 chuẩn, đừng vặn

_cache: dict[str, tuple] = {}   # doc_id -> (parts, tfs, lens, idf, avglen)


def _merge(parts: list[str], target: int) -> list[str]:
    """Gộp đoạn liền kề cho tới ~target ký tự. Không cắt đoạn -- chỉ nối."""
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) + 2 > target:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def _index(ctx_dir: str, doc_id: str) -> tuple:
    """Băm + tokenize + dựng thống kê BM25 một lần rồi nhớ.

    Một văn bản xuất hiện ở nhiều câu hỏi nên cache trả về gần như miễn phí.
    idf tính TRONG PHẠM VI văn bản: term xuất hiện ở mọi đoạn thì không phân
    biệt được gì bên trong văn bản đó, dù nó hiếm trên toàn corpus.
    """
    if doc_id not in _cache:
        parts = chunks_of(read_passage(ctx_dir, doc_id))
        if MERGE_CHARS:
            parts = _merge(parts, MERGE_CHARS)
        tfs = [Counter(tok(c)) for c in parts]
        lens = [sum(t.values()) for t in tfs]
        df = Counter()
        for t in tfs:
            df.update(t.keys())   # .keys() = SỐ ĐOẠN chứa term (document frequency).
                                  # df.update(t) với Counter thì CỘNG TẦN SUẤT -> idf sai bét.
        n = len(parts)
        idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        _cache[doc_id] = (parts, tfs, lens, idf, (sum(lens) / n) if n else 1.0)
    return _cache[doc_id]


def doc_chunks(ctx_dir: str, doc_id: str) -> list[str]:
    """Danh sách đoạn của một văn bản. `len(...)` dùng để đếm chi phí."""
    return _index(ctx_dir, doc_id)[0]


def pick_chunks(question: str, ctx_dir: str, doc_id: str, k: int = DEFAULT_K) -> list[str]:
    """K đoạn hợp câu hỏi nhất, chấm bằng BM25 TRONG PHẠM VI văn bản.

    Bản cũ dùng `len(qs & terms)` -- đếm term riêng biệt trùng nhau. Đo trên
    oracle 17/08: BM25 hơn nó +1,34 điểm dev300, vì count thưởng đoạn dài và
    không phân biệt term hiếm với term rác.

    Vẫn KHÔNG cần index BM25 của D -- thống kê tính tại chỗ trên chính các
    đoạn của văn bản này, nên chạy được ngay trên Kaggle.
    """
    parts, tfs, lens, idf, avg = _index(ctx_dir, doc_id)
    if len(parts) <= k:
        return list(parts)
    qs = set(tok(question))

    def score(i: int) -> float:
        tf, L = tfs[i], lens[i]
        return sum(idf.get(w, 0.0) * tf[w] * (K1 + 1) /
                   (tf[w] + K1 * (1 - B + B * L / avg))
                   for w in qs if w in tf)

    # key=-score chứ KHÔNG phải reverse=True: hoà điểm (rất nhiều đoạn ăn 0) thì
    # reverse=True lật ngược thứ tự đoạn, lấy đoạn CUỐI văn bản thay vì đoạn đầu.
    return [parts[i] for i in sorted(range(len(parts)), key=lambda i: -score(i))[:k]]


def count_deep_chunks(questions: dict, scores: dict, ctx_dir: str,
                      m: int = DEFAULT_M, k: int = DEFAULT_K, skip: int = 0) -> int:
    """Đếm TRƯỚC khi chấm. Quy tắc 6: ước chi phí bằng số chunk, không bằng cảm giác."""
    n = 0
    for qid in questions:
        s = scores.get(str(qid)) or {}
        for d in sorted(s, key=lambda x: -s[x]["ce"])[skip:m]:
            n += min(len(doc_chunks(ctx_dir, d)), k)
    return n


def deepen_one(question: str, doc_scores: dict, ctx_dir: str, score_fn,
               m: int = DEFAULT_M, k: int = DEFAULT_K, skip: int = 0) -> dict:
    """
    doc_scores: {doc_id: {"ce": float, "bm25": float}} của MỘT câu (tầng 1).
    Trả về dict cùng dạng, văn bản hạng skip..m có thêm khoá "ce_deep". "ce" GIỮ NGUYÊN.

    `skip` để nới M mà không chấm lại phần đã chấm: truyền vào scores của lượt
    M=10 rồi gọi skip=10, m=20 -> chỉ chấm hạng 11-20, `ce_deep` cũ được giữ
    nguyên vì hàm này copy doc_scores chứ không dựng lại. Thứ tự vẫn sort theo
    `ce` (tầng 1, không bao giờ bị ghi đè) nên hạng không đổi giữa hai lượt.

    CẢNH BÁO: `skip` chỉ đúng khi hai lượt dùng CÙNG `MERGE_CHARS` và cùng
    `pick_chunks`. Đổi cách băm/chọn thì `ce_deep` cũ không so được với mới --
    phải chấm lại từ hạng 0.

    Gộp toàn bộ đoạn vào MỘT lần .predict(). Bug hiệu năng đã ghi trong
    rerank_from_d.py: gọi predict() với batch tí hon thì overhead dispatch GPU
    ăn hết thời gian, không phải tính toán thật.
    """
    out = {d: dict(v) for d, v in doc_scores.items()}
    top = sorted(doc_scores, key=lambda d: -doc_scores[d]["ce"])[skip:m]

    texts, owner = [], []
    for d in top:
        for c in pick_chunks(question, ctx_dir, d, k):
            texts.append(c)
            owner.append(d)
    if not texts:
        return out

    sc = (score_fn.predict([[question, t] for t in texts])
          if hasattr(score_fn, "predict") else score_fn(question, texts))

    for d, s in zip(owner, sc):
        s = float(s)
        if s > out[d].get("ce_deep", float("-inf")):
            out[d]["ce_deep"] = s
    return out


def deepen_all(questions: dict, scores: dict, ctx_dir: str, score_fn,
               m: int = DEFAULT_M, k: int = DEFAULT_K, every: int = 25, skip: int = 0) -> dict:
    """Chạy deepen_one cho mọi câu. In tiến độ để biết còn bao lâu, đừng ngồi đoán."""
    import time

    t0 = time.time()
    out, n = {}, len(questions)
    for i, qid in enumerate(questions, 1):
        qid = str(qid)
        out[qid] = deepen_one(questions[qid], scores.get(qid) or {}, ctx_dir, score_fn, m, k, skip)
        if i % every == 0 or i == n:
            el = time.time() - t0
            print(f"  {i}/{n} câu | {el/60:.1f} phút | còn ~{el/i*(n-i)/60:.1f} phút", flush=True)
    return out


# --------------------------------------------------------------------------
# Ba cách đọc điểm -- chạy trên CPU từ file scores đã lưu, KHÔNG cần GPU
# --------------------------------------------------------------------------
VARIANTS = {
    "base":    lambda v: v["ce"],
    "max":     lambda v: max(v["ce"], v.get("ce_deep", float("-inf"))),
    "replace": lambda v: v.get("ce_deep", v["ce"]),
}


def rank_by(doc_scores: dict, variant: str = "max") -> list[str]:
    f = VARIANTS[variant]
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: -f(x[1]))]


if __name__ == "__main__":
    # Test logic không cần GPU, không cần corpus thật.
    fake = {"a": {"ce": 1.0, "bm25": 0.0}, "b": {"ce": 0.5, "bm25": 0.0}}
    fake["b"]["ce_deep"] = 2.0
    assert rank_by(fake, "base") == ["a", "b"]
    assert rank_by(fake, "max") == ["b", "a"], "ce_deep cao phải kéo b lên"
    assert rank_by(fake, "replace") == ["b", "a"]
    # doc không có ce_deep: 'max' giữ nguyên ce, 'replace' cũng rơi về ce
    assert VARIANTS["max"]({"ce": 3.0}) == 3.0
    assert VARIANTS["replace"]({"ce": 3.0}) == 3.0

    assert _merge(["a" * 100] * 5, 250) == ["a" * 100 + "\n\n" + "a" * 100] * 2 + ["a" * 100]
    assert _merge([], 100) == [] and _merge(["x"], 100) == ["x"]

    def _put(doc_id, parts):
        """Nhồi cache tay để test pick_chunks mà khỏi đọc corpus."""
        tfs = [Counter(tok(c)) for c in parts]
        lens = [sum(t.values()) for t in tfs]
        df = Counter()
        for t in tfs:
            df.update(t.keys())   # .keys() = SỐ ĐOẠN chứa term (document frequency).
                                  # df.update(t) với Counter thì CỘNG TẦN SUẤT -> idf sai bét.
        n = len(parts)
        _cache[doc_id] = (parts, tfs, lens,
                          {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()},
                          sum(lens) / n)

    # BM25 phải bỏ qua term xuất hiện ở MỌI đoạn (idf~0) và bắt term hiếm.
    # "thuế" có ở cả 3 đoạn -> vô giá trị; "trước bạ" chỉ ở đoạn 2 -> quyết định.
    _put("t", ["thuế thuế thuế thuế thuế thuế",
               "thuế trước bạ",
               "thuế thuế thuế thuế thuế thuế thuế"])
    assert pick_chunks("thuế trước bạ", ".", "t", k=1) == ["thuế trước bạ"], \
        "BM25 phải chọn đoạn có term hiếm, không phải đoạn nhiều 'thuế' nhất"
    # bản `count` cũ sẽ hoà giữa 3 đoạn (đều trùng đúng 1 term riêng biệt 'thuế'
    # ở đoạn 1/3, 2 term ở đoạn 2) -- test này chính là thứ nó không bắt được.

    _put("s", ["một đoạn"])                       # ít hơn k -> trả hết, không sort
    assert pick_chunks("bất kỳ", ".", "s", k=20) == ["một đoạn"]

    # skip: chấm hạng 3-4 thì hạng 1-2 KHÔNG bị đụng, ce_deep cũ phải còn nguyên
    for c in "abcd":
        _put(c, ["đoạn " + c])
    sc = {c: {"ce": 10.0 - i} for i, c in enumerate("abcd")}
    sc["a"]["ce_deep"] = 99.0                      # giả lượt trước đã chấm hạng 1-2
    fake_fn = lambda q, texts: [1.0] * len(texts)
    r = deepen_one("x", sc, ".", fake_fn, m=4, skip=2)
    assert r["a"]["ce_deep"] == 99.0, "lượt sau xoá mất ce_deep của lượt trước"
    assert "ce_deep" not in r["b"], "skip không loại đúng hạng đã chấm"
    print("OK - xếp hạng, gộp đoạn, BM25 chọn đoạn, skip: đúng hết")
