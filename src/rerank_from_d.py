"""
rerank_from_d.py (ponytail)
D giờ bàn giao candidate đã CHUNK SẴN (top-3 chunk/document, kèm bm25_score) -- không
còn cần load_corpus()/chunk_corpus.py/top_chunks_per_doc nữa, D đã làm hết rồi
(và làm tốt hơn: BM25 thật để chọn top chunk, không phải word-overlap thô).

Format D bàn giao (bm25_top100_dev.json):
{
  "qid": [
    {"doc_id": str, "top_chunks": [{"chunk_id": str, "text": str, "bm25_score": float}, ...]},
    ...
  ]
}
"""


def score_docs_from_d(question: str, candidates: list[dict], score_fn) -> dict[str, float]:
    """
    Chấm điểm cross-encoder cho TỪNG document, KHÔNG cắt top-k.
    Trả về {doc_id: điểm cao nhất trong các chunk của nó}. Rỗng nếu không có text nào.

    candidates: list các {"doc_id": str, "top_chunks": [{"text": str, ...}, ...]} -- từ D.
    score_fn: CrossEncoder thật (.predict) hoặc callable(question, list[text]) -> list[score].

    QUAN TRỌNG: gộp TOÀN BỘ chunk của tất cả candidate thành 1 lần gọi .predict() duy
    nhất (batch lớn), KHÔNG gọi riêng từng document (100 lần/câu, mỗi lần 1-3 chunk).
    Bug hiệu năng đã đo được thật: 15,000 lần gọi predict() với batch cực nhỏ tốn
    ~1 tiếng cho 150 câu (0.24s/lần, phần lớn là overhead dispatch GPU, không phải
    tính toán thật) -- gộp 1 lần/câu giảm overhead này gần hết.

    Mỗi document lấy điểm CAO NHẤT trong các chunk của nó.
    """
    all_texts: list[str] = []
    text_doc_ids: list[str] = []
    for c in candidates:
        doc_id = str(c.get("doc_id", ""))
        chunks = c.get("top_chunks") or []
        texts = [ch["text"] for ch in chunks if ch.get("text")]
        if not doc_id or not texts:
            continue  # candidate lỗi (thiếu doc_id hoặc rỗng hết chunk) -- bỏ qua an toàn
        all_texts.extend(texts)
        text_doc_ids.extend([doc_id] * len(texts))

    if not all_texts:
        return {}

    scores = (
        score_fn.predict([[question, t] for t in all_texts])
        if hasattr(score_fn, "predict")
        else score_fn(question, all_texts)
    )

    doc_scores: dict[str, float] = {}
    for doc_id, score in zip(text_doc_ids, scores):
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score

    return doc_scores


def rerank_question_from_d(question: str, candidates: list[dict], score_fn, k: int = 5) -> list[str]:
    """Top-k doc_id theo điểm cross-encoder. Giữ nguyên hành vi cũ."""
    ranked = sorted(
        score_docs_from_d(question, candidates, score_fn).items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [doc_id for doc_id, _ in ranked[:k]]


def bm25_doc_scores(candidates: list[dict]) -> dict[str, float]:
    """Điểm BM25 mức document = max qua các chunk -- cùng quy tắc gộp với cross-encoder."""
    return {
        str(c["doc_id"]): max(
            (ch.get("bm25_score", 0.0) for ch in c.get("top_chunks") or []), default=0.0
        )
        for c in candidates
        if c.get("doc_id")
    }


def clean_doc_name(name: str) -> str:
    """
    Slug URL của D -> chuỗi đọc được.
    'Luat-Pha-san-2014-238641' -> 'Luat Pha san 2014'

    Bỏ id số ở cuối (không mang nghĩa, chỉ làm nhiễu), đổi gạch nối thành khoảng trắng.
    LƯU Ý: slug KHÔNG DẤU, trong khi câu hỏi có dấu -> khớp kém hơn tiêu đề thật
    trong passage. Đây là bản rẻ tiền, thử trước vì không cần đụng corpus.
    """
    import re

    return re.sub(r"-\d+$", "", str(name or "")).replace("-", " ").strip()


def with_doc_name(candidates: dict, sep: str = "\n") -> dict:
    """
    Ghép tên văn bản vào ĐẦU mỗi chunk. Trả về dict mới, không sửa dict gốc.

    Vì sao cần: nhóm văn bản gần trùng (điều lệ các quỹ, quy chế các hội) có nội dung
    Điều gần như y hệt, chỉ khác tên văn bản. Chunk không mang tên -> cross-encoder
    không có cách nào phân biệt. Ca 147712 trên dev150 hỏng đúng vì lý do này.

    Chỉ đổi text đưa cho reranker, KHÔNG đụng index BM25 -> tầng của D không đổi,
    recall@100 giữ nguyên, rủi ro cô lập trong phần E sở hữu.
    """
    out = {}
    for qid, cands in candidates.items():
        new = []
        for c in cands:
            nm = clean_doc_name(c.get("name", ""))
            new.append({
                **c,
                "top_chunks": [
                    {**ch, "text": (nm + sep + ch["text"]) if nm else ch["text"]}
                    for ch in (c.get("top_chunks") or [])
                ],
            })
        out[str(qid)] = new
    return out


def blend_bm25_first(reranked: list, bm25_ranked: list, k: int = 5, n_bm25: int = 1) -> list[str]:
    """
    Giữ n_bm25 slot đầu cho BM25, phần còn lại theo reranker. Khử trùng, bù đủ k.

    Vì sao cần: reranker thuần vứt sạch thứ tự BM25. Đo trên dev_150_locked, 17 câu
    reranker làm hỏng thì 16 câu có gold nằm trong top-20 BM25, 6 câu ngay hạng 1.

        n_bm25=0 -> 0.8567 (rerank thuần)
        n_bm25=1 -> 0.8900   +3.3 điểm, cứu 6 câu / hy sinh 1
        n_bm25=2 -> 0.8733
        n_bm25=3 -> 0.8500

    ĐÃ THỬ VÀ THUA, đừng làm lại (đo trên scores_dev_AITeamVN_Vietnamese_Reranker.json):

        fusion điểm min-max  a=0.7 -> 0.8678
        fusion điểm z-score  a=0.6 -> 0.8678
        RRF                  k=10  -> 0.8644
        blend n=1 CHỒNG LÊN fusion -> 0.8744   (tệ hơn blend thuần)

    Fusion trộn điểm nên bôi ảnh hưởng BM25 lên cả 5 slot. blend chỉ GIỮ CHỖ 1 slot,
    4 slot còn lại vẫn thuần cross-encoder -- đó là lý do nó thắng.

    Bootstrap 3000 lần trên 150 câu: blend n=1 > pure ce ở 96.9% lần resample.
    CẢNH BÁO: n_bm25 chọn sau khi nhìn dev (150 câu, CI ±0.05). Xác nhận lại trên
    public LB trước khi tin hoàn toàn.
    """
    out: list[str] = []
    for d in list(bm25_ranked[:n_bm25]) + list(reranked) + list(bm25_ranked):
        d = str(d)
        if d not in out:
            out.append(d)
        if len(out) >= k:
            break
    return out[:k]


def score_all_from_d(questions: dict, candidates_dict: dict, score_fn) -> dict:
    """
    {qid: {doc_id: {"ce": float, "bm25": float}}} -- dump đủ để tinh chỉnh fusion offline.

    Lý do tồn tại: rerank_all_from_d vứt hết điểm, chỉ giữ 5 id. Mỗi lần đổi trọng số
    fusion lại phải chạy lại GPU. Dump một lần, mọi thí nghiệm sau chạy CPU vài giây.
    """
    out = {}
    for qid, question in questions.items():
        cands = candidates_dict.get(str(qid), [])
        bm = bm25_doc_scores(cands)
        # float(): CrossEncoder.predict trả numpy.float32 -> json.dump ném TypeError
        out[str(qid)] = {
            d: {"ce": float(s), "bm25": float(bm.get(d, 0.0))}
            for d, s in score_docs_from_d(question, cands, score_fn).items()
        }
    return out


def rerank_all_from_d(questions: dict, candidates_dict: dict, score_fn, k: int = 5) -> dict:
    predicted = {}
    for qid, question in questions.items():
        cands = candidates_dict.get(str(qid), [])
        predicted[str(qid)] = rerank_question_from_d(question, cands, score_fn, k=k)
    return predicted


if __name__ == "__main__":
    candidates = [
        {"doc_id": "A", "top_chunks": [
            {"chunk_id": "A#0", "text": "quy định về đăng ký xe máy cho người nước ngoài", "bm25_score": 10.0},
            {"chunk_id": "A#1", "text": "nội dung khác không liên quan", "bm25_score": 3.0},
        ]},
        {"doc_id": "B", "top_chunks": [
            {"chunk_id": "B#0", "text": "quy định về nghĩa vụ quân sự hằng năm", "bm25_score": 8.0},
        ]},
        {"doc_id": "C_rong", "top_chunks": []},  # ca lỗi: document không có chunk nào
        {"doc_id": "", "top_chunks": [{"chunk_id": "x", "text": "abc"}]},  # ca lỗi: thiếu doc_id
    ]

    class FakeModel:
        def predict(self, pairs):
            return [sum(1 for w in q.split() if w in d) for q, d in pairs]

    result = rerank_question_from_d("đăng ký xe máy người nước ngoài", candidates, FakeModel(), k=5)
    assert result[0] == "A", result
    assert "C_rong" not in result and "" not in result, result
    print("Test 1 OK - rerank đúng, bỏ qua an toàn candidate lỗi (rỗng chunk / thiếu doc_id):", result)

    # test callable-style score_fn (không phải object .predict)
    def fake_callable(q, texts):
        return [sum(1 for w in q.split() if w in t) for t in texts]

    result2 = rerank_question_from_d("đăng ký xe máy người nước ngoài", candidates, fake_callable, k=5)
    assert result2[0] == "A", result2
    print("Test 2 OK - callable-style score_fn hoạt động đúng:", result2)

    # test rerank_all_from_d cho nhiều câu
    questions = {"q1": "đăng ký xe máy người nước ngoài", "q2": "nghĩa vụ quân sự"}
    cands_dict = {"q1": candidates, "q2": candidates}
    predicted = rerank_all_from_d(questions, cands_dict, FakeModel(), k=2)
    assert predicted["q1"][0] == "A" and predicted["q2"][0] == "B"
    print("Test 3 OK - rerank_all_from_d chạy đúng cho nhiều câu:", predicted)

    # test 4: score_all_from_d dump đủ điểm mọi document + json.dump được (numpy float32)
    import json

    dumped = score_all_from_d({"q1": "đăng ký xe máy người nước ngoài"}, {"q1": candidates}, FakeModel())
    assert set(dumped["q1"]) == {"A", "B"}, dumped          # bỏ đúng 2 candidate lỗi
    assert dumped["q1"]["A"]["bm25"] == 10.0, dumped        # bm25 = max qua chunk (10.0 > 3.0)
    assert dumped["q1"]["A"]["ce"] >= dumped["q1"]["B"]["ce"], dumped
    json.dumps(dumped)                                      # ném TypeError nếu quên float()
    print("Test 4 OK - score_all_from_d dump đủ điểm ce+bm25, serialize được:", dumped)

    # test 5: rerank_question_from_d vẫn khớp với xếp hạng từ score_docs_from_d (không đổi hành vi)
    full = score_docs_from_d("đăng ký xe máy người nước ngoài", candidates, FakeModel())
    assert rerank_question_from_d("đăng ký xe máy người nước ngoài", candidates, FakeModel(), k=2) == [
        d for d, _ in sorted(full.items(), key=lambda x: x[1], reverse=True)[:2]
    ]
    print("Test 5 OK - tách hàm không đổi hành vi cũ")

    # test 6: blend_bm25_first -- BM25 hạng 1 luôn có mặt, không đẩy reranker ra quá 1 slot
    assert blend_bm25_first(["r1", "r2", "r3", "r4", "r5"], ["b1"], k=5) == ["b1", "r1", "r2", "r3", "r4"]
    # đã có sẵn trong top-k reranker -> không tốn thêm slot, kết quả đổi thứ tự chứ không mất ai
    assert blend_bm25_first(["r1", "b1", "r3", "r4", "r5"], ["b1"], k=5) == ["b1", "r1", "r3", "r4", "r5"]
    assert blend_bm25_first(["r1"], ["b1", "b2", "b3", "b4", "b5"], k=5) == ["b1", "r1", "b2", "b3", "b4"]
    assert blend_bm25_first([], ["b1", "b2"], k=5, n_bm25=0) == ["b1", "b2"]  # rỗng -> rơi hết về bm25
    assert blend_bm25_first([1, 2], [3], k=3) == ["3", "1", "2"]              # ép str
    print("Test 6 OK - blend_bm25_first giữ slot cho BM25, khử trùng, bù đủ k")

    # test 7: clean_doc_name + with_doc_name
    assert clean_doc_name("Luat-Pha-san-2014-238641") == "Luat Pha san 2014"
    assert clean_doc_name("") == "" and clean_doc_name(None) == ""
    src = {"q1": [{"doc_id": "A", "name": "Dieu-le-Quy-X-123",
                   "top_chunks": [{"chunk_id": "A#0", "text": "Điều 8. Hội đồng quản lý Quỹ"}]}]}
    got = with_doc_name(src)
    assert got["q1"][0]["top_chunks"][0]["text"] == "Dieu le Quy X\nĐiều 8. Hội đồng quản lý Quỹ"
    assert src["q1"][0]["top_chunks"][0]["text"] == "Điều 8. Hội đồng quản lý Quỹ", "đã sửa dict gốc!"
    assert with_doc_name({"q1": [{"doc_id": "A", "top_chunks": []}]})["q1"][0]["top_chunks"] == []
    print("Test 7 OK - with_doc_name ghép tên, không sửa dict gốc, chịu được thiếu name")

    print("OK - rerank_from_d.py hoạt động đúng")
