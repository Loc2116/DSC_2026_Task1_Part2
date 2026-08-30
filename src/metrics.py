"""
metrics.py
Tính Recall@K và Precision@K theo đúng công thức chính thức DSC2026 Task 1.

Recall(câu)    = |gold ∩ predicted_top_k| / |gold|
Precision(câu) = |gold ∩ predicted_top_k| / |predicted_top_k|   (0 nếu predicted rỗng)

Điểm cuối cùng = trung bình cộng qua toàn bộ câu hỏi (macro average).
"""

from typing import Sequence


def _reject_bare_string(**kwargs) -> None:
    """
    Chặn lỗi chí tử: string là iterable trong Python, nên nếu lỡ truyền thẳng
    1 doc_id dạng string (quên bọc list) thay vì list, gold/predicted sẽ bị
    tách thành list từng KÝ TỰ một cách ÂM THẦM, không hề raise lỗi, và
    recall/precision sẽ tính sai hoàn toàn mà không có dấu hiệu gì để phát hiện.
    """
    for name, value in kwargs.items():
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"{name} phải là list các doc_id, không phải string/bytes trực tiếp: {value!r}. "
                f"String sẽ bị Python tách thành list ký tự -> tính sai âm thầm. "
                f"Có phải bạn quên bọc list, ví dụ ['{value}'] thay vì '{value}'?"
            )


def _validate_k(k) -> None:
    """
    Chặn lỗi chí tử: k <= 0 hoặc âm không hề raise lỗi khi dùng để cắt list
    (predicted[:k]), mà Python âm thầm hiểu theo nghĩa slicing khác hẳn ý định
    (vd k=0 -> list rỗng, k=-1 -> bỏ mất phần tử CUỐI thay vì "lấy top -1"),
    khiến recall/precision bị tính sai mà không có dấu hiệu gì để phát hiện.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(
            f"k phải là số nguyên dương (số lượng top-k), nhận được k={k!r}. "
            f"k<=0 hoặc âm sẽ khiến predicted[:k] cắt sai vị trí một cách ÂM THẦM "
            f"thay vì báo lỗi."
        )


def recall_at_k(gold: Sequence[str], predicted: Sequence[str], k: int = 5) -> float:
    """Recall cho 1 câu hỏi."""
    _reject_bare_string(gold=gold, predicted=predicted)
    _validate_k(k)
    gold_set = set(str(g) for g in gold)
    if not gold_set:
        return 0.0
    pred_set = set(str(p) for p in predicted[:k])
    return len(gold_set & pred_set) / len(gold_set)


def precision_at_k(gold: Sequence[str], predicted: Sequence[str], k: int = 5) -> float:
    """Precision cho 1 câu hỏi. Không trả về gì -> 0 theo quy định BTC."""
    _reject_bare_string(gold=gold, predicted=predicted)
    _validate_k(k)
    pred_k = [str(p) for p in predicted[:k]]
    if not pred_k:
        return 0.0
    gold_set = set(str(g) for g in gold)
    # QUAN TRỌNG: dùng set-intersection đúng công thức chính thức (|gold ∩ predicted_top_k|),
    # KHÔNG dùng sum(1 for p in pred_k if p in gold_set). Nếu predicted (trước khi qua
    # guard_one khử trùng lặp, vd khi evaluate() được gọi trực tiếp trên output thô của
    # retriever để debug) lỡ có id trùng lặp mà id đó trùng luôn với gold, cách đếm theo
    # từng phần tử sẽ đếm 2 lần cho cùng 1 doc đúng -> precision bị THỔI PHỒNG một cách
    # ÂM THẦM (không raise gì), làm tưởng model tốt hơn thực tế mà không có dấu hiệu gì
    # để phát hiện.
    hit = len(gold_set & set(pred_k))
    return hit / len(pred_k)


def evaluate(gold_dict: dict, predicted_dict: dict, k: int = 5) -> dict:
    """
    Chấm toàn bộ tập câu hỏi.

    gold_dict:      {"question_id": ["gold_doc_id", ...], ...}
    predicted_dict: {"question_id": ["doc_id", ...], ...}   (danh sách đã sắp theo score giảm dần)

    Trả về dict gồm: recall (macro), precision (macro), và chi tiết từng câu (per_question)
    để dễ soi ca khó sau này (việc E4).
    """
    _validate_k(k)

    # Chuẩn hóa key về str -- nếu không làm bước này, key kiểu int ở 1 trong 2 dict
    # (rất dễ xảy ra nếu ai đó build dict bằng int(qid) ở đâu đó) sẽ làm dict.get()
    # luôn trả về [] một cách ÂM THẦM, không hề raise lỗi, và điểm sẽ bị tính sai
    # hoàn toàn mà không có dấu hiệu gì để phát hiện.
    gold_dict = {str(k_): v for k_, v in gold_dict.items()}
    predicted_dict = {str(k_): v for k_, v in predicted_dict.items()}

    # Chặn lỗi chí tử: gold_dict rỗng sẽ làm macro_recall = sum(...) / 0 crash với
    # ZeroDivisionError khó hiểu, thay vì thông báo rõ ràng lý do. Trường hợp này dễ
    # xảy ra thật: vd gọi lại evaluate() trên chính tập hard_cases() để đo lại sau khi
    # sửa xong -- nếu đã sửa hết (hard_cases rỗng), đó là tín hiệu TỐT chứ không phải lỗi,
    # nhưng code không được crash mà phải báo rõ để người dùng biết không có gì để chấm.
    if not gold_dict:
        raise ValueError(
            "gold_dict rỗng -- không có câu hỏi nào để chấm. Nếu bạn đang gọi evaluate() "
            "trên tập hard_cases() và nó rỗng, nghĩa là mọi câu đã đạt recall=1.0 (tín hiệu "
            "tốt), không phải lỗi hệ thống -- chỉ đừng gọi evaluate() trên tập rỗng."
        )

    # Sanity check: nếu 2 tập key gần như không giao nhau dù cả 2 dict đều không rỗng,
    # gần như chắc chắn là bug (lệch key / lệch nguồn dữ liệu), không phải hệ thống dở.
    if gold_dict and predicted_dict:
        overlap = set(gold_dict.keys()) & set(predicted_dict.keys())
        if len(overlap) == 0:
            raise ValueError(
                "Không có câu hỏi nào trùng key giữa gold_dict và predicted_dict. "
                "Khả năng cao là lỗi định dạng ID câu hỏi (int vs str, thừa/thiếu khoảng trắng...), "
                "không phải do hệ thống retrieval dở. Kiểm tra lại nguồn dữ liệu trước khi tin vào recall."
            )

    per_question = {}
    for qid, gold in gold_dict.items():
        predicted = predicted_dict.get(qid, [])
        try:
            r = recall_at_k(gold, predicted, k)
            p = precision_at_k(gold, predicted, k)
        except TypeError as e:
            raise TypeError(f"Câu hỏi '{qid}': {e}") from e
        per_question[qid] = {"recall": r, "precision": p, "n_gold": len(gold)}

    n = len(gold_dict)
    macro_recall = sum(v["recall"] for v in per_question.values()) / n
    macro_precision = sum(v["precision"] for v in per_question.values()) / n

    return {
        "recall": macro_recall,
        "precision": macro_precision,
        "n_questions": n,
        "per_question": per_question,
    }


def hard_cases(eval_result: dict, threshold: float = 1.0) -> list[str]:
    """
    Lấy danh sách question_id có recall < threshold (mặc định: tìm mọi câu chưa đạt 1.0).
    Dùng cho E4 -- phân tích ca khó.
    """
    return [
        qid
        for qid, v in eval_result["per_question"].items()
        if v["recall"] < threshold
    ]


if __name__ == "__main__":
    # test 1: cả 2 câu đều đúng hết -> recall = 1.0
    gold = {"q1": ["54776", "72534"], "q2": ["177504"]}
    pred = {
        "q1": ["11111", "54776", "22222", "33333", "72534"],
        "q2": ["99999", "88888", "177504", "77777", "66666"],
    }
    result = evaluate(gold, pred, k=5)
    assert result["recall"] == 1.0, f"expected 1.0, got {result['recall']}"

    # test 2: câu 2-gold chỉ trúng 1/2 -> recall câu đó = 0.5
    gold2 = {"qA": ["54776", "72534"]}
    pred2 = {"qA": ["54776", "11111", "22222", "33333", "44444"]}  # thiếu 72534
    result2 = evaluate(gold2, pred2, k=5)
    assert result2["recall"] == 0.5, f"expected 0.5 (partial), got {result2['recall']}"

    # test 3: câu hoàn toàn không có prediction (D quên trả) -> recall = 0, không crash
    gold3 = {"qB": ["177504"]}
    pred3 = {}  # không có key "qB" luôn
    result3 = evaluate(gold3, pred3, k=5)
    assert result3["recall"] == 0.0

    # test 4: hard_cases lọc đúng câu chưa đạt recall = 1.0
    combined_gold = {**gold, **gold2, **gold3}
    combined_pred = {**pred, **pred2, **pred3}
    combined = evaluate(combined_gold, combined_pred, k=5)
    hc = hard_cases(combined, threshold=1.0)
    assert set(hc) == {"qA", "qB"}, f"expected qA,qB, got {hc}"

    # test 5: key câu hỏi lệch kiểu (int vs str) -- ĐÂY LÀ BUG CHÍ MẠNG VỪA TÌM RA.
    # Trước khi fix: recall bị tính = 0.0 dù predicted đúng 100%, không hề có exception.
    # Sau khi fix: evaluate() tự chuẩn hóa key về str nên vẫn chấm đúng.
    gold_int_key_test = {"999": ["abc"]}
    pred_int_key_test = {999: ["abc", "d", "e", "f", "g"]}  # key là int, cố ý
    result5 = evaluate(gold_int_key_test, pred_int_key_test, k=5)
    assert result5["recall"] == 1.0, (
        f"BUG TÁI PHÁT: key lệch kiểu (int vs str) khiến recall tính sai, "
        f"expected 1.0 got {result5['recall']}"
    )

    # test 6: gold và predicted hoàn toàn không giao nhau về key -> phải raise lỗi rõ ràng
    # thay vì âm thầm trả về recall = 0.0 (làm tưởng hệ thống retrieval dở)
    try:
        evaluate({"aaa": ["x"]}, {"bbb": ["x", "y", "z", "w", "v"]}, k=5)
        raise AssertionError("Lẽ ra phải raise lỗi vì key hoàn toàn không giao nhau!")
    except ValueError as e:
        assert "không giao nhau" in str(e) or "key" in str(e).lower()

    # test 7: gold/predicted truyền nhầm string thay vì list -- ĐÂY LÀ LỖI CHÍ TỬ VỪA VÁ.
    # Trước khi fix: string bị Python tách thành list ký tự, tính sai âm thầm không raise gì.
    # Sau khi fix: phải raise TypeError rõ ràng ngay lập tức.
    try:
        recall_at_k("54776", ["11111", "54776", "22222", "33333", "44444"], k=5)
        raise AssertionError("Lẽ ra phải raise TypeError vì gold là string, không phải list!")
    except TypeError as e:
        assert "string" in str(e).lower()
    try:
        evaluate({"qX": "54776"}, {"qX": ["54776"]}, k=5)
        raise AssertionError("Lẽ ra phải raise TypeError vì gold của qX là string!")
    except TypeError as e:
        assert "qX" in str(e)
    print("Test 7 OK - bắt đúng lỗi string bị coi là list ký tự")

    # test 8: predicted có id trùng lặp trùng với gold -- ĐÂY LÀ LỖI PRECISION VỪA VÁ.
    # Trước khi fix: hit đếm theo từng phần tử (kể cả trùng lặp) -> precision bị thổi
    # phồng cao hơn thực tế. Sau khi fix: hit = |gold ∩ set(predicted)|, chỉ đếm 1 lần.
    gold8 = ["a"]
    pred8 = ["a", "a", "b", "c", "d"]  # "a" bị lặp 2 lần
    p8 = precision_at_k(gold8, pred8, k=5)
    assert p8 == 1 / 5, f"BUG TÁI PHÁT: precision bị thổi phồng do đếm trùng lặp, expected {1/5}, got {p8}"
    print("Test 8 OK - precision không còn bị thổi phồng khi predicted có id trùng lặp")

    # test 9: k <= 0 phải raise lỗi rõ ràng, không được âm thầm cắt sai predicted[:k]
    for bad_k in (0, -1, -5):
        try:
            recall_at_k(["a"], ["a", "b", "c"], k=bad_k)
            raise AssertionError(f"Lẽ ra phải raise ValueError vì k={bad_k}!")
        except ValueError as e:
            assert "k" in str(e).lower()
    print("Test 9 OK - bắt đúng lỗi k không hợp lệ (k<=0) trong recall_at_k/precision_at_k")

    # test 10: gold_dict rỗng (vd gọi evaluate() trên tập hard_cases đã rỗng vì hết ca khó)
    # phải raise lỗi rõ ràng thay vì crash ZeroDivisionError khó hiểu.
    try:
        evaluate({}, {}, k=5)
        raise AssertionError("Lẽ ra phải raise ValueError vì gold_dict rỗng!")
    except ValueError as e:
        assert "rỗng" in str(e)
    print("Test 10 OK - bắt đúng lỗi gold_dict rỗng thay vì ZeroDivisionError")

    print("Recall (test 1):", result["recall"])
    print("Recall (test 2, partial):", result2["recall"])
    print("Recall (test 3, missing prediction):", result3["recall"])
    print("Hard cases (test 4):", hc)
    print("Recall (test 5, int key vs str key):", result5["recall"])
    print("Test 6 OK - bắt đúng lỗi key không giao nhau")
    print("OK - metrics.py hoạt động đúng, đã test cả case đúng nửa, thiếu prediction, và lệch kiểu key")
