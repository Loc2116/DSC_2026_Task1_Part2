"""
submission.py
Guard + đóng gói submission.json đúng format BTC.

QUAN TRỌNG: vượt quá 5 id ở BẤT KỲ câu nào -> toàn bộ submission = 0 điểm.
Guard này PHẢI chạy trước mọi lần nộp bài, không có ngoại lệ.
"""

import json
import zipfile
from pathlib import Path


def guard_one(predicted: list, ranked_fallback: list | None = None, k: int = 5) -> list[str]:
    """
    Chuẩn hóa danh sách dự đoán cho 1 câu hỏi:
      1. Khử trùng lặp, giữ nguyên thứ tự ưu tiên
      2. Nếu thiếu (< k), bù thêm từ ranked_fallback (toàn bộ candidate đã rank,
         thường dài hơn top-5, ví dụ top-100 của D) cho đủ k
      3. Cắt đúng k, luôn ép về dạng string

    predicted:        top-k hiện có (đã rank), có thể bị trùng hoặc thiếu
    ranked_fallback:   toàn bộ danh sách đã rank (để bù nếu thiếu). Nếu None thì
                        không bù được -- sẽ raise lỗi nếu vẫn thiếu sau khử trùng lặp.
    """
    # Chặn lỗi chí tử: k <= 0 hoặc âm không hề raise lỗi khi dùng để cắt list
    # (seen[:k] ở cuối hàm) -- Python âm thầm hiểu theo nghĩa slicing khác hẳn ý
    # định (vd k=-1 sẽ bỏ mất id CUỐI CÙNG thay vì lấy đủ k id), khiến submission
    # thiếu đúng 1 id một cách ÂM THẦM mà không hề báo lỗi.
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(
            f"k phải là số nguyên dương, nhận được k={k!r}. k<=0 hoặc âm sẽ khiến "
            f"seen[:k] cắt sai một cách ÂM THẦM thay vì báo lỗi."
        )

    # Chặn lỗi chí tử: string là iterable trong Python. Nếu lỡ quên bọc list
    # (vd truyền "123456" thay vì ["123456"]), predicted sẽ bị tách thành
    # list từng KÝ TỰ một cách ÂM THẦM -- không raise gì, và nếu chuỗi đó có
    # đủ >=k ký tự khác nhau (rất dễ xảy ra với id 5-6 chữ số), guard sẽ trả
    # về 5 ký tự rác làm id tài liệu mà không có cách nào phát hiện trước khi nộp.
    for name, value in (("predicted", predicted), ("ranked_fallback", ranked_fallback)):
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"{name} phải là list các doc_id, không phải string/bytes trực tiếp: {value!r}. "
                f"String sẽ bị tách thành list ký tự -> nộp bài rác mà không hề báo lỗi. "
                f"Có phải bạn quên bọc list, ví dụ ['{value}'] thay vì '{value}'?"
            )
        if isinstance(value, (set, frozenset)):
            raise TypeError(
                f"{name} là set -- set không giữ thứ tự, trong khi guard_one cần list/tuple "
                f"ĐÃ RANK theo độ liên quan giảm dần. Xáo trộn thứ tự trước khi cắt top-k có thể "
                f"loại nhầm id đúng khỏi top-k một cách âm thầm. Giữ nguyên dạng list (guard_one "
                f"đã tự khử trùng lặp, không cần set() trước)."
            )
        # Chặn lỗi chí tử: format D bàn giao là {"candidates": [...], "scores": [...]}.
        # Nếu quên .get("candidates") mà lỡ truyền thẳng cả dict này vào guard_one, dict
        # sẽ bị duyệt qua CÁC KEY ("candidates", "scores", ...) chứ không phải doc_id bên
        # trong. Nếu dict tình cờ có đủ >=k field, guard_one "thành công" mà KHÔNG hề raise
        # gì -- âm thầm nộp tên field làm doc_id rác, không có cách nào phát hiện trước khi nộp.
        if isinstance(value, dict):
            raise TypeError(
                f"{name} là dict, không phải list doc_id. Có phải bạn quên trích candidate list "
                f"ra trước, ví dụ d_output['candidates'] thay vì truyền thẳng d_output? "
                f"Truyền thẳng dict sẽ bị duyệt qua các KEY của dict (vd 'candidates', 'scores') "
                f"làm doc_id rác mà không hề báo lỗi."
            )

    seen = []
    for p in predicted:
        p = str(p)
        if p not in seen:
            seen.append(p)

    if len(seen) < k and ranked_fallback:
        for p in ranked_fallback:
            p = str(p)
            if p not in seen:
                seen.append(p)
            if len(seen) >= k:
                break

    if len(seen) < k:
        raise ValueError(
            f"Chỉ có {len(seen)} candidate sau khử trùng lặp, thiếu {k - len(seen)} "
            f"và không có ranked_fallback đủ để bù. Kiểm tra lại nguồn candidate."
        )

    return seen[:k]


def build_submission(
    predicted_dict: dict,
    ranked_fallback_dict: dict | None = None,
    k: int = 5,
    expected_qids: set[str] | None = None,
) -> dict:
    """
    predicted_dict:        {"question_id": [doc_id, ...], ...}
    ranked_fallback_dict:  {"question_id": [doc_id, ...] (dài hơn, vd top-100), ...} hoặc None
    expected_qids:         toàn bộ id câu hỏi PHẢI có trong bài nộp (vd toàn bộ warmup.json).
                            Nếu None, không kiểm tra thiếu câu -- CHỈ bỏ qua khi bạn chắc chắn
                            predicted_dict đã đủ, vì thiếu 1 câu có thể làm cả bài 0 điểm (R1).

    Trả về dict đúng format BTC:
    {"question_id": {"answer": [doc_id x k]}, ...}
    """
    # Chuẩn hóa key câu hỏi về str -- cùng lý do như trong metrics.evaluate():
    # key lệch kiểu (int vs str) sẽ làm .get() âm thầm trả về None/[] mà không báo lỗi.
    predicted_dict = {str(k_): v for k_, v in predicted_dict.items()}
    if ranked_fallback_dict is not None:
        ranked_fallback_dict = {str(k_): v for k_, v in ranked_fallback_dict.items()}
    if expected_qids is not None:
        expected_qids = {str(q) for q in expected_qids}

    submission = {}
    errors = []
    for qid, predicted in predicted_dict.items():
        fallback = ranked_fallback_dict.get(qid) if ranked_fallback_dict else None
        try:
            top_k = guard_one(predicted, fallback, k=k)
        except ValueError as e:
            errors.append(f"{qid}: {e}")
            continue
        submission[qid] = {"answer": top_k}

    if expected_qids is not None:
        missing = set(expected_qids) - set(submission.keys())
        if missing:
            errors.append(
                f"THIẾU {len(missing)} câu hỏi hoàn toàn trong submission (rủi ro R1 -- "
                f"nộp thiếu câu = cả bài 0 điểm): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )

    if errors:
        raise ValueError("Guard thất bại:\n" + "\n".join(errors))

    return submission


def save_submission_zip(submission: dict, out_dir: str = ".", filename: str = "submission.json") -> str:
    """
    Ghi submission.json rồi nén vào submission.zip (chỉ chứa duy nhất file json này,
    đúng yêu cầu nộp bài của BTC). Trả về đường dẫn file zip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / filename
    zip_path = out_dir / "submission.zip"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=filename)

    return str(zip_path)


def validate_submission_file(path: str, expected_qids: set[str] | None = None, k: int = 5) -> list[str]:
    """
    Kiểm tra lại 1 file submission.json đã ghi ra đĩa (double-check trước khi nộp thật).
    Trả về danh sách lỗi (rỗng nếu ổn).
    """
    errors = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for qid, obj in data.items():
        answer = obj.get("answer", [])
        # Chặn lỗi chí tử: nếu submission.json này KHÔNG được ghi ra qua
        # build_submission/guard_one (vd 1 script khác tự tay json.dump doc_id dạng
        # số, như 54776 thay vì "54776"), id sẽ lọt vào dưới dạng int mà JSON không
        # hề báo lỗi gì. Lúc chấm điểm thật, so khớp string "54776" != int 54776 có
        # thể khiến id đúng bị tính là sai một cách ÂM THẦM.
        non_str = [a for a in answer if not isinstance(a, str)]
        if non_str:
            errors.append(
                f"{qid}: có id KHÔNG phải string trong file JSON ({non_str!r}) -- doc_id phải "
                f"là string. Nếu file này không được ghi qua build_submission()/guard_one(), "
                f"hãy build lại qua guard_one để đảm bảo mọi id đã được ép str()."
            )
        if len(answer) > k:
            errors.append(f"{qid}: có {len(answer)} id, vượt quá {k} -> TOÀN BỘ SUBMISSION = 0 ĐIỂM")
        if len(answer) != len(set(answer)):
            errors.append(f"{qid}: có id trùng lặp trong cùng 1 câu")
        if len(answer) == 0:
            errors.append(f"{qid}: câu trả lời rỗng")
        elif len(answer) < k:
            errors.append(
                f"{qid}: cảnh báo -- chỉ nộp {len(answer)}/{k} id. Không bị 0 điểm nhưng "
                f"đang bỏ lỡ cơ hội ăn Recall miễn phí, nên bù đủ {k}"
            )

    if expected_qids is not None:
        missing = expected_qids - set(data.keys())
        if missing:
            errors.append(f"Thiếu {len(missing)} câu hỏi trong submission: {sorted(missing)[:5]}...")
        # Chặn lỗi thật đã tìm ra: scoring.py gốc của BTC chỉ check SỐ LƯỢNG câu khớp
        # (len(ids_preds) != len(ids_truth)), không check ĐÚNG BỘ id. Nếu submission
        # có câu THỪA (không nằm trong expected_qids) trong khi vẫn thiếu 1 câu khác
        # -- số lượng có thể tình cờ KHÔNG khớp (bắt bởi check khác), nhưng nếu số
        # lượng thừa/thiếu bù trừ nhau vừa khít, sẽ crash "NoneType has no len()"
        # phía BTC khi họ tra y_pred.get(k) cho câu bị thiếu -- lỗi khó hiểu, khó debug
        # từ xa. Chặn thẳng ở đây: không cho phép có câu thừa ngay từ đầu.
        extra = set(data.keys()) - expected_qids
        if extra:
            errors.append(
                f"THỪA {len(extra)} câu hỏi không nằm trong expected_qids: {sorted(extra)[:5]}... "
                f"-- nếu đi kèm thiếu câu khác, đây là quả mìn crash phía scoring.py của BTC "
                f"(len(None) trên câu bị thiếu). Xoá câu thừa này trước khi nộp."
            )

    return errors


if __name__ == "__main__":
    # test 1: trùng lặp + thiếu -> guard tự bù, dư -> guard tự cắt
    predicted = {
        "q1": ["a", "b", "a", "c"],  # có trùng, thiếu (chỉ 3 unique)
        "q2": ["x", "y", "z", "w", "v", "u"],  # dư, phải cắt còn 5
    }
    fallback = {
        "q1": ["a", "b", "c", "d", "e", "f"],  # dùng để bù cho q1
    }
    sub = build_submission(predicted, fallback, k=5)
    for qid, obj in sub.items():
        assert len(obj["answer"]) == 5, f"{qid} không đủ 5"
        assert len(obj["answer"]) == len(set(obj["answer"])), f"{qid} bị trùng"
    print("Test 1 OK:", json.dumps(sub, ensure_ascii=False))

    # test 2: THIẾU CẢ 1 CÂU trong predicted_dict -> phải raise lỗi rõ ràng, không được im lặng bỏ qua
    expected = {"q1", "q2", "q3"}  # q3 không hề có trong predicted -- mô phỏng lỗi thật (D quên trả)
    try:
        build_submission(predicted, fallback, k=5, expected_qids=expected)
        raise AssertionError("Lẽ ra phải raise lỗi vì thiếu q3, nhưng không raise!")
    except ValueError as e:
        assert "THIẾU" in str(e)
        print("Test 2 OK - bắt đúng lỗi thiếu câu hỏi:", str(e).split(chr(10))[0])

    # test 3: predicted truyền nhầm string thay vì list -- LỖI CHÍ TỬ VỪA VÁ.
    # Trước khi fix: "123456" bị tách thành ['1','2','3','4','5'], KHÔNG raise gì,
    # âm thầm nộp id rác. Sau khi fix: phải raise TypeError ngay lập tức.
    try:
        guard_one("123456", ranked_fallback=None, k=5)
        raise AssertionError("Lẽ ra phải raise TypeError vì predicted là string, không phải list!")
    except TypeError as e:
        assert "string" in str(e).lower()
    print("Test 3 OK - bắt đúng lỗi predicted là string thay vì list")

    # test 4: predicted truyền nhầm set() -- set không giữ thứ tự, dễ làm mất đúng
    # top-k liên quan nhất mà không raise gì. Sau khi fix: phải raise TypeError.
    try:
        guard_one({"a", "b", "c", "d", "e", "f"}, k=5)
        raise AssertionError("Lẽ ra phải raise TypeError vì predicted là set, không phải list!")
    except TypeError as e:
        assert "set" in str(e).lower()
    print("Test 4 OK - bắt đúng lỗi predicted là set (mất thứ tự) thay vì list")

    # test 5: k <= 0 phải raise lỗi rõ ràng ngay trong guard_one, không được âm thầm
    # cắt sai seen[:k]
    for bad_k in (0, -1, -5):
        try:
            guard_one(["a", "b", "c", "d", "e", "f"], k=bad_k)
            raise AssertionError(f"Lẽ ra phải raise ValueError vì k={bad_k}!")
        except ValueError as e:
            assert "k" in str(e).lower()
    print("Test 5 OK - bắt đúng lỗi k không hợp lệ (k<=0) trong guard_one")

    # test 6: submission.json có id dạng số nguyên (KHÔNG phải string) -- lỗi tiềm ẩn
    # xảy ra nếu 1 script khác tự tay ghi file này mà không đi qua build_submission()/
    # guard_one() (nên không được ép str() sẵn). validate_submission_file phải bắt
    # được lỗi này trước khi nộp thật, thay vì để lọt id kiểu int gây sai lệch khi
    # so khớp ở bước chấm điểm.
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        bad_path = os.path.join(tmp, "bad_submission.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump({"q1": {"answer": [54776, "11111", "22222", "33333", "44444"]}}, f)
        errs = validate_submission_file(bad_path, k=5)
        assert any("KHÔNG phải string" in e for e in errs), f"Lẽ ra phải bắt lỗi id không phải string, got: {errs}"
    print("Test 6 OK - bắt đúng lỗi id dạng int lọt vào submission.json (không qua guard_one)")

    # test 7: predicted truyền nhầm CẢ DICT (format D bàn giao {"candidates":..,"scores":..})
    # thay vì list -- LỖI VỪA VÁ. Trước khi fix: nếu dict có đủ >=k field, guard_one
    # "thành công" mà không raise gì, nộp tên field làm doc_id rác.
    d_output = {"candidates": ["a", "b"], "scores": [0.9, 0.8]}
    try:
        guard_one(d_output, k=5)
        raise AssertionError("Lẽ ra phải raise TypeError vì predicted là dict, không phải list!")
    except TypeError as e:
        assert "dict" in str(e).lower()
    d_output_big = {"candidates": [], "scores": [], "query": "", "top_score": 0, "method": ""}
    try:
        guard_one(d_output_big, k=5)
        raise AssertionError("Lẽ ra phải raise TypeError -- dict đủ 5 key trước đây lọt qua guard!")
    except TypeError as e:
        assert "dict" in str(e).lower()
    print("Test 7 OK - bắt đúng lỗi truyền nhầm dict (quên .get('candidates')) thay vì list")

    # test 8: submission có câu THỪA (không nằm trong expected_qids) -- LỖI VỪA VÁ.
    # Đây là điều kiện có thể kích hoạt quả mìn crash "NoneType has no len()" phía
    # scoring.py gốc của BTC khi số lượng thừa/thiếu bù trừ vừa khít.
    with tempfile.TemporaryDirectory() as tmp:
        extra_path = os.path.join(tmp, "extra_submission.json")
        json.dump(
            {"q1": {"answer": ["a", "b", "c", "d", "e"]}, "q_thua": {"answer": ["a", "b", "c", "d", "e"]}},
            open(extra_path, "w"),
        )
        errs = validate_submission_file(extra_path, expected_qids={"q1"}, k=5)
        assert any("THỪA" in e for e in errs), f"Lẽ ra phải bắt câu thừa, got: {errs}"
    print("Test 8 OK - bắt đúng lỗi câu thừa (nguy cơ crash scoring.py phía BTC)")

    print("OK - submission.py hoạt động đúng, đã test cả case thiếu câu hỏi hoàn toàn (R1)")
