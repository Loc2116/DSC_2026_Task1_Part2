"""
rerank_qwen.py -- reranker họ Qwen (Qwen3-Reranker, Prism-Qwen...) và mọi LLM decoder
chấm bằng cơ chế P(token "yes").

Tách riêng vì chúng KHÔNG phải cross-encoder cổ điển: không có đầu phân loại,
điểm = P(token "yes") ở vị trí cuối. Không gọi trực tiếp file này -- dùng
rerank.load_reranker(), nó tự nhận diện tên model rồi gọi sang đây.

23/08: thêm `load_4bit=True` (NF4, chạy được model to trên T4) và khuôn `template="instruct"`
cho model chat thường.

⛔ 23/08 SỬA GẤP: bản trưa nay cho `load_4bit=True` ĐI VÒNG QUA ngân sách 3B vì tưởng đó
là giới hạn bộ nhớ T4. SAI — **3B là LUẬT BTC**: tổng đội 4B tham số, D đã lấy 1B, E còn 3B.
Đã chạy 8B (8,19B) và 4B (4,02B) trên hard15, cả hai đều VƯỢT TRẦN, kết quả không dùng
cho bài nộp được. Giờ ngân sách áp **không ngoại lệ**, kể cả khi lượng tử hoá.

Cái bẫy làm hỏng phép kiểm cũ: dưới NF4, bitsandbytes gói 2 trọng số vào 1 byte `uint8`
nên `sum(p.numel())` báo **chưa tới một nửa** số tham số thật (Qwen3-Reranker-4B báo
2,205 tỉ cho một model 4,02 tỉ). Đếm kiểu cũ thì model vượt trần vẫn lọt. `_real_params`
nhân đôi phần `uint8` để trả về số thật.
"""

# LUẬT BTC, KHÔNG PHẢI GIỚI HẠN PHẦN CỨNG. Đừng nới, đừng cho tham số nào đi vòng qua.
# Tổng đội 4B tham số · D giữ 1B (khâu truy hồi) · E được 3B (khâu xếp hạng).
BUDGET = 3_000_000_000

_INSTRUCT = "Given a Vietnamese legal question, retrieve the legal document that answers it"

# Hai khuôn prompt. Cả hai đều kết thúc ngay trước lượt của assistant, nên logit ở vị trí
# CUỐI chính là phân phối của token đầu tiên model sắp nói -> so "yes" với "no" ở đó.
_TPL = {
    # Khuôn chính thức của Qwen3-Reranker.
    "qwen3_reranker": lambda q, d: (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
        f"<Instruct>: {_INSTRUCT}\n<Query>: {q}\n<Document>: {d}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    # Cho model chat thường: hỏi thẳng bằng tiếng Việt, ép trả lời một từ.
    "instruct": lambda q, d: (
        "<|im_start|>system\nBạn là trợ lý tra cứu văn bản pháp luật Việt Nam.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Câu hỏi: {q}\n\nVăn bản:\n{d}\n\n"
        "Văn bản trên có chứa câu trả lời cho câu hỏi không? "
        'Chỉ trả lời đúng một từ: "yes" hoặc "no".<|im_end|>\n<|im_start|>assistant\n'
    ),
}


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _real_params(model, torch) -> int:
    """Số tham số THẬT, đúng cả khi đã lượng tử hoá 4 bit.

    NF4 gói 2 trọng số vào 1 byte uint8 -> `p.numel()` của lớp đã nén chỉ bằng một nửa.
    Embedding và lm_head không bị nén nên đếm bình thường. Không nhân đôi phần uint8 thì
    một model 4,02 tỉ báo về 2,205 tỉ và lọt qua trần 3 tỉ.
    """
    return sum(p.numel() * (2 if p.dtype == torch.uint8 else 1) for p in model.parameters())


def _fmt(query: str, doc: str, instruct: str = _INSTRUCT) -> str:
    return _TPL["qwen3_reranker"](query, doc)


class QwenReranker:
    """
    Wrapper cho LLM decoder chấm bằng P("yes") -> phơi ra .predict(pairs) y như CrossEncoder,
    nên cắm thẳng vào score_docs_from_d / rerank_* mà không sửa gì khác.

    Vì sao cần: CrossEncoder của sentence-transformers gắn đầu phân loại ngẫu nhiên
    lên model không có sẵn -> chạy được nhưng ĐIỂM LÀ RÁC, không hề báo lỗi.
    """

    def __init__(self, model_name, device="cuda", max_length=1024, batch_size=8, dtype=None,
                 load_4bit=False, template="qwen3_reranker", device_map=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.max_length, self.batch_size = torch, device, max_length, batch_size
        self.fmt = _TPL[template]
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")  # last-token logit
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if dtype is None:
            # KHÔNG dùng float16: Qwen huấn luyện ở bfloat16 (dải động rộng hơn nhiều).
            # Ép xuống fp16 gây tràn số -> logit bão hoà -> mọi cặp ~cùng điểm.
            # Triệu chứng đã gặp thật: Qwen3-0.6B fp16 chỉ đạt 0.7711 (BM25 thuần 0.7578).
            # T4 (Turing) không có bf16 phần cứng nhưng torch vẫn chạy được, chậm hơn.
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        print(f"[QwenReranker] {model_name} · dtype={dtype} · 4bit={load_4bit} · tpl={template}")
        if load_4bit:
            # NF4 giữ trọng số 4 bit nhưng GIẢI NÉN ra dtype ở trên để tính -> tránh đúng
            # cái bẫy fp16 nói trên, mà 8B vẫn chỉ chiếm ~6GB, vừa một T4.
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            # device_map="auto" -> accelerate TRẢI model qua MỌI GPU đang có.
            # BẪY 25/08: `{"": 0}` nhồi hết vào cuda:0. Trên T4 x2 nghĩa là chỉ dùng 14,5GB
            # trong khi có 29GB, và bản `transformers` mới nạp tốn hơn bản cũ -> OOM ngay
            # lúc nạp, dù 8B NF4 chỉ ~7GB. Lượt 23/08 lọt được là nhờ môi trường CŨ.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=qc,
                device_map=device_map or {"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype).to(device).eval()

        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        unk = self.tok.unk_token_id
        if self.yes_id in (None, unk) or self.no_id in (None, unk):
            raise ValueError(f"{model_name}: không lấy được token id của 'yes'/'no' -> template không hợp")

    def predict(self, pairs) -> list[float]:
        torch = self.torch
        out: list[float] = []
        for chunk in _batches(list(pairs), self.batch_size):
            texts = [self.fmt(q, d) for q, d in chunk]
            enc = self.tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            # Cắt bớt ở ĐUÔI sẽ nuốt mất "<|im_start|>assistant" -> logit cuối thành rác mà
            # không báo lỗi. Chặn tại chỗ thay vì đi tìm nguyên nhân sau khi đã chấm xong.
            if enc["input_ids"].shape[1] >= self.max_length:
                raise ValueError(
                    f"prompt chạm trần {self.max_length} token -> khuôn bị cắt đuôi. "
                    f"Rút ngắn văn bản đầu vào hoặc tăng max_length.")
            with torch.inference_mode():
                logits = self.model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
            out.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().tolist())
        return out


def load_qwen_reranker(model_name="Qwen/Qwen3-Reranker-0.6B", device="cuda", load_4bit=False,
                       thay_offline=False, **kw):
    """`thay_offline=True`: CHỈ dùng để SINH NHÃN ngoại tuyến cho chưng cất.

    BTC đã xác nhận (25/08) chưng cất từ model lớn hơn trần là HỢP LỆ, vì khi nộp bài chỉ có
    model học trò <=3B chạy. Cửa này mở đúng cho việc đó và không cho việc gì khác.

    ⛔ TUYỆT ĐỐI KHÔNG bật cờ này trong bất kỳ notebook nào sinh ra file `submission*`.
       Model được nạp qua cửa này KHÔNG ĐƯỢC phép chấm đề thi.
    """
    if thay_offline and load_4bit:
        kw.setdefault("device_map", "auto")   # thầy to -> trải qua mọi GPU, đừng nhồi cuda:0
    m = QwenReranker(model_name, device=device, load_4bit=load_4bit, **kw)
    n = _real_params(m.model, m.torch)
    print(f"[load_qwen_reranker] {model_name}: {n:,} ({n / 1e9:.3f}B tham số thật)")
    if n > BUDGET and not thay_offline:
        raise ValueError(
            f"{model_name} = {n/1e9:.2f}B tham số, VƯỢT ngân sách {BUDGET/1e9:.0f}B của E.\n"
            f"Đây là LUẬT BTC (tổng đội 4B, D giữ 1B), không phải giới hạn phần cứng — "
            f"lượng tử hoá KHÔNG nới được. Chọn model nhỏ hơn.\n"
            f"(Sinh nhãn ngoại tuyến cho chưng cất: đặt thay_offline=True, ĐỌC docstring trước.)")
    if n > BUDGET:
        print("\n" + "!" * 76)
        print(f"!! MODEL THẦY {n/1e9:.2f}B — VƯỢT TRẦN 3B, ĐANG CHẠY Ở CHẾ ĐỘ SINH NHÃN NGOẠI TUYẾN.")
        print("!! Đầu ra của nó CHỈ được dùng làm nhãn huấn luyện.")
        print("!! Nếu notebook này sinh ra file submission thì BÀI NỘP KHÔNG HỢP LỆ. Dừng lại.")
        print("!" * 76 + "\n")

    # Sanity check: template sai -> mọi cặp cùng điểm. Rẻ, chạy 1 lần, chặn 1 lượt GPU vứt đi.
    hop, sai = m.predict([
        ["Thủ tục đăng ký xe máy?", "Điều 1. Hồ sơ đăng ký xe mô tô, xe gắn máy gồm giấy tờ sau"],
        ["Thủ tục đăng ký xe máy?", "Điều 5. Tiêu chuẩn quốc gia về phương pháp thử nghiệm hạt giống lúa"],
    ])
    print(f"[sanity] liên quan={hop:.4f}  không liên quan={sai:.4f}  (chênh {hop - sai:.4f})")
    if hop - sai < 0.05:
        print(
            f"  [!] Chênh lệch quá nhỏ ({hop - sai:.4f}). Model gần như không phân biệt được "
            f"cặp rõ ràng liên quan với cặp hoàn toàn lạc đề -- nghi tràn số (dtype) hoặc sai "
            f"template. Điểm sẽ dồn sát nhau và xếp hạng gần như vô nghĩa."
        )
    if hop <= sai:
        raise ValueError(
            f"{model_name}: cặp LIÊN QUAN ({hop:.4f}) không cao hơn cặp KHÔNG liên quan ({sai:.4f}). "
            f"Gần như chắc chắn sai template prompt -- đừng chạy cả bộ, sẽ ra điểm rác."
        )
    return m



if __name__ == "__main__":
    assert list(_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(_batches([], 4)) == []
    print("Test _batches OK")

    p = _fmt("câu hỏi", "văn bản")
    assert p.startswith("<|im_start|>system") and p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<Query>: câu hỏi" in p and "<Document>: văn bản" in p
    print("Test _fmt OK - prompt đúng khung Qwen3-Reranker")

    # Cả hai khuôn phải kết thúc ở lượt assistant, nếu không thì logit cuối chấm nhầm chỗ.
    for k, f in _TPL.items():
        s = f("hỏi", "đáp")
        assert "<|im_start|>assistant" in s and s.index("<|im_start|>assistant") > s.index("đáp"), k
        assert "hỏi" in s and "đáp" in s, k
    print("Test _TPL OK - cả hai khuôn kết thúc sau <|im_start|>assistant")

    import math
    assert abs(math.exp(2.0) / (math.exp(2.0) + math.exp(0.5)) - 0.8176) < 1e-3
    print("Test công thức P(yes) OK")

    # Ngân sách BTC: phải bắt được model 4-bit vượt trần. Dựng lại ĐÚNG số của
    # Qwen3-Reranker-4B trong log 23/08 (báo 2.205.126.656 khi NF4, thật là 4,02B).
    class _T:  uint8 = "uint8"; float16 = "float16"
    class _P:
        def __init__(s, n, d): s._n, s.dtype = n, d
        def numel(s): return s._n
    class _M:
        def __init__(s, ps): s._p = ps
        def parameters(s): return s._p
    emb, quant = 389_000_000, 1_816_126_656          # không nén / đã nén (numel một nửa)
    m4 = _M([_P(emb, _T.float16), _P(quant, _T.uint8)])
    assert sum(p.numel() for p in m4.parameters()) == 2_205_126_656, "chưa khớp log 23/08"
    assert _real_params(m4, _T) == emb + 2 * quant == 4_021_253_312
    assert _real_params(m4, _T) > BUDGET, "model 4,02B PHẢI bị chặn bởi trần 3B"
    m06 = _M([_P(100_000_000, _T.float16), _P(248_000_000, _T.uint8)])
    assert _real_params(m06, _T) <= BUDGET, "model 0,6B không được bị chặn oan"
    print("Test ngân sách OK - đếm đúng tham số thật dưới NF4, chặn 4,02B, tha 0,6B")

    print("OK - rerank_qwen.py")
