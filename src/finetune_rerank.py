"""
finetune_rerank.py -- E3: fine-tune cross-encoder trên dữ liệu BTC cấp.

===========================================================================
RÀNG BUỘC QUY CHẾ -- đọc trước khi sửa
===========================================================================
BTC: chỉ được dùng dữ liệu BTC cấp, KHÔNG được dùng kỹ thuật tăng cường dữ liệu.

File này tuân thủ bằng ba chốt chặn tự động (hàm `_assert_hop_le`):
  1. Mọi câu hỏi huấn luyện phải có question_id nằm trong train.json
  2. Mọi văn bản (đúng lẫn sai) phải có doc_id tồn tại trong selected-contexts/
  3. Không sinh, không diễn giải lại, không dịch xuôi ngược -- không có chỗ nào
     trong file này tạo ra chữ mới; mọi text đều đọc thẳng từ corpus

Hard negative mining KHÔNG phải tăng cường dữ liệu: nó chọn văn bản có sẵn của
BTC làm ví dụ sai, không tạo dữ liệu mới. Chọn negative là bước bắt buộc của mọi
quy trình huấn luyện reranker -- không chọn hard thì vẫn phải chọn random.

===========================================================================
BẪY CHẾT NGƯỜI: RÒ RỈ DEV
===========================================================================
dev_150_locked.json được cắt từ train.json. Huấn luyện trên cả 7000 câu = model
đã thấy đáp án của 150 câu dev -> recall@5 phồng lên giả tạo, MẤT LUÔN thước đo,
và không có gì báo lỗi. Cùng họ với vụ nộp nhầm dev hôm 09/08.

=> Mặc định loại `dev_1000_locked.json`, huấn luyện trên 6000 câu.
   dev1000 BAO TRÙM dev300 và dev150, nên loại một file là loại hết.

===========================================================================
Cách dùng
===========================================================================
    # negative ngẫu nhiên (chắc chắn hợp lệ, yếu hơn)
    python finetune_rerank.py --out ft_model

    # hard negative (mạnh hơn, cần file candidate của D hoặc bản _FALLBACK)
    python finetune_rerank.py --candidates candidates_public_FB.json --out ft_model

Xong thì trỏ RERANKER_MODEL trong notebook vào thư mục ft_model.
"""

import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_candidates_fallback import read_passage, tok   # dùng lại, không chép
import deep_chunk as DC                                  # MỘT cửa chọn đoạn, chung với inference

_rel = lambda p: p if os.path.isabs(p) else os.path.join(HERE, p)


# --------------------------------------------------------------------------
# Dựng dữ liệu -- phần này test được mà không cần torch
# --------------------------------------------------------------------------
def best_chunk(question: str, ctx_dir: str, doc_id: str) -> str:
    """Đoạn dùng làm ví dụ huấn luyện -- PHẢI là đúng hàm chọn của lúc suy luận.

    ===========================================================================
    18/08 -- ĐÂY LÀ LÝ DO LƯỢT FINE-TUNE 13/08 THUA
    ===========================================================================
    Bản cũ:

        parts = chunks_of(passage)
        return max(parts, key=lambda c: len(qs & set(tok(c))), default="")

    Đếm từ trùng thô, không IDF, không chuẩn hoá độ dài, không gộp đoạn. Đúng
    hàm mà 17/08 đo được là kém BM25 mức đoạn 1,34 điểm dev300 -- xem bảng trong
    docstring của deep_chunk.py.

    Hậu quả có hai tầng:

    1. BẰNG CHỨNG DƯƠNG BỊ NHIỄU. Với nhóm câu mà cross-encoder chấm ~0,00 trên
       toàn bộ văn bản gold (7/16 ca kẹt trên dev300), đoạn được chọn làm positive
       là dòng MỤC LỤC không chứa câu trả lời. Model bị dạy "dòng mục lục này trả
       lời câu hỏi này" -- dạy nhiễu, đúng chế độ hỏng đang thấy lúc suy luận.

    2. TRAIN LỆCH INFERENCE. Lúc chấm ta đưa đoạn BM25 gộp ~1.460 ký tự; lúc
       huấn luyện ta đưa đoạn term-overlap ~650 ký tự. Model chưa từng thấy loại
       đầu vào nó phải chấm.

    Nên gọi thẳng DC.pick_chunks, KHÔNG chép lại logic -- chép là hai bên trôi
    khỏi nhau, mà trôi khỏi nhau chính là bug này.

    Dùng doc_id thay vì passage để ăn `DC._cache`: 30.000 lượt gọi mà băm lại mỗi
    lần thì mất ~50 phút CPU. Nhớ DC._cache.clear() sau khi dựng xong.
    """
    sel = DC.pick_chunks(question, ctx_dir, doc_id, k=1)
    return sel[0] if sel else ""


def _assert_hop_le(qid, doc_ids, train_qids, corpus_ids):
    if qid not in train_qids:
        raise ValueError(f"question_id {qid!r} không có trong train.json -- dữ liệu ngoài luồng BTC")
    for d in doc_ids:
        if d not in corpus_ids:
            raise ValueError(f"doc_id {d!r} không có trong selected-contexts/ -- dữ liệu nguồn khác")


def build_pairs(
    train: dict,
    dev_qids: set,
    corpus_ids: set,
    ctx_dir: str,
    candidates: dict | None = None,
    n_neg: int = 4,
    seed: int = 0,
    limit: int = 0,
):
    """
    Trả về list (question, doc_text, label). label 1 = gold, 0 = negative.

    candidates=None  -> negative ngẫu nhiên trong corpus
    candidates=dict  -> HARD negative: lấy từ top-100 của câu đó, bỏ gold ra
    """
    rng = random.Random(seed)
    train_qids = set(train)
    pool = sorted(corpus_ids)          # sorted -> tái lập được với cùng seed
    qids = [q for q in train if q not in dev_qids]
    if len(qids) != len(train) - len(dev_qids & train_qids):
        raise AssertionError("lọc dev không khớp")
    if limit:
        qids = qids[:limit]

    pairs, n_hard = [], 0
    for qid in qids:
        item = train[qid]
        question = item["question"]
        gold = [str(g) for g in item["answer"]]
        _assert_hop_le(qid, gold, train_qids, corpus_ids)

        cand = [str(c["doc_id"]) for c in candidates.get(qid, [])] if candidates else []
        negs = [d for d in cand if d not in gold][:n_neg]
        n_hard += len(negs)
        while len(negs) < n_neg:       # thiếu thì bù ngẫu nhiên
            d = rng.choice(pool)
            if d not in gold and d not in negs:
                negs.append(d)
        _assert_hop_le(qid, negs, train_qids, corpus_ids)

        for d in gold:
            pairs.append((question, best_chunk(question, ctx_dir, d), 1))
        for d in negs:
            pairs.append((question, best_chunk(question, ctx_dir, d), 0))

    print(f"  {len(qids):,} câu -> {len(pairs):,} cặp | hard negative: {n_hard:,}/{len(qids)*n_neg:,}")
    return pairs


# --------------------------------------------------------------------------
# Huấn luyện
# --------------------------------------------------------------------------
def train_model(pairs, base_model, out_dir, epochs=1, bs=8, lr=2e-5, max_length=512, device="cuda"):
    """
    T4 16GB: bs=16 fp32 là OOM (đã dính 12/08). 568M tham số ăn ~9,1GB chỉ riêng
    weight + grad + trạng thái AdamW, còn ~5,4GB cho activation -> không đủ.
    => bs=8 + AMP fp16. AMP giảm nửa activation và nhanh ~3x nhờ tensor core.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokzr = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def collate(batch):
        q, d, y = zip(*batch)
        enc = tokzr(list(q), list(d), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
        return enc, torch.tensor(y, dtype=torch.float)

    dl = DataLoader(pairs, batch_size=bs, shuffle=True, collate_fn=collate)
    total = len(dl) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)

    model.train()
    step = 0
    for ep in range(epochs):
        for enc, y in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = lossf(model(**enc).logits.squeeze(-1), y.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                       # phải unscale trước khi clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30 if amp else 0
                print(f"  epoch {ep+1} step {step}/{total} loss {loss.item():.4f} | VRAM đỉnh {mem:.1f}GB", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokzr.save_pretrained(out_dir)
    n = sum(p.numel() for p in model.parameters())
    print(f"Đã lưu {out_dir} | {n:,} tham số ({n/1e9:.3f}B)")
    if n > 3_000_000_000:
        raise ValueError("vượt ngân sách 3B")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Input/train.json")
    ap.add_argument("--dev", default="dev_1000_locked.json",
                    help="BẮT BUỘC. Mặc định dev1000 vì nó BAO TRÙM cả dev300 lẫn dev150 "
                         "-> loại một file là loại hết, không thể quên tập nào.")
    ap.add_argument("--contexts", default="Input/selected-contexts")
    ap.add_argument("--candidates", default=None, help="file top-100 -> bật hard negative")
    ap.add_argument("--base", default="AITeamVN/Vietnamese_Reranker")
    ap.add_argument("--out", default="ft_model")
    ap.add_argument("--merge-chars", type=int, default=1800,
                    help="gộp đoạn tới ~N ký tự. PHẢI khớp cấu hình inference "
                         "(final_v2_run.ipynb dùng 1800). 0 = giữ chunks_of thô.")
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8, help="8 là mức vừa T4 16GB với AMP. 16 = OOM")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu, để thử nhanh")
    ap.add_argument("--dry-run", action="store_true", help="chỉ dựng dữ liệu, không huấn luyện")
    a = ap.parse_args()

    DC.MERGE_CHARS = a.merge_chars          # đặt TRƯỚC mọi lời gọi -- nó đổi cache
    ctx = _rel(a.contexts)
    train = json.load(open(_rel(a.train), encoding="utf-8"))
    dev_qids = set(json.load(open(_rel(a.dev), encoding="utf-8")))
    corpus_ids = {
        f[len("context_"):-len(".json")] for f in os.listdir(ctx) if f.startswith("context_")
    }
    cands = json.load(open(_rel(a.candidates), encoding="utf-8")) if a.candidates else None

    print(f"train {len(train):,} câu | dev loại ra {len(dev_qids)} | corpus {len(corpus_ids):,} văn bản")
    print(f"negative: {'HARD (từ ' + a.candidates + ')' if cands else 'NGẪU NHIÊN'}")
    print(f"chọn đoạn: DC.pick_chunks BM25, MERGE_CHARS={DC.MERGE_CHARS} (phải khớp inference)")

    pairs = build_pairs(train, dev_qids, corpus_ids, ctx, cands, a.n_neg, limit=a.limit)

    leak = dev_qids & {q for q in train if q not in dev_qids}
    assert not leak, f"RÒ RỈ DEV: {leak}"
    print("  [ok] không có câu dev nào lọt vào tập huấn luyện")

    n_cache = len(DC._cache)
    DC._cache.clear()                       # ~9GB nếu phủ hết corpus -- trả lại trước khi train
    print(f"  đã băm {n_cache:,} văn bản, xoá cache")

    if a.dry_run:
        print("\n--dry-run: dừng ở đây. Ví dụ 1 cặp:")
        q, d, y = pairs[0]
        print(f"  label={y} | Q: {q[:70]}\n           D: {d[:120]}")
        return

    train_model(pairs, a.base, _rel(a.out), a.epochs, a.bs, a.lr, a.max_length, a.device)
    print("\nXong. Đổi RERANKER_MODEL trong notebook thành đường dẫn thư mục này rồi đo lại trên dev.")


if __name__ == "__main__":
    main()
