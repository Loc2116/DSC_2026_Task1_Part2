"""bi_encoder.py -- tầng truy hồi dày (dense) bằng AITeamVN/Vietnamese_Embedding.

Vì sao có file này
==================
Đối chiếu 19/08 với các giải pháp mạnh ở cùng bài toán: kiến trúc chuẩn là
bi-encoder (truy hồi) + cross-encoder (xếp hạng), CẢ HAI đều fine-tune.
Nhóm mình chỉ có BM25 + cross-encoder gốc -> thiếu hẳn nửa dày. Top LB 0.95 vs
mình 0.8871 = chênh 63 câu, quá lớn để vá bằng cần gạt dưới 1 điểm.

7 câu dev300 có gold NGOÀI top-100 BM25 = 2,17 điểm E hiện không chạm tới được.
Đó là thứ chỉ tầng truy hồi mới cứu được.

Chọn model: AITeamVN/Vietnamese_Embedding
  - cùng nền bge-m3 với reranker đang dùng (AITeamVN/Vietnamese_Reranker)
  - 2048 token/lần đọc, gấp 4 lần cross-encoder (512) -> nuốt trọn đoạn dài
  - đã đo trên Zalo Legal 2021: Accuracy@5 = 0.9305

Tài sản sinh ra (quy tắc 2): `emb_corpus.npy` + `emb_meta.json`. Nhúng MỘT LẦN
cho cả kho; kho dùng chung giữa dev và đề thi nên dùng lại mãi. Có hai file đó
thì mọi thí nghiệm hoà điểm chạy được trên CPU, không cần GPU nữa.

Dùng:
    python bi_encoder.py count      # đếm mảnh + ước chi phí, CPU, làm TRƯỚC
    python bi_encoder.py selftest
Trên Kaggle: import rồi gọi encode_corpus() / load() / doc_scores().
"""
import json, os, sys
import numpy as np

from make_candidates_fallback import chunks_of, read_passage   # dùng lại, không chép

EMBED_CHARS = 4000        # ~1400 token tiếng Việt, dưới trần 2048 của model.
                          # Cross-encoder đang dùng 1800; ở đây nới gấp đôi vì
                          # bi-encoder đọc được dài hơn -> ít mảnh hơn, rẻ hơn.
MODEL_NAME  = "AITeamVN/Vietnamese_Embedding"
BATCH       = 32


def _merge(parts, target):
    """Gộp đoạn liền kề tới ~target ký tự. Bản sao ngắn của deep_chunk._merge --
    KHÔNG import deep_chunk để tránh kéo theo cache BM25 nặng của nó."""
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) + 2 > target:
            out.append(buf); buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def doc_chunks(ctx_dir, doc_id):
    return _merge(chunks_of(read_passage(ctx_dir, doc_id)), EMBED_CHARS)


def corpus_ids(ctx_dir):
    return sorted(f[8:-5] for f in os.listdir(ctx_dir)
                  if f.startswith("context_") and f.endswith(".json"))


def count(ctx_dir=None):
    """Quy tắc 6: ước chi phí bằng SỐ MẢNH trên CPU trước khi chạm GPU."""
    ctx_dir = ctx_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "Input", "selected-contexts")
    ids = corpus_ids(ctx_dir)
    n = sum(len(doc_chunks(ctx_dir, d)) for d in ids)
    print(f"{len(ids):,} văn bản · {n:,} mảnh @ {EMBED_CHARS} ký tự")
    print(f"ước GPU @ 30 mảnh/s: {n/30/60:.0f} phút · file .npy fp16: {n*1024*2/1e6:.0f} MB")
    return n


def encode_corpus(model, ctx_dir, out_dir, shard=500):
    """Nhúng cả kho, lưu theo lô để đứt giữa chừng vẫn chạy tiếp được (quy tắc 2)."""
    os.makedirs(out_dir, exist_ok=True)
    ids = corpus_ids(ctx_dir)
    meta_p = os.path.join(out_dir, "emb_meta.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    for i in range(0, len(ids), shard):
        part = os.path.join(out_dir, f"emb_{i:06d}.npy")
        if os.path.exists(part):
            continue
        texts, owner = [], []
        for d in ids[i:i + shard]:
            cs = doc_chunks(ctx_dir, d)
            texts += cs; owner += [d] * len(cs)
        v = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float16)
        np.save(part, v)
        meta[str(i)] = owner
        json.dump(meta, open(meta_p, "w", encoding="utf-8"))
        print(f"  lô {i:>6}/{len(ids)}  {len(texts):>6} mảnh  -> {os.path.basename(part)}", flush=True)
    return meta_p


def load(out_dir):
    """-> (ma trận mảnh × 1024, mảng doc_id song song từng dòng)."""
    meta = json.load(open(os.path.join(out_dir, "emb_meta.json"), encoding="utf-8"))
    keys = sorted(meta, key=int)
    V = np.concatenate([np.load(os.path.join(out_dir, f"emb_{int(k):06d}.npy")) for k in keys])
    owner = np.array([d for k in keys for d in meta[k]])
    assert len(V) == len(owner), f"lệch: {len(V)} vector vs {len(owner)} nhãn"
    return V, owner


def doc_scores(qvec, V, owner, docs=None):
    """Điểm mỗi văn bản = cosine CAO NHẤT trong các mảnh của nó.

    Lấy max chứ không lấy trung bình: chỉ cần MỘT đoạn chứa câu trả lời là văn
    bản đó đáng chọn -- cùng lập luận đã dùng cho ce_deep ngày 15/08.
    """
    sims = V.astype(np.float32) @ qvec
    out = {}
    for s, d in zip(sims, owner):
        if docs is not None and d not in docs:
            continue
        if s > out.get(d, -9):
            out[d] = float(s)
    return out


def selftest():
    p = ["a" * 1500, "b" * 1500, "c" * 1500, "d" * 900]
    m = _merge(p, 4000)
    assert len(m) == 2 and len(m[0]) == 3002 and len(m[1]) == 2402, [len(x) for x in m]
    V = np.array([[1, 0], [0, 1], [.6, .8]], dtype=np.float16)
    own = np.array(["A", "B", "A"])
    s = doc_scores(np.array([1, 0], dtype=np.float32), V, own)
    assert abs(s["A"] - 1.0) < 1e-3 and abs(s["B"]) < 1e-3, s      # max, không phải trung bình
    assert doc_scores(np.array([0, 1], dtype=np.float32), V, own, docs={"B"}) == {"B": 1.0}
    print("selftest OK — gộp mảnh đúng ngưỡng, doc_scores lấy MAX và lọc được theo rổ")


if __name__ == "__main__":
    (count if len(sys.argv) > 1 and sys.argv[1] == "count" else selftest)()
