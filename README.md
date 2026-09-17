# project_DSC — Truy hồi văn bản pháp luật tiếng Việt (DSC 2026, Task 1)

Hệ thống tra cứu văn bản pháp luật: cho một câu hỏi, trả về **5 văn bản** có khả năng
chứa câu trả lời, trong kho **8.532 văn bản**. Chỉ số chấm là `Recall@5` macro.

> *Vietnamese legal document retrieval. Two-stage pipeline: hybrid retrieval
> (chunk-level BM25 ⊕ dense bi-encoder, fused with RRF) → cross-encoder reranking with
> deep chunk reading → slot-reservation blending. Public LB 0.9124. Reports in `docs/`
> are in Vietnamese.*

**Public leaderboard: 0.9124** · Đây là phần **xếp hạng & chốt top-5** của hệ thống
(thành viên E); khâu sinh ứng viên do một thành viên khác phụ trách.

---

## Kết quả — từng bước một

Mỗi dòng là một thay đổi, đo trên public leaderboard:

| bước | LB | Δ |
|---|---|---|
| cross-encoder xếp hạng thuần | 0.8350 | — |
| + giữ chỗ 2 slot cho thứ tự rổ | 0.8573 | **+2,23** |
| + đọc sâu nhiều đoạn mỗi văn bản (M=20, K=20) | 0.8840 | **+2,67** |
| + chọn đoạn bằng BM25 mức đoạn, gộp 1.800 ký tự | 0.8871 | +0,31 |
| + đổi rổ ứng viên sang **hoà RRF** (BM25 ⊕ dense) | 0.9063 | **+1,92** |
| + M=10 → M=20 | 0.9109 | +0,46 |
| + rổ dùng model nhúng khớp với reranker | 0.9118 | +0,09 |
| + K=20 → K=50, giữ chỗ n=2 | **0.9124** | +0,06 |

**Hai đòn lớn nhất đều là đổi *thứ nạp vào*, không phải tinh chỉnh cái đang có.**
Đọc sâu (+2,67) và đổi rổ ứng viên (+1,92) cộng lại lớn hơn toàn bộ phần vặn tham số
của cả dự án.

## Pipeline

```mermaid
flowchart TD
    A["Kho 8.532 văn bản"] --> B["Cắt đoạn theo Điều/Khoản<br/>gộp tới 1.800 ký tự"]
    B --> C["BM25 mức đoạn<br/>điểm văn bản = max điểm đoạn"]
    B --> D["Dense bi-encoder<br/>Vietnamese_Embedding"]
    C --> E["Hoà RRF → rổ 50 văn bản/câu<br/>kèm 3 đoạn tốt nhất mỗi văn bản"]
    D --> E
    E --> F["Tầng 1 — cross-encoder chấm cả 50 văn bản"]
    F --> G["Cổng M=20 → Tầng 2 đọc sâu K=50 đoạn<br/>điểm = max(ce, ce_deep)"]
    G --> H["Chốt top-5 — giữ chỗ n=2 slot đầu theo thứ tự rổ"]
    H --> I["Nộp 5 doc_id"]
```

**Ba tham số đáng nhớ.** `M` = lấy bao nhiêu **văn bản** đầu bảng để đọc sâu ·
`K` = với mỗi văn bản đó, chấm bao nhiêu **đoạn** · `n` = giữ bao nhiêu slot đầu của
đáp án theo thứ tự rổ thay vì theo bộ chấm.

## Phát hiện đáng kể nhất: rổ tốt lên thì reranker teo lại

Trên 300 câu dev, đếm trực tiếp bộ chấm so với thứ tự rổ thô:

| | số câu |
|---|---|
| rổ xếp văn bản đúng ngoài top-5, bộ chấm **kéo vào được** | 14 |
| rổ xếp văn bản đúng trong top-5, bộ chấm **đẩy ra** | **22** |
| **ròng** | **−8 câu** |

Nộp thẳng 5 văn bản đầu rổ, không chấm lại gì, được **0,9117**. Chấm lại đầy đủ bằng
cross-encoder 568 triệu tham số được **0,9100** — *kém hơn*. Không phải bộ chấm yếu:
mỗi câu nó cứu được thì làm hỏng 1,6 câu. Đó là lý do quy tắc chốt top-5 ở đây là
**giữ chỗ** (khoanh vùng thiệt hại vào đúng `5−n` slot) chứ không phải gộp điểm
(bôi ảnh hưởng của bộ chấm lên cả 5 slot). Giữ chỗ thắng z-score, min-max và RRF
hai thứ hạng ở **mọi** tham số đã quét.

## Cấu trúc repo

```
src/         15 module Python — cắt đoạn, chọn đoạn, chấm, chốt top-5, đo đạc
notebooks/   25 notebook Kaggle, mỗi cái một lượt thí nghiệm
docs/        báo cáo kỹ thuật + nhật ký + bảng đối chứng model
results/     số liệu đủ để tái hiện MỌI phân tích trong docs/ trên CPU
data/        trống — xem data/README.md
```

⚠️ **Trên Kaggle, mọi file trong `src/` được upload PHẲNG vào một dataset rồi
`sys.path.append`.** Cấu trúc thư mục ở đây chỉ để đọc cho dễ; notebook import theo
tên module (`from deep_chunk import ...`), không theo đường dẫn `src/`.

**Điểm vào chính:**

| file | việc |
|---|---|
| `src/deep_chunk.py` | cắt đoạn, `pick_chunks` (chọn K đoạn bằng BM25 trong phạm vi văn bản), `deepen_all` |
| `src/rerank_from_d.py` | chấm ứng viên, `blend_bm25_first` (quy tắc giữ chỗ) |
| `src/metrics.py` | `Recall@k` / `Precision@k` đúng công thức BTC, kèm 10 test bắt lỗi âm thầm |
| `src/rerank_qwen.py` | chạy reranker kiểu decoder, đếm tham số thật dưới lượng tử hoá 4-bit |
| `notebooks/fusion_v3_run.ipynb` → `notebooks/fusion_k50_run.ipynb` | hai lượt chạy nối tiếp sinh ra bài nộp hiện tại (0.9124) — **thứ tự đầy đủ, kèm tên file trung gian, ở [`PIPELINE.md`](PIPELINE.md)** |

⚠️ **Repo này chỉ là khâu xếp hạng + chốt top-5.** Rổ ứng viên đầu vào
(`fusion_rrf_top50_public_matchEmbedded.json`) do thành viên khác tạo, **không sinh được từ
mã trong repo này** — xem cảnh báo đầu `PIPELINE.md`.

## Tái hiện phân tích — không cần GPU

Toàn bộ số liệu trong `docs/` dựng lại được từ `results/` trong vài giây:

```bash
pip install -r requirements.txt
python - <<'PY'
import json
sc = json.load(open('results/dev300/scores_dev300_moc_k3.json'))
ro = json.load(open('results/dev300/order_dev300_matchEmb.json'))

def chot(sc_q, ro_q, n, k=5):
    """Giữ chỗ n slot đầu cho thứ tự rổ, phần còn lại theo bộ chấm."""
    xep, out = sorted(sc_q, key=lambda d: -sc_q[d]), []
    for d in list(ro_q[:n]) + xep + list(ro_q):
        if d not in out: out.append(d)
        if len(out) >= k: break
    return out[:k]

print(chot(sc['29410'], ro['29410'], n=3))
PY
```

Muốn ra được điểm thì cần nhãn — xem `data/README.md`.

## Những hướng đã thử và THẤT BẠI

Phần này có lẽ hữu ích hơn phần thành công. Mỗi dòng là một phép đo đã trả tiền bằng
giờ GPU, không phải phỏng đoán.

| hướng | kết quả đo được |
|---|---|
| Tinh chỉnh toàn phần cross-encoder (2 lần) | **−0,50** rồi **−6,50** điểm |
| Chưng cất từ model 8B | ròng −16 nhóm/600. **Thầy (0,5417) thua trò (0,5683)** |
| Học lại đầu phân loại, đóng băng encoder | Δ tốt nhất +0,67; 18 câu sai → vẫn 18 câu sai |
| Ensemble nhiều bộ chấm / trộn hai pipeline | âm ở mọi trọng số |
| Gộp điểm z-score / min-max / RRF ở khâu chốt | thua quy tắc giữ chỗ ở **mọi** tham số |
| Chọn `n` thích nghi theo từng câu | trần oracle chỉ +1,17 → dưới sàn nhiễu |
| Ưu thế theo loại văn bản / năm ban hành / thẩm quyền | âm, **kể cả khi ước tham số ngay trên chính tập dev** |
| Ưu tiên văn bản mới nhất | gold là văn bản mới nhất đúng 2,0% = ngẫu nhiên |
| Ghép tên văn bản vào truy vấn | slug không dấu −2,0; tiêu đề có dấu +1,00 nhưng 83% trùng tín hiệu sẵn có |
| Nâng `K` 20→50 | 14,8 giờ GPU cho **+0,06 điểm** |

**Vì sao tinh chỉnh hỏng** (chẩn đoán, không phải "không làm được"): negative lấy từ
BM25 top-20 là **khó nhất có thể** — trong kho pháp luật, top-20 đầy văn bản trả lời
được câu hỏi nhưng không được gán nhãn. Dạy máy "những cái đó SAI" là dạy điều không
đúng. Cộng thêm nhãn dương chọn bằng BM25 chỉ đúng 33%.

## Vài bài học phương pháp

1. **Luôn in kèm dòng "rổ trần, không chấm lại".** Cấu hình đang đo còn thua dòng đó
   thì dừng, đừng đọc Δ. Một dòng code — đã cứu 3 giờ GPU và cả một nhánh nghiên cứu.
2. **So sánh phải GHÉP CẶP.** Cùng tập câu, chỉ đổi một tham số → chỉ câu *bất đồng*
   mang thông tin. Sai số không ghép cặp là 1,30 điểm; ghép cặp (McNemar) đọc được
   hiệu ứng +2,33 ở p = 0,039. **Cùng dữ liệu, mạnh hơn ~3 lần, 0 giờ GPU.**
3. **Đo trần trước khi chạy.** Oracle chạy trên CPU vài giây và cho biết trần tuyệt
   đối của hướng đang định thử — đã đóng ba hướng lớn trước khi tiêu một giờ GPU nào.
4. **Kiểm chồng lấn trước khi cộng dồn.** Đọc sâu, ghép tiêu đề, rổ tốt hơn, chấm lại
   — bốn "giải pháp" này sửa *cùng* một nhóm câu. Đo riêng thì cái nào cũng dương,
   gộp lại thì không cộng được.
5. **Ước chi phí bằng SỐ ĐOẠN, đếm cho đúng cấu hình sắp chạy.** Một lần bắc cầu qua
   con số của cấu hình khác đã cho ra ước 6,9 giờ trong khi thực tế là 14,8 giờ.
6. **Tên repo model không đáng tin — phải đếm tham số thật.** Dưới lượng tử hoá NF4,
   `sum(p.numel())` báo chưa tới một nửa số thật.

Đầy đủ trong `docs/NHAT_KY_KY_THUAT.md`.

## Tài liệu

| file | nội dung |
|---|---|
| `docs/BaoCao_TomTat_Retrieval_Khim.docx` | 4 trang — phương pháp và các hướng đã thử, kèm kết quả nhanh |
| `docs/BaoCao_ChiTiet_Retrieval_Khim.docx` | 15 trang — giao thức đo, tham số, số liệu, bẫy đã dính |
| `docs/pipeline_loi_DSC.html` | sơ đồ pipeline + bảng phân tích điểm mất theo nguyên nhân |
| `docs/bang_ablation.md` | đối chứng 10 cross-encoder trên **cùng 15.000 cặp** |
| `docs/NHAT_KY_KY_THUAT.md` | nhật ký kỹ thuật — mọi kết luận kèm phép đo đứng sau nó |

**Một kết quả từ bảng đối chứng:** số tham số không dự đoán được chất lượng.
`mmarco-mMiniLMv2` (0,118B) đạt R@5 0,7217, hơn `ViRanker` (0,568B, 0,6867) tới
**+3,50 điểm** dù nhỏ hơn 4,8 lần và nhanh hơn 16 lần. Thứ quyết định là dữ liệu
huấn luyện tiếng Việt/pháp lý, không phải kích thước.

## Không có gì trong repo này

- **Kho văn bản và tập câu hỏi của BTC** — không tái phát tán. Xem `data/README.md`.
- **Mọi tạo phẩm chứa dự đoán trên đề thi** (`scores_public_*`, `order_public_*`,
  các file bài nộp). Cuộc thi chưa kết thúc; công khai dự đoán đề thi là đưa đáp án
  cho người khác. Sẽ bổ sung sau khi kết thúc nếu thể lệ cho phép.
- **Vector đã nhúng** (`.npy`, 52 MB) và các rổ ứng viên (207 MB mỗi file) — quá lớn
  cho git, dựng lại được từ mã nguồn.
