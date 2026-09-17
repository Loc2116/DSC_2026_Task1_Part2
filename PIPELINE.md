# PIPELINE.md — chuỗi bước để dựng lại đúng `submission_matchEmb_K50_n2.zip` (public LB 0.9124)

> Viết ra vì README hiện tại chỉ nói "kết quả" mà không nói **thứ tự chạy**. Đọc file này
> TRƯỚC khi mở bất kỳ notebook nào nếu mục tiêu là tái lập đúng con số 0.9124.

## ⚠️ Phạm vi thật của repo này — đọc trước

Repo này chỉ là **khâu xếp hạng + chốt top-5** (thành viên E). Nó nhận **rổ 50 ứng viên/câu
đã fusion (BM25 ⊕ dense, RRF)** làm đầu vào có sẵn — nó **không** chứa mã sinh ra rổ đó.

**Không có notebook/script nào trong `src/` hoặc `notebooks/` build ra
`fusion_rrf_top50_public_matchEmbedded.json`.** `src/ens.py` và `src/devfuse.py` chỉ có hàm
RRF dùng cho phân tích/đối chứng (ensemble hai hệ đã có sẵn điểm), không phải script sinh rổ
gốc từ BM25 + bi-encoder cho 1000 câu đề thi. `notebooks/biencoder_run.ipynb` cũng chỉ đo
"dense một mình mạnh tới đâu" (chẩn đoán), không ghép RRF. **Đây là phụ thuộc ngoài repo, do
thành viên D tạo ra** — repo này không cần chứa code sinh ra nó, chỉ cần dữ liệu D đã sinh
sẵn (xem dataset ở Bước 0).

### 📦 Dataset Kaggle (public)

**https://www.kaggle.com/datasets/locdovan211/project-ir/data**

Chứa toàn bộ file trung gian mà các notebook ở đây cần làm `INPUT_DIR`, bao gồm
`fusion_rrf_top50_public_matchEmbedded.json` (rổ ứng viên do D sinh) — **đã public, đã xác
nhận đủ file** (17/09). BTC add dataset này vào Kaggle notebook (`+ Add Input`) là dùng được
ngay, không cần build lại từ đầu.

## Sơ đồ phụ thuộc giữa các notebook (chỉ phần E, từ rổ D giao tới submission)

```
[D giao] fusion_rrf_top50_public_matchEmbedded.json  (ngoài repo)
                    │
                    ▼
        fusion_v3_run.ipynb  (MODE="public_a")
        M=10, K=20 · skip=0
                    │  → scores_public_fusion_M10_K20_matchEmbedded.json
                    ▼
        fusion_v3_run.ipynb  (MODE="public_b")
        nạp lại output public_a, mở rộng M 10→20, skip=10
                    │  → scores_public_fusion_M20_K20_matchEmbedded.json   [mốc ~0.9109–0.9118]
                    ▼
        fusion_k50_run.ipynb  (MODE="public_k50")
        nạp scores_public_fusion_M20_K20_matchEmbedded.json làm tầng 1,
        BỎ ce_deep cũ (K=20), chấm lại tầng 2 ở K_CHUNK=50
                    │  → scores_public_fusion_M20_K50_matchEmbedded.json
                    │  → submission_matchEmb_K50_n{1,2,3,4}.zip
                    ▼
        CHỌN n=2  →  submission_matchEmb_K50_n2.zip  ==  0.9124 (BÀI CHỐT)
```

## Các bước cụ thể

### Bước 0 — Dữ liệu
Add dataset **[locdovan211/project-ir](https://www.kaggle.com/datasets/locdovan211/project-ir/data)**
làm input cho notebook Kaggle (`INPUT_DIR` tự dò ra `/kaggle/input/project-ir` hoặc
`/kaggle/input/datasets/locdovan211/project-ir`). Dataset này đã có sẵn:
- `public-official.json`, `selected-contexts/` — dữ liệu BTC.
- **`fusion_rrf_top50_public_matchEmbedded.json`** — rổ ứng viên do D sinh.
- Toàn bộ `src/*.py` (upload phẳng, không giữ cấu trúc thư mục — xem cảnh báo trong README).

Chi tiết định dạng từng file raw của BTC (nếu cần build lại dataset này từ đầu): `data/README.md`.

### Bước 1 — `notebooks/fusion_v3_run.ipynb`, chạy **HAI lần**
1. Đặt `MODE = "public_a"` → chạy. Ra `scores_public_fusion_M10_K20_matchEmbedded.json`
   (tầng 1 CE cho cả 50 ứng viên + tầng 2 đọc sâu K=20 cho top-10). Tải file này về, upload
   lại vào dataset Kaggle (để lượt sau nạp qua biến `PREV`).
2. Đặt `MODE = "public_b"` → chạy lại. Notebook tự nạp file ở bước 1 làm `PREV`, mở rộng
   đọc sâu từ top-10 lên top-20 (`skip=10`). Ra `scores_public_fusion_M20_K20_matchEmbedded.json`.
   Đây là mốc ~0.9109–0.9118 (chưa phải bài chốt).

*(Chi phí: theo comment trong code, mỗi lượt cỡ vài giờ GPU T4; lượt `public_b` rẻ hơn vì
tái dùng tầng 1 của `public_a`, chỉ tính thêm phần đọc sâu M10→M20.)*

### Bước 2 — `notebooks/fusion_k50_run.ipynb`, `MODE = "public_k50"`
- Nạp `scores_public_fusion_M20_K20_matchEmbedded.json` (output bước 1.2), **bỏ `ce_deep` cũ**
  (đo ở K=20, không so được với K=50 — notebook tự làm việc này, xem `assert` trong ô "Bước 2").
- Chấm lại tầng 2 ở `K_CHUNK=50` cho toàn bộ top-20 văn bản/câu.
- ⏱ Nếu ước >6h, chia `Q_TU, Q_DEN` làm 2 lát (0–500, 500–1000), chạy 2 phiên Kaggle nối tiếp
  (mỗi phiên tự lưu `scores_*.json` sau mỗi lô 100 câu — nối tiếp được nếu đứt phiên).
- **Chi phí thật đã đo: 14,8 giờ GPU cho cả bước này** (ghi trong `docs/NHAT_KY_KY_THUAT.md`).
- Kết thúc: ghi `scores_public_fusion_M20_K50_matchEmbedded.json` và tự đóng gói
  `submission_matchEmb_K50_n{1,2,3,4}.zip` (biến `n` trong `blend_bm25_first`).

### Bước 3 — Chọn bài nộp
**Nộp `submission_matchEmb_K50_n2.zip`** (n_bm25=2). Đây là cấu hình chốt — n=3 cho 0.9118,
thấp hơn 0,06 điểm (xem bảng 2×2 K×n trong `docs/NHAT_KY_KY_THUAT.md`).

## Kiểm lại trước khi báo "tái lập được"

- [ ] `scores_public_fusion_M20_K20_matchEmbedded.json` dựng lại từ bước 1 khớp md5 với bản
      đã dùng khi nộp thật (nếu còn giữ) — guard đã có sẵn trong `fusion_k50_run.ipynb` Bước 2.
- [ ] `submission_matchEmb_K50_n2.zip` dựng lại khớp **1000/1000 câu** với bài đã nộp thật lên
      Codabench (không chỉ khớp điểm số — hai bài khác nhau có thể tình cờ ra cùng điểm).
- [x] Dataset [locdovan211/project-ir](https://www.kaggle.com/datasets/locdovan211/project-ir/data)
      đã Public, đã xác nhận có đủ file (17/09).

## Sửa so với README hiện tại

README đang ghi: *"`notebooks/fusion_v3_run.ipynb` — lượt chạy sinh ra bài nộp hiện tại."*
**Không còn đúng.** `fusion_v3_run.ipynb` sinh ra bài nộp ở K=20 (mốc 0.9118, đã bị vượt).
Bài nộp hiện tại (0.9124) được đóng gói ở cuối `fusion_k50_run.ipynb`. Đây đúng là kiểu lỗi
"quên cập nhật con trỏ khi đổi bài chốt" mà `docs/NHAT_KY_KY_THUAT.md` đã cảnh báo nhiều lần —
chỉ là lần này nó lọt vào chính README.
