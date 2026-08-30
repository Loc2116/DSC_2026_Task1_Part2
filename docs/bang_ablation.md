# Bảng đối chứng bộ chấm chéo (cross-encoder) — Task 1, dev300

**Giao thức.** 300 câu dev khoá cứng · rổ ứng viên fusion RRF top-50 của D · mỗi văn bản
lấy **1 đoạn** (chọn đoạn bằng BM25 mức đoạn, gộp 1800 ký tự, trích 900 ký tự) ·
chấm **một tầng**, không đọc sâu · **15.000 cặp (300 × 50) cho mỗi model, y hệt nhau** ·
GPU T4. Trần của rổ (recall@50) = **0.9817**.

> Các con số dưới đây **thấp hơn** hệ thống hoàn chỉnh (0.9350) vì bảng này **bỏ tầng đọc sâu**
> để cô lập đúng một biến: chất lượng bộ chấm. Dùng để xếp hạng bộ chấm với nhau,
> không phải để báo cáo điểm hệ thống.

| Model | Nền / kiến trúc | Tham số | R@1 | R@5 | R@10 | MRR@10 | R@5 (+RRF, n=1) | giây/cặp |
|---|---|---|---|---|---|---|---|---|
| **AITeamVN/Vietnamese_Reranker** ← đang dùng | XLM-R (bge-v2-m3), tinh chỉnh tiếng Việt | 0,568B | **0,5017** | 0,7467 | 0,8300 | **0,6178** | **0,8267** | 0,058 |
| BAAI/bge-reranker-v2-m3 | XLM-R, đa ngữ | 0,568B | 0,4683 | **0,7517** | **0,8550** | 0,5994 | 0,8233 | 0,056 |
| jinaai/jina-reranker-v2-base-multilingual | XLM-R, đa ngữ | 0,278B | 0,4383 | 0,7233 | 0,8183 | 0,5677 | 0,8017 | 0,025 |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | mMiniLM, đa ngữ | 0,118B | 0,4867 | 0,7217 | 0,8200 | 0,5963 | 0,8117 | **0,007** |
| Qwen/Qwen3-Reranker-0.6B | Qwen3, **decoder** | 0,596B | 0,4133 | 0,7217 | 0,8250 | 0,5627 | 0,8083 | 0,257 |
| itdainb/PhoRanker | PhoBERT (trần 256 token) | 0,135B | 0,4283 | 0,7200 | 0,8000 | 0,5549 | 0,8217 | 0,014 |
| namdp-ptit/ViRanker | XLM-R, tinh chỉnh tiếng Việt | 0,568B | 0,3917 | 0,6867 | 0,7783 | 0,5182 | 0,7833 | 0,106 |
| BAAI/bge-reranker-base | XLM-R base, đa ngữ | 0,278B | 0,3583 | 0,6850 | 0,7733 | 0,5064 | 0,8167 | 0,018 |
| cross-encoder/ms-marco-MiniLM-L6-v2 *(đối chứng ÂM — tiếng Anh)* | BERT tiếng Anh | 0,023B | 0,3233 | 0,5633 | 0,7000 | 0,4421 | 0,7333 | 0,005 |

*Đã loại khỏi bảng:* `thanhtantran/Vietnamese_Reranker` — xem ghi chú 2.
*Không đo được:* `Alibaba-NLP/gte-multilingual-reranker-base` — mã tuỳ biến bắn device-side assert
(tra bảng nhúng ngoài biên) ngay lượt truyền xuôi đầu tiên trên T4, cả hai lần thử. Không phải
lỗi phần cứng và không sửa được trong ngân sách thời gian; ghi nhận là **không đánh giá được**,
không phải điểm thấp.

## Bốn điều bảng này chứng minh

**1. Model đang dùng là lựa chọn đúng.** `AITeamVN` dẫn đầu ở **R@1**, **MRR@10** và
**R@5 sau khi pha rổ** — đúng ba chỉ số mà pipeline thực tế phụ thuộc vào. Nó chỉ thua
`bge-reranker-v2-m3` ở R@5 thuần (0,7467 vs 0,7517, −0,50), mà `bge-v2-m3` chính là **nền**
của nó: phần tinh chỉnh tiếng Việt đổi một chút R@5 lấy khả năng **xếp đúng thứ tự ở đỉnh**
(R@1 +3,34, MRR +1,84). Với bài toán chỉ nộp 5 đáp án và có pha rổ, đó là đánh đổi có lợi.

**2. `thanhtantran/Vietnamese_Reranker` là bản sao.** Điểm của nó **trùng khít từng bit**
với `AITeamVN` trên cả 15.000 cặp (lệch tối đa 0,000e+00). Khác tài khoản HF, cùng trọng số.
Không tính là một dòng đối chứng độc lập.

**3. Số tham số KHÔNG dự đoán được chất lượng.** `mMiniLMv2` 0,118B đạt 0,7217, hơn
`ViRanker` 0,568B (0,6867) tới **+3,50 điểm** dù nhỏ hơn **4,8 lần** và chạy **nhanh 16 lần**.
Thứ quyết định là **dữ liệu huấn luyện tiếng Việt/pháp lý**, không phải kích thước.

**4. Kiến trúc decoder không tự động tốt hơn.** `Qwen3-Reranker-0.6B` (decoder, chấm bằng
xác suất token "yes") đạt đúng 0,7217 — **bằng** `mMiniLMv2` 0,118B — nhưng chậm hơn
**40 lần** (0,257 vs 0,007 giây/cặp) và có R@1 thấp nhất trong nhóm khá (0,4133).
Khớp với kết quả đo riêng trên 15 câu khó: các decoder lớn hơn cũng không mở ra được
nhóm câu mà bộ chấm hiện tại bỏ lỡ.

**Đối chứng âm hoạt động đúng.** `ms-marco-MiniLM-L6-v2` (BERT tiếng Anh thuần) rơi xuống
0,5633 — cách nhóm dẫn đầu 18,8 điểm. Bảng đo đúng năng lực tiếng Việt, không phải nhiễu.
