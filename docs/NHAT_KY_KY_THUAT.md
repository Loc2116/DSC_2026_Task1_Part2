# project_DSC — DSC2026 Task 1 (LegalIR) · thành viên E

> **Bản CỐT LÕI, gọn lại 28/08.** Toàn bộ nhật ký chi tiết 12–28/08 nằm ở
> `CLAUDE_LUUTRU_den_28-08.md` (4.009 dòng). Chỉ mở file đó khi cần dựng lại lý do
> của một kết luận cụ thể — mọi thứ cần để **làm việc** đều có ở đây.

## 🎯 TRẠNG THÁI

| | |
|---|---|
| **BÀI CHỐT** | `Ketqua_E/submission_matchEmb_K50_n2.zip` — **public LB 0.9124** (30/08) |
| cấu hình | rổ `_matchEmbedded` của D · `M=20` · **`K=50`** · `MERGE_CHARS=1800` · **`blend n_bm25=2`** |
| model | `AITeamVN/Vietnamese_Reranker` (0,568B) — tốt nhất trong 10 model đã đối chứng |
| hạn chót | **15/09** · quota GPU 60h/tuần (dư gấp ~3 lần kế hoạch) |
| mục tiêu hiện tại | **tối đa điểm public** — phải lọt top mới vào vòng private |

**Đang chạy / chờ chạy:**

1. ~~`fusion_k50_run.ipynb` — nâng `K` 20→50~~ **✅ XONG 30/08. ĐÓNG — xem mục K50 dưới.**
   Cả 1000 câu đã có `ce_deep` (đúng 20 vb/câu). Kỳ vọng ròng **+0,06 câu = +0,006 điểm LB**.
   Đừng nộp; đừng chạy `K` cao hơn.

2. `fusion_dev1000_run.ipynb` — kiểm `n=3` trên 1000 câu dev (~3,8h + 1,7h). **Phòng thủ cho vòng private, hoãn được** nếu đang gấp điểm public.

## ⛔⛔ RÀNG BUỘC CỨNG — ĐỌC TRƯỚC KHI CHỌN MODEL

> **Ngân sách BTC: cả đội 4B tham số. D lấy 1B (truy hồi). E CÒN 3B.**
> **Lượng tử hoá KHÔNG nới được** — 4 bit đổi chỗ chứa, không đổi số tham số.

| model | tham số thật | E dùng được? |
|---|---|---|
| **AITeamVN/Vietnamese_Reranker** ← đang dùng | 0,568B | ✅ |
| Qwen3-Reranker-0.6B | 0,596B | ✅ |
| Qwen3-Reranker-**4B** | **4,02B** | ❌ tên "4B" nhưng thật 4,02B |
| Qwen3-Reranker-**8B** | **8,19B** | ❌ vượt ngân sách **và** ngoài danh sách trắng |

**Danh sách trắng BTC** kịch trần ở ~4B, không model nào 7B/8B. **Mọi LLM thương mại
(GPT/Gemini/Claude) không được chấm điểm.** Cấm gán nhãn tay, cấm dữ liệu ngoài.

⚠️ **Dưới NF4, `sum(p.numel())` báo chưa tới một nửa số thật** (bitsandbytes gói 2 trọng số
vào 1 byte `uint8`). Qwen3-Reranker-4B báo 2,205B cho model 4,02B. Dùng `_real_params`
trong `rerank_qwen.py`. **Tên repo không đáng tin, phải đếm.**

🔑 **Có cửa mở:** thể lệ cho phép đề xuất bổ sung model mã nguồn mở **trước 10 ngày so với
hạn private-test**. Muốn model ngoài danh sách thì đây là đường duy nhất.

## 🗺️ BẢN ĐỒ DƯ ĐỊA (đo lại 28/08 trên rổ `_matchEmbedded`, dev300)

**300 câu đang nằm đâu trong rổ:**

| hạng gold trong rổ | số câu | ai lo |
|---|---|---|
| 1 | 192 | — xong |
| 2–5 | 84 | — xong (cộng dồn **276 câu = 0,9117**, không cần chấm gì) |
| 6–10 · 11–20 · 21–50 | 11 · 4 · 6 | E |
| **ngoài rổ 50** | **3** | **D** |

| | dev300 |
|---|---|
| rổ trần @5 (không chấm lại gì) | **0,9117** |
| tầng 1 đầy đủ (k=3, MAX) | 0,9100 ← **kém hơn không chấm** |
| `blend n=3` (cấu hình đang nộp) | **0,9467** |
| **TRẦN của E = recall@50 của rổ** | **0,9883** |

**13 câu còn sai:** 3 ngoài rổ (D) · 1 gold ở rổ hạng 5 bị CE đẩy ra · 9 cả rổ lẫn CE đều vùi.

**Số mạnh nhất cho báo cáo:** rổ trần 0,9117 **hơn** tầng 1 đầy đủ 0,9100. Chấm lại 50 ứng
viên bằng cross-encoder 568 triệu tham số ra **kém hơn không chấm**. Reranker không yếu —
rổ đã giỏi tới mức không còn gì để sửa. Toàn bộ giá trị còn lại nằm ở tầng 2.

**Việc của D đã đổi chỗ:** recall@50 = 0,9883, gần cạn. Nhưng `blend n=3` đưa **3 slot đầu
của rổ THẲNG vào đáp án**, không qua reranker → **recall@1 của rổ (đang 0,6217) đi thẳng vào
kết quả cuối**. Câu hỏi cho D không còn là "lấy thêm văn bản" mà là "xếp đúng hơn ở hạng 1–5".

## 📉 E7 — ĐỘ LỆCH dev ↔ public LB. **BẢNG QUAN TRỌNG NHẤT FILE NÀY.**

| ngày | cấu hình | dev300 | public LB |
|---|---|---|---|
| 12/08 | candidate D, n=2 | 0.8883 | 0.8573 |
| 17/08 | deepchunk M=20/K=20, n=2 | 0.9083 | 0.8840 |
| 19/08 | BM25 chọn đoạn + gộp 1800, n=2 | 0.9183 | 0.8871 |
| 22/08 | rổ fusion M=10, **n=1** | **0.9350** ← đỉnh dev | 0.8943 ← **thấp nhất** |
| 22/08 | rổ fusion M=10, n=2 | 0.9283 | 0.9033 |
| 22/08 | rổ fusion M=10, **n=3** | 0.9233 ← thấp nhất dev | **0.9063** ← **đỉnh LB** |
| 23/08 | rổ fusion M=20, n=2 | — | 0.9109 |
| 25/08 | **rổ mới M=20, n=3** | — | **0.9118** ← BÀI CHỐT |
| 25/08 | rổ mới M=20, n=1 · n=2 · n=4 | — | 0.9031 · 0.9104 · 0.9056 |
| 30/08 | **rổ mới M=20, `K=50`, n=2** | — | **0.9124** ← BÀI CHỐT |
| 30/08 | rổ mới M=20, `K=50`, n=3 | — | 0.9118 |

**⛔ KHÔNG CÓ HẰNG SỐ QUY ĐỔI.** Độ lệch thay đổi theo `n`: −5,08 (n=0) · −4,07 (n=1) ·
−2,50 (n=2). Và **dev đã từng ĐẢO DẤU**: chênh dev −1,17 ứng với chênh LB **+1,20**.

> **dev300 dự báo được THỨ TỰ khi hiệu ứng LỚN, KHÔNG dự báo được mức tuyệt đối.**

⚠️ **SỬA 30/08 — quy tắc "dưới 1,5 điểm không đo được" chỉ đúng cho so sánh KHÔNG ghép cặp.**
Sai số không ghép cặp thật ra là **1,30 điểm** (√(p(1−p)/300)). Nhưng **mọi phép so trong dự án
này đều ghép cặp** — cùng 300 câu, cùng rổ, chỉ đổi một tham số — nên chỉ câu **bất đồng** mới
mang thông tin. Dùng McNemar trên rổ `_matchEmbedded`, K=20:

| cặp | A thắng | B thắng | bất đồng | Δ điểm | p |
|---|---|---|---|---|---|
| n=2 vs **n=3** | 1 | 6 | 7 | +1,67 | 0,125 |
| n=1 vs **n=2** | 1 | 8 | 9 | +2,33 | **0,039** |
| n=0 vs **n=3** | 2 | 23 | 25 | +7,00 | **<0,0001** |

**dev300 mạnh hơn ta tưởng ~3 lần, miễn là so ĐÚNG CÁCH. 0 giờ GPU.**
→ Từ nay báo cáo mọi Δ dev kèm **(thắng / thua / bất đồng)**, không chỉ hiệu hai số trung bình.

Rổ `_matchEmbedded` là ví dụ đắt nhất: **dev +3,83 → public +0,11.**

## 📈 CHUỖI TĂNG TIẾN — BASELINE

**Mọi thí nghiệm mới phải báo cáo dạng Δ so với hai con số này:**
> **dev300 = 0.8883** · **public LB = 0.8573** (AITeamVN gốc + `blend n_bm25=2`, candidate D)

| bước | public LB | Δ |
|---|---|---|
| rerank AITeamVN, n=0 | 0.8350 | — |
| + `blend_bm25_first(n=2)` | 0.8573 | +2,23 |
| + deepchunk M=20/K=20, `max` | 0.8840 | +2,67 |
| + BM25 chọn đoạn + gộp 1800 | 0.8871 | +0,31 |
| + rổ FUSION (M=10, n=3) | 0.9063 | +1,92 |
| + M=20, n=2 | 0.9109 | +0,46 |
| + rổ `_matchEmbedded`, n=3 | **0.9118** | +0,09 |

**Đọc bảng:** hai đòn lớn nhất (reranker +11 trên dev, rổ tốt hơn +2,4) đều là **đổi thứ
nạp vào**, không phải tinh chỉnh cái đang có.

## ⛔ HƯỚNG ĐÃ ĐÓNG — ĐỪNG THỬ LẠI. Mỗi dòng là một phép đo đã trả tiền.

| # | hướng | kết quả |
|---|---|---|
| 1 | fine-tune toàn phần (2 lần) | **−0,50** rồi **−6,50** |
| 2 | chưng cất từ Qwen3-8B | NET **−16** nhóm/600. Thầy 0,5417 **thua** trò 0,5683. Còn vi phạm danh sách trắng |
| 3 | probe — học lại đầu phân loại (encoder đóng băng) | Δ tốt nhất **+0,67**; **18 câu sai: 5/18 → 5/18, không đổi câu nào** |
| 4 | `M=30` | trần **~0** (3 câu chạm tới, 2 có oracle thấp hơn điểm hiện tại, 1 có 378 đoạn) |
| 5 | ensemble nhiều bộ chấm | âm |
| 6 | trộn hai pipeline (rổ cũ ⊕ rổ mới) | âm |
| 7 | gộp điểm z-score / RRF hai thứ hạng | **thua `blend n=3` ở MỌI tham số** |
| 8 | chọn `n` thích nghi theo câu | trần **oracle +1,17** — dưới ngưỡng |
| 9 | ưu thế theo loại văn bản / thẩm quyền | âm, **kể cả khi ước tham số ngay trên dev300** |
| 10 | ưu tiên văn bản mới / còn hiệu lực | gold là văn bản mới nhất đúng **2,0%** = ngẫu nhiên |
| 11 | ghép tên văn bản | slug không dấu **−2,0**; tiêu đề có dấu **+1,00** nhưng **83% trùng** tầng 2 |
| 12 | Gemini / LLM ngoài | hoà, **và BTC cấm** |
| 13 | "gợi ý liên quan" kiểu Google | thua cả ba biến thể |
| 14 | tăng `top_chunks` 3→5 | đã rút lại |
| 15 | cắt ứng viên 100→50→25 | đòn **chi phí**, không phải đòn điểm |
| 16 | cải tiến `pick_chunks` | `K=20` đã bắt **90%** giá trị oracle → gần cạn |

**Còn sống:** chỉ còn **truy hồi của D** (recall@1 = 0,6217). `K` và `n` đã đóng bằng bảng 2×2 đầy đủ.

## 🟢 30/08 — **BẢNG 2×2 `K` × `n` ĐÃ ĐỦ. CHỐT `K=50, n=2` = 0.9124. ĐÓNG CẢ HAI CẦN GẠT.**

| public LB | **n=2** | **n=3** | hiệu ứng `n` |
|---|---|---|---|
| **K=20** | 0.9104 | 0.9118 | **+0,14** |
| **K=50** | **0.9124** ← chốt | 0.9118 | **−0,06** |
| **hiệu ứng `K`** | **+0,20** | **0,00** | |

**`K` và `n` THAY THẾ NHAU — chúng sửa cùng một chỗ.** Cả hai đều nhắm vào phần bộ chấm làm hỏng
(cứu 14 / hỏng 22). `n` che bằng cách giữ chỗ; `K=50` chữa bằng cách cho bộ chấm đọc kỹ hơn. Làm
một cái rồi thì cái kia hết việc: `K` ăn +0,20 ở `n=2` nhưng **đúng 0,00** ở `n=3`.
Đây là ví dụ sống của quy luật 7 — *"bốn giải pháp hoá ra là một, đừng cộng dồn"*.

**Không nộp tiếp `n=1` hay `n=4` ở K=50.** Ở K=20 chúng cách đỉnh −0,73 và −0,62; mức `K` ăn được
nhiều nhất là +0,20. Số học không cho chúng vượt 0.9124.

### 📏 GIÁ THẬT CỦA CẢ HƯỚNG `K=50`: **14,8 giờ GPU cho +0,06 điểm** (0.9118 → 0.9124 = 0,6 câu).

### 🔮 ƯỚC LƯỢNG TRƯỚC-KHI-NỘP: dùng được, nhưng chỉ ở mức bậc độ lớn

Phương pháp (29/08): với mỗi câu đổi đáp án, lấy hạng-trong-rổ của văn bản bị đẩy ra / đưa vào,
nhân với `P(gold | hạng rổ)` đo trên dev300, cộng lại. Chạy CPU vài giây.

| | ước | thật | lệch |
|---|---|---|---|
| n=3 | +0,006 | **+0,000** | trúng |
| n=2 | +0,021 | **+0,200** | hụt 10× (nhưng chỉ 1,3σ) |

**Kết luận về công cụ:** đúng dấu cả hai lần, đúng độ lớn một lần, và **đúng ở việc quan trọng
nhất — nói rằng đây là cần gạt dưới 0,5 điểm.** Nếu chạy nó TRƯỚC thì đã tiết kiệm 14,8 giờ.
→ **Từ nay: mọi lượt GPU trên 1 giờ phải chạy ước lượng này trước.** Nó chỉ cần một file scores
đã có sẵn và thứ tự rổ.

🚧 **BIÊN GIỚI HIỆU LỰC của công cụ (đo 30/08 — quan trọng):** nó **CHỈ dùng được khi hai phía so
sánh đến từ CÙNG một cơ chế chọn.** Thử áp cho `n=2` vs `n=3`: ước **+3,49 điểm** cho n=3, thật
chỉ **+0,14** — hụt **25 lần**.
**Vì sao:** `P(gold | hạng rổ)` là xác suất tiên nghiệm của một văn bản *bất kỳ* ở hạng đó. Nhưng
slot của `n` được **bộ chấm CHỌN** — điều kiện "CE thích nó" đẩy P(gold) lên cao hơn hẳn tiên
nghiệm theo hạng. So `K=20` vs `K=50` thì cả hai phía đều do CE chọn → thiên lệch **triệt tiêu**,
ước đúng. So `n` thì một phía lấy theo rổ, một phía do CE chọn → thiên lệch **không** triệt tiêu.
Đây đúng là quy tắc 3 ("đừng đánh giá trên mẫu được chọn bằng một cơ chế khác") ở dạng khác.

⚠️ Đừng đọc bảng 2×2 quá mạnh: chênh lệch giữa các ô là **0,6–2 câu**. Cơ chế "thay thế nhau"
khớp với mọi ô và khớp với cơ chế đã biết của `n`, nhưng nó tựa trên 2 câu.

## 🧠 QUY LUẬT ĐÃ TRẢ GIÁ ĐỂ HỌC

1. **Ngưỡng phải TƯƠNG ĐỐI so với mốc hiện có, không bao giờ là con số tuyệt đối bốc ra.**
   Đã sai 3 lần (pre-register +2,0 cho `rescore8b`; cổng ≥0,60 cho nhãn thầy; cổng probe).
2. **Mọi phép đo bộ chấm phải in kèm dòng "rổ trần, không reranker".** Cấu hình đang đo còn
   thua dòng đó thì DỪNG, đừng đọc Δ. Một dòng code — đã cứu 3h GPU + cả nhánh LoRA.
3. **Không đánh giá model B trên mẫu được chọn bằng LỖI của model A.** Hồi quy về trung bình.
   Suýt tốn 25h GPU. So thì so trên mẫu ngẫu nhiên, hoặc đếm cả hai chiều cứu/hỏng.
4. **Mọi con số dư địa phải đo ở ĐÚNG cấu hình bài đang nộp**, và ghi kèm cấu hình bên cạnh
   con số. Đã sai 2 lần trong một ngày (trần 0,80 của *chọn đoạn* áp cho *sửa model*;
   đếm "22 câu hỏng" ở `n=1` trong khi bài nộp chạy `n=3` — thật ra chỉ hỏng 1 câu).
5. **"Có tín hiệu thô" ≠ "cộng vào được".** Một ưu thế TOÀN CỤC không thắng nổi bộ chấm đã
   đọc nội dung, trừ khi nó nói được điều gì đó **riêng cho từng câu**.
6. **"Không thấy hiệu ứng ở MỘT điểm đo" ≠ "hiệu ứng không tồn tại".**
7. **Bốn "giải pháp" hoá ra là một — đừng cộng dồn chúng.** Đọc sâu, tiêu đề, rổ tốt hơn,
   chấm lại: tất cả sửa **cùng một nhóm câu**. Đo riêng thì cái nào cũng dương, gộp lại thì
   không cộng được. Kiểm chồng lấn **trước** khi cộng.
8. **Chỉ chạy thí nghiệm kỳ vọng đổi ≥2 điểm.** Dưới ~1,5 điểm là dưới sàn nhiễu dev300.
9. **Đo trần TRƯỚC khi chạy**, đừng đoán. Đã cứu M=30 (~4h), chưng cất (~18h), thích nghi `n`.
10. **Quy tắc 4 áp cho CẢ CHI PHÍ, không chỉ dư địa.** 28/08: tôi đo mẫu 240 văn bản ra
    37,7 đoạn/văn bản ở `K=50` (→ 754/câu, đúng), rồi lại **quy đổi qua con số "157.008 đoạn"**
    trong quy tắc 6 — vốn là của một cấu hình KHÁC (157/câu). Ra 6,9h, thật là **14,8h**.
    Sai gấp đôi. **Có mẫu đo trực tiếp thì dùng thẳng nó, đừng bắc cầu qua số cũ.**
    *(Số đúng cho M=20/K=20 trên đề thi là ~338/câu, không phải 157/câu. Quy tắc 6 cần đo lại.)*

## 📋 QUY ƯỚC LÀM VIỆC

- **Câu đầu tiên luôn là kết luận** — ngắn, rõ, trực tiếp. Chi tiết để sau.
- **CLAUDE.md cập nhật liên tục, không cần hỏi.** `BaoCao_E.md` **CHỈ sửa khi Khim yêu cầu**.
- ⚠️ `BaoCao_E.md` **đang CŨ, dừng ở 12/08** (còn ghi `n_bm25=1`, LB 0.8440). Khi cập nhật thì
  lấy số từ file này — **đừng tự làm trước**.

### BẮT BUỘC: gọi skill `ponytail` mức `full` TRƯỚC MỌI VIỆC CODE

Gọi skill thật, trước dòng code đầu tiên. Áp dụng cho: viết/sửa file, notebook, refactor,
review, chọn thư viện. **Không** áp dụng cho: phân tích số liệu, đọc log, viết CLAUDE.md.

Vì sao gắt: repo đã có sẵn `chunks_of`, `tok`, `read_passage`, `blend_bm25_first`,
`evaluate`, `score_all_from_d`, `load_reranker`, `pick_chunks`, `deepen_all`. Gần như mọi
việc mới là **ghép lại đồ có sẵn**.

### NHẮC MỞ CHAT MỚI khi ngữ cảnh phồng

Nhắc Khim mở chat mới khi: đã đọc ≥5 file lớn · hội thoại >25 lượt · đổi việc · vừa đóng một
hạng mục và đã ghi kết luận. Một câu ở cuối câu trả lời, nhắc xong thì thôi.
**Điều kiện bắt buộc trước khi nhắc: đã cập nhật CLAUDE.md.**

### QUOTA GPU — 6 quy tắc

**Quota KHÔNG còn là nút thắt** (60h/tuần, ~240h tới 15/09, kế hoạch ăn ~78h). Chỗ chật thật
là **trần cứng 12h/lượt commit**, **số lượt nộp/ngày**, và **độ phân giải dev300**.

1. **Làm hết phần CPU trước khi chạm GPU.** Kiểm đường dẫn, key qid, dry-run, đếm chunk.
2. **Mỗi lượt GPU BẮT BUỘC sinh `scores_*.json` và tải về trước khi đóng phiên.** Không có
   scores = mọi thí nghiệm hậu kỳ phải chạy lại GPU. Lỗi đắt nhất đã mắc.
   *(Hệ quả mới 28/08: probe hai lượt đầu viết `MOC, _ = cham(...)` — vứt điểm đầu gốc, nên
   không phân tích từng câu được. Lượt k3 giữ lại → mọi phân tích sau đó miễn phí trên CPU.)*
3. **Mỗi lượt chỉ đổi MỘT biến.**
4. **Chỉ chạy thí nghiệm kỳ vọng đổi ≥2 điểm.**
5. **Gộp việc vào một phiên** — một lần khởi động, một lần tải model.
6. **Ước chi phí bằng SỐ CHUNK**, đừng ước bằng cảm giác — và **đếm cho ĐÚNG cấu hình sắp chạy**.
   Mốc đã kiểm 28/08: **đề thi M=20/K=50 = 685.196 đoạn = 14,8h @12,9 đoạn/s.**
   ⚠️ Con số cũ "đề thi = 157.008 chunk ≈ 1h45m" **không phải** của M=20/K=20 — đừng bắc cầu
   qua nó. `DC.count_deep_chunks` chạy CPU 3–4 phút và cho số thật; luôn đọc nó trước khi chấm.
7. **Trước MỌI lượt chạy trên 1 giờ, hỏi thành viên kia có đang làm việc đó không.** 19/08 mất
   4h dựng bi-encoder trong khi D đã làm xong. Chi phí 2 phút, cứu cả lượt GPU.

## 🪤 BẪY ĐÃ DÍNH, ĐỪNG DÍNH LẠI

**① ĐƯỜNG DẪN KAGGLE — tự dò, đừng hardcode:**

```python
INPUT_DIR = next(p for p in ("/kaggle/input/project-ir",
                             "/kaggle/input/datasets/locdovan211/project-ir")
                 if os.path.isdir(p))
CTX = next(p for p in (f"{INPUT_DIR}/selected-contexts/selected-contexts",
                       f"{INPUT_DIR}/selected-contexts")
           if os.path.isdir(p) and any(f.startswith("context_") for f in os.listdir(p)))
```
Kiểm bằng **có file `context_*.json` bên trong**, không phải `isdir`. Bản LỒNG phải để trước.

**② GUARD KIỂM BẰNG TEXT TỰ BẮN VÀO CHÂN.** `inspect.getsource` chứa cả docstring — docstring
bản mới trích code cũ là assert nổ oan, giết một lượt GPU.
→ **Kiểm bằng CHỮ KÝ HÀM hoặc hành vi:** `assert list(inspect.signature(f).parameters) == [...]`

**③ CACHE VECTOR KIỂM BẰNG SỐ HÀNG LÀ KIỂM HỜ.** Vector đoạn 900 và 1800 ký tự có **cùng số
hàng** → `nap()` nạp nhầm dữ liệu hỏng mà không báo gì. **Đổi TÊN FILE theo cấu hình.**

**④ `deepen_all` CHỈ trả về câu được truyền vào.** Gán đè `scores = deepen_all(lát, ...)` là
**mất tầng 1 của các câu ngoài lát**. Phải `scores.update(...)`.

**⑤ "TRA BẢNG NHÚNG NGOÀI BIÊN"** — device-side assert trên T4, đã dính 3 lần với model
`trust_remote_code`. Không phải lỗi phần cứng.

**⑥ NỘP NHẦM ĐỀ.** Bài 09/08 bị từ chối vì nộp dự đoán 150 câu dev. Guard `expected_qids`
không tự biết bạn nộp nhầm đề. → **Mọi file dev phải giữ hậu tố `_dev` + số câu trong tên.**

**⑦ Vượt 5 id ở một câu → CHỈ câu đó bị Recall/Precision = 0**, không phải cả bài.

**⑧ KHOÁ XẾP HẠNG CUỐI LÀ `max(ce, ce_deep)`, KHÔNG PHẢI `ce_deep`.** 30/08 tôi phân tích K50
bằng `ce_deep` và ra kết luận **ngược dấu** (K50 kéo top-5 ra XA thứ tự rổ). Dùng đúng khoá thì
ngược lại — kéo **về gần** rổ. → **Mọi phân tích CPU trên `scores_*.json` phải dựng lại bài nộp
và kiểm khớp `1000/1000` TRƯỚC khi đọc bất kỳ Δ nào.** Phép kiểm mất 5 giây và đã cứu một kết
luận sai hoàn toàn.

## 📁 DỮ LIỆU

| file | nội dung |
|---|---|
| `Input/public-official.json` | **1000 câu đề thi**, `answer: null`, giao với train = 0 |
| `Input/train.json` | 7000 câu có nhãn |
| `Input/selected-contexts/` | corpus **8.532** document · `context_<id>.json` → `{link, passage, id}` |
| `dev_1000_locked.json` | toà án cuối. Bao trùm dev300 + dev150 |
| `dev_300_locked.json` | thử nhanh hàng ngày (1,05 gold/câu) |
| `dev_150_locked.json` | **ĐÃ KHAI TỬ** — 256/300 câu giờ nằm trong tập huấn luyện |
| `fusion_rrf_top50_{dev_1000,public}_matchEmbedded.json` | **rổ đang dùng** (207MB mỗi file) |
| `bm25_ids_train_FALLBACK.json` | hard negative, 7000 câu × 20 doc_id |

`train.json 7000 ⊃ dev1000 ⊃ dev300`. Loại dev1000 khỏi huấn luyện là loại hết mọi tập dev.

**Tài sản phân tích (Ketqua_E/) — cho phép đo dev300 trên CPU trong vài giây, không cần GPU:**

| file | nội dung |
|---|---|
| `order_dev300_matchEmb.json` · `rrf_dev300_matchEmb.json` | thứ tự + điểm rổ, tách sẵn từ file 207MB |
| `scores_dev300_{moc,probe}_k3.json` | điểm từng văn bản của **cả đầu gốc lẫn đầu học lại** |
| `vanban_meta.json` | 5.244 văn bản: loại · năm ban hành · có phải cấp địa phương |
| `oracle_chunks_dev300.json` | CE chấm **toàn bộ đoạn** của 23 văn bản gold (7.362 cặp) |
| `vec_{train,dev}_1800.npy` | vector CLS đã cache — huấn luyện lại đầu phân loại mất 15s CPU |
| `bang_ablation.md` | bảng đối chứng 10 bộ chấm, **dán thẳng vào báo cáo** |
| `scores_public_fusion_M20_K50_matchEmbedded.json` | **điểm đề thi ĐẦY ĐỦ NHẤT** (1000 câu × 50 vb, 20 vb có `ce_deep` ở K=50). `ce` tầng 1 khớp bit-perfect với bản K=20; `ce_deep` chỉ tăng. Dùng file này thay bản K=20 cho mọi thí nghiệm CPU trên đề thi |
| `submission_matchEmb_K50_n{0..4}.zip` | 5 bài nộp K=50, **đã dựng lại từ scores khớp 1000/1000**. Hợp lệ nhưng **không đáng nộp** |

## 👥 VAI TRÒ

Khim = **thành viên E** — Xếp hạng & Chốt top-5. Sở hữu `recall@5`.
D lo hạ tầng + sinh ứng viên, E tiêu thụ và chốt submission.

## 📝 BÁO CÁO — bốn mục có số liệu sẵn, không cần thêm giờ GPU

1. **Bảng ablation 10 bộ chấm** (`bang_ablation.md`) — số tham số KHÔNG dự đoán chất lượng:
   `mMiniLMv2` 0,118B hơn `ViRanker` 0,568B tới +3,50 điểm.
   ⚠️ cả bảng đo ở `EXC=900` (đoạn bị cắt đôi) — thứ hạng tin được vì cùng giao thức, con số
   tuyệt đối thấp hơn thực tế.
2. **Vì sao chưng cất không ăn** — 600 nhóm, thầy 8B thua trò 0,568B, NET −16.
3. **Vì sao reranker teo khi rổ mạnh lên** — rổ trần 0,9117 vs tầng 1 0,9100. **Mục mạnh nhất.**
4. **Hệ thống cổng đặt trước đã cứu ~21h GPU** — `congkiem` (18h) + `assert MOC >= SAN` (3h + nhánh LoRA).

**Về nhóm câu không cứu được, viết cho đúng:** đọc thật 2 ca cho thấy model tối ưu **độ khớp
câu chữ**, còn đáp án đúng đòi biết **văn bản nào có thẩm quyền** (ca `138214`: model chọn QĐ
UBND Hà Nội thay vì Nghị định toàn quốc — trùng chữ nhiều hơn nhưng sai phạm vi). Cross-encoder
chấm tương đồng câu hỏi–đoạn **không có chỗ biểu diễn thứ bậc pháp lý**. Đó là lý do thêm tham
số (8B) cũng không cứu được. *(Mới đọc 2/13 ca — giả thuyết có ví dụ, chưa phải số liệu.)*

**Câu KHÔNG được viết:** "chúng tôi chứng minh model không còn tín hiệu để khai thác" — không
có bằng chứng cho câu đó. **Câu viết được:** "chúng tôi đo được đóng góp biên của khâu xếp hạng
đã tụt xuống dưới ngưỡng phát hiện của tập dev, nên đóng hướng".
