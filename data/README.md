# Dữ liệu

Thư mục này cố ý để trống. Kho văn bản và tập câu hỏi do **Ban tổ chức DSC 2026**
cung cấp; repo này không tái phát tán chúng.

## Cần những gì để chạy lại

Đặt vào đúng thư mục này:

```
data/
├── selected-contexts/      8.532 file  context_<doc_id>.json → {link, passage, id}
├── train.json              7.000 câu có nhãn
└── public-official.json    1.000 câu đề thi, answer = null
```

Lấy từ trang cuộc thi DSC 2026 — Task 1 (Legal Information Retrieval).

## Dựng lại các tập dev

Kết quả trong `results/` đo trên hai tập dev khoá cứng, tách ra từ `train.json`.
Repo chỉ kèm **danh sách ID câu hỏi** (`results/dev300_qids.json`,
`results/dev1000_qids.json`) — không kèm nội dung câu hỏi hay nhãn.

```python
import json
train = json.load(open('data/train.json', encoding='utf-8'))
qids  = json.load(open('results/dev300_qids.json'))
dev300 = {q: train[q] for q in qids}
json.dump(dev300, open('data/dev_300_locked.json', 'w', encoding='utf-8'),
          ensure_ascii=False)
```

`dev1000 ⊃ dev300`, và cả hai đều nằm trong `train.json`. Loại `dev1000` khỏi mọi
quá trình huấn luyện là loại hết mọi tập dev.

## Vì sao khoá cứng

Tập dev cố định từ đầu và không bao giờ đổi. Đo đi đo lại trên một tập trôi nổi thì
không so được hai lượt cách nhau vài ngày — và mọi kết luận trong `docs/` đều dựa
trên việc so như vậy.

⚠️ Một tập dev 150 câu đời đầu đã bị **khai tử**: 256/300 câu của nó về sau lọt vào
tập huấn luyện. Nếu bạn tự cắt tập dev, hãy cắt trước khi làm bất cứ gì khác.

## Tạo phẩm trung gian

Không kèm trong repo vì quá lớn (mỗi file 96–335 MB), dựng lại được từ `src/`:

| file | nội dung |
|---|---|
| `bm25_top100_*.json` | rổ BM25 top-100, kèm nội dung đoạn |
| `fusion_rrf_top50_*.json` | rổ ứng viên đang dùng — 50 văn bản/câu, kèm 3 đoạn tốt nhất mỗi văn bản |
| `vec_*_1800.npy` | vector CLS đã cache cho tập train/dev |
