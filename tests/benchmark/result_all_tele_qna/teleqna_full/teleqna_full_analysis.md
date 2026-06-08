# Phân tích chi tiết run benchmark `teleqna_full`

**Nguồn:** [tests/benchmark/result_all_tele_qna/teleqna_full/](../tests/benchmark/result_all_tele_qna/teleqna_full/)
**Thời điểm:** 2026-05-16 12:08 → 2026-05-17 17:59 (~30h, có resume)
**Cấu hình:** `mode=fixed` · model `qwen3:14b` · `think=False` · 10 000/10 000 câu (TeleQnA full)

---

## 1. Tóm tắt kết quả

| Chỉ số | Giá trị |
|---|---|
| Tổng câu | 10 000 |
| **Accuracy tổng** | **70.31 %** (7 031 / 10 000) |
| Errors (exception) | 0 |
| Extraction failure (`extracted_choice = None`) | 191 (1.91 %) |
| Avg latency | 5 681 ms · p50 5 512 · p95 7 397 · p99 8 752 · max 96 436 |
| Graph retrieval thành công (`graph_count > 0`) | 3 555 / 10 000 (35.55 %) |
| Graph errors (Cypher invalid / forbidden) | 41 |

> Batch cuối (3 298 câu resume): 71.04 % — sai số trong-run ~0.7 pp giữa batch và toàn cục, tức ổn định.

---

## 2. Accuracy theo category

| Category | n | Accuracy | Avg latency | Avg sources | Avg graph | Graph hit-rate | Ext-fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Lexicon** | 500 | **88.8 %** | 5 716 ms | 1.84 | 4.15 | 51.2 % | 0 |
| Standards overview | 1 000 | 69.8 % | 6 086 ms | 1.35 | 3.85 | 49.0 % | 3 |
| Research publications | 4 500 | 69.5 % | 5 411 ms | 0.45 | 2.15 | 27.7 % | 105 |
| Standards specifications | 2 000 | 69.2 % | 6 456 ms | 2.62 | 4.25 | 53.5 % | 10 |
| Research overview | 2 000 | 68.8 % | 5 303 ms | 0.36 | 1.90 | 24.7 % | 73 |

Nhận xét:
- **Lexicon vượt trội (+19 pp so với phần còn lại)** — câu kiểu "What does X stand for?" thuộc vùng mạnh của graph (Term node có `abbreviation`/`full_name`), pipeline khớp Pattern A rất tốt.
- Hai nhóm **Research** chỉ đạt ~27 % graph hit-rate — KG không có nội dung research publication, retrieval gần như rơi về parametric knowledge của LLM.
- Hai nhóm **Standards** có graph hit-rate ~50 % nhưng accuracy không vượt Lexicon → bottleneck nằm ở **rerank/answer** chứ không ở recall.

---

## 3. Đóng góp của RAG (RAG-lift)

Chia 10 000 câu theo việc có hay không retrieval (`sources_count>0 OR graph_count>0`):

| Subset | n | Accuracy |
|---|---:|---:|
| Có retrieval | 4 838 | 70.7 % |
| Không retrieval (parametric-only) | 5 162 | 70.0 % |
| **RAG-lift tổng** | | **+0.7 pp** |

Tách riêng **Standards specifications** (đúng vùng KG phục vụ):

| Subset | n | Accuracy |
|---|---:|---:|
| Standards specs có retrieval | 1 563 | 71.6 % |
| Standards specs không retrieval | 437 | 60.6 % |
| **RAG-lift trên Standards specs** | | **+11.0 pp** |

→ **RAG chỉ thực sự "kéo" được accuracy ở vùng Standards specs**. Trên 5 162 câu không có retrieval nào (51.6 %), Qwen3-14B vẫn đúng 70 % bằng knowledge sẵn có — đây là baseline "free" cần trừ đi khi đánh giá đóng góp của hệ thống.

Trong category Standards specs, accuracy tăng đều theo `sources_count`:

| sources_count | n | Accuracy |
|---:|---:|---:|
| 0 | 704 | 60.7 % |
| 1 | 194 | 69.1 % |
| 2 | 174 | 66.7 % |
| 3 | 140 | 74.3 % |
| 4 | 156 | 71.8 % |
| 5 | 140 | 73.6 % |
| **6** | 492 | **78.9 %** |

→ Khi rerank trả đủ 6 chunk (max top-k của pipeline), accuracy lên ~79 %. Đây là kết quả tốt nhất có thể kỳ vọng từ kiến trúc hiện tại.

---

## 4. Pattern Cypher generator: tỉ lệ chạm

Phân bố `graph_count`:

- `0` → **6 445** câu (Pattern C sentinel, hoặc graph_error)
- `8` → **3 364** câu (Pattern A/B trả đủ default LIMIT)
- 1-7 lẻ (~210) — query khớp được vài kết quả nhưng dưới `top_k`

Trong nhóm Standards specifications, phân bố `graph_count = 0` là 1 440 / 2 000 (72 %) — tức gần ba phần tư câu Standards rơi vào nhánh **chỉ-vector** (Pattern C hoặc graph rỗng). Đây là không gian cải thiện chính: nâng recall của Cypher generator ở vùng câu hỏi không có term/section anchor rõ ràng.

---

## 5. Lỗi Cypher (`graph_error`)

41 graph errors tổng cộng. Nhóm theo loại:

| Loại | Số lần |
|---|---:|
| `Forbidden keyword 'SET'` | 30 |
| `CypherSyntaxError` | 5 |
| `Forbidden keyword 'DROP'` | 4 |
| `Forbidden keyword 'MERGE'` | 2 |

→ LLM thỉnh thoảng vẫn sinh **write clause** dù prompt yêu cầu read-only. Filter `_WRITE_CLAUSES` ở [rag-engine/main.py](../rag-engine/main.py) đang **chặn đúng** (không có write nào vào graph). Nhưng tỉ lệ "SET" cao gợi ý prompt template chưa nhấn đủ mạnh — chấp nhận được vì chỉ 0.41 % câu hỏi bị ảnh hưởng.

---

## 6. Extraction failure (191 câu)

Phân bố theo category:

- Research publications: 105
- Research overview: 73
- Standards specifications: 10
- Standards overview: 3
- Lexicon: 0

Đặc điểm: response rất ngắn (avg 99 ký tự), thường là `"Context does not cover the question."` hoặc giải thích vì sao không có context. Mô hình **bỏ qua hẳn việc chọn A/B/C/D/E** thay vì đoán bừa. Đa số rơi vào Research — nơi context rỗng vì KG không có research papers.

Quick win: chèn instruction "If unsure, output a best-guess letter; never refuse" vào answer prompt — kỳ vọng nâng accuracy ~0.4 pp (giả định guess đúng 1/5 thì ~38 câu thêm).

---

## 7. Hallucinated citations

**4 008 / 10 000 câu** (40.08 %) — response chứa `[ts_XX.Y §Z]` nhưng `sources_count = 0` AND `graph_count = 0`. Accuracy trên nhóm này là **74.1 %** (cao hơn trung bình).

Lưu ý: `sources_count = 0` **không có nghĩa là vector search không chạy**. Vector search luôn chạy với `top_k = 10` ở [`orchestrator.py:223`](../../../rag-engine/pipeline/orchestrator.py#L223). Trường hợp này xảy ra do **cross-encoder rerank gán logit < 0 cho tất cả 10 vector hits → floor cắt sạch → `reranked = []` → LLM context rỗng**. Kiểm chứng qid=2 (`debug.log`): `retrieval_vector` count = 10 (top-1 `ts_45_914 §Performance of optimized user diversity`), `rerank` output = 0, nhưng response cite `[ts_38.201 §5.1.1.1]` — spec_id khác hoàn toàn vector top-1. Vậy citation là **parametric/imagined**, không phải copy từ vector hits bị cắt.

Rủi ro về **trust/faithfulness** nghiêm trọng hơn là về accuracy:

- Người dùng nhìn citation → tin tưởng → click vào không có nội dung tương ứng.
- Đề xuất 2 lớp post-check: (1) nếu `graph_count == 0`, nới rerank floor để giữ top-1/top-2 vector hits làm fallback context; (2) nếu sau đó vẫn `reranked == []`, strip `[ts_…]` khỏi response hoặc gắn nhãn "no retrieval".

---

## 8. Confusion matrix (gold → pred)

```
gold\pred   A     B     C     D     E    None
   A     1560   163   155   162   124    46
   B      253  1461   184   147    92    42
   C      189   129  1585   135    84    47
   D      158   132   155  1519   147    42
   E       88    92   102    87   906    14
```

- Diagonal dominant — không có bias chọn lệch về A hay E.
- Hơi yếu ở E (906/1 289 = 70.3 %) — option "E. None of the above / context does not cover" dễ bị model né.
- Off-diagonal lớn nhất là B→A (253) — model có xu hướng chọn A khi do dự giữa A và B.

---

## 9. Accuracy theo 3GPP Release

Trên 1 810 câu có tag `[3GPP Release N]`:

| Release | n | Accuracy |
|---|---:|---:|
| 14 | 139 | 65.5 % |
| 15 | 54 | 68.5 % |
| **16** | **87** | **56.3 %** ⚠ |
| 17 | 733 | 73.7 % |
| 18 | 780 | 71.5 % |
| 19 | 17 | 82.4 % |

→ **Release 16 thấp bất thường (56 %)**. Có thể: corpus `processed_json_v4` thiếu/yếu rel16, hoặc câu hỏi rel16 thiên về vùng mismatch (NR features mà KG chưa đầy đủ). Đáng kiểm tra: đếm chunk theo `Document.spec_id`/release để xác nhận distribution.

Câu nhắc đến chuẩn không phải 3GPP (IEEE/IETF/802.15.4...): 778 câu, accuracy **65.8 %** — yếu hơn trung bình, hợp lý vì KG chỉ chứa 3GPP.

---

## 10. Latency outliers

7 câu vượt 20 s, trong đó 5 câu **trên 90 s**:

| qid | Latency | Category | graph | src | correct |
|---:|---:|---|---:|---:|:---:|
| 9243 | 96.4 s | Standards overview | 8 | 1 | ✓ |
| 6540 | 96.1 s | Standards specs | 8 | 1 | ✓ |
| 4223 | 95.4 s | Standards specs | 8 | 0 | ✓ |
| 5219 | 95.1 s | Standards specs | 0 | 0 | ✗ |
| 9259 | 94.7 s | Standards overview | 0 | 0 | ✓ |
| 3672 | 31.0 s | Lexicon | 60 | 0 | ✓ |
| 0 | 21.5 s | Standards specs | 8 | 6 | ✓ |

5/7 outlier đến từ Standards — nhiều khả năng generation phase dài (qwen3 stream chậm) hơn là retrieval. qid 3672 có `graph_count=60` đặc biệt — pattern B (regex section_title) khớp quá rộng.

`p95 = 7.4 s` và `p99 = 8.8 s` cho thấy 99 % câu hoàn thành dưới 9 s — outlier chỉ là long tail và không tác động UX với chế độ batch.

---

## 11. Điểm cải thiện đề xuất (ưu tiên giảm dần)

1. **Tăng graph recall ở Standards specs khi không có Term anchor** (72 % câu Standards có `graph_count=0`). Hiện Pattern C sentinel quá thường gặp; có thể thêm Pattern D: regex trên `section_title` cho keyword 2-3 từ khi vector top-1 trả về section title rõ ràng.
2. **Bù 191 câu extraction-fail** bằng instruction "always output a letter" — +0.4 pp accuracy gần như miễn phí.
3. **Audit corpus Release 16** — tại sao accuracy 56 % so với 73 % của rel 17/18. Nếu thiếu chunk thì rebuild với input bổ sung; nếu chunk có mà retrieval miss thì điều chỉnh Cypher prompt.
4. **Chặn hallucinated citations** (~40 % response): nếu `sources_count==0 AND graph_count==0` thì strip citation hoặc tag rõ "answer is unverified by retrieval".
5. **Kéo answer prompt** để chọn E (None of the above) đúng hơn — gold E hiện chỉ đúng 70.3 % so với trung bình 70.31 % nhưng tỉ trọng dataset E (12.9 %) thấp, mỗi pp ở đây ~13 câu.

---

## 12. Files trong run

| File | Kích thước | Mô tả |
|---|---:|---|
| `results.jsonl` | 7.1 MB | 1 dòng JSON / câu — input cho mọi phân tích downstream |
| `results.json` | 8.1 MB | Bản pretty-print, tương đương `results.jsonl` |
| `analysis.md` | 1.2 KB | Tóm tắt do runner tự sinh |
| `debug.log` | **1.03 GB** | Full SSE trace + Cypher generated — chỉ mở khi cần debug 1 câu |
| `run.log` | 1.1 MB | Progress per-question, accuracy chạy theo batch |
