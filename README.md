# OCR tiếng Việt trên ảnh có hoa văn / vật cản

Bài thực hành: dùng model OCR có sẵn (**EasyOCR**, không tự train) để đọc chữ tiếng Việt trên ảnh nhiễu
(nền hoa văn, watermark, con dấu đỏ, mờ, nhiễu hạt), kèm tiền xử lý ảnh và hậu xử lý chữ.

## Tính năng
- Giao diện web (Gradio): tải 1 ảnh hoặc nhiều ảnh → thêm nhiễu → tiền xử lý → **Scan** → xuất chữ / CSV
- **Chế độ chính xác cao**: quét 2 lần (mặc định + phóng to/ngưỡng thấp) rồi gộp, giữ bản đọc tự tin hơn
- **Gộp chữ theo dòng** đúng thứ tự đọc, chịu được biển bị nghiêng
- **Sửa chính tả tiếng Việt** bằng từ điển + thống kê cặp từ từ nhãn train VinText, giữ kiểu IN HOA
- Lọc rác (chữ ≤ 2 ký tự độ tin cậy thấp)
- Script benchmark: đo % từ đọc đúng trên tập ảnh có nhãn

## Cài đặt (Windows, Python 3.12)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dữ liệu: VinText (VinAI)
Tải bản *Original* tại https://github.com/VinAIResearch/dict-guided (mục Dataset), giải nén để có:
```
data/vintext/vietnamese/{test_image, unseen_test_images, train_images, labels, vn_dictionary.txt, general_dict.txt}
```
> Dataset chỉ dùng cho nghiên cứu/học tập và **không được phát tán lại**, nên repo này không chứa dữ liệu.

## Chạy
**Giao diện web**
```
python app.py        # mở http://127.0.0.1:7860
```
**Benchmark bằng script**
```
python pick_samples.py --ids 1518 1984 1997     # hoặc --n 10 [--seed 7]
python add_noise.py
python run_ocr.py --dir samples --pipelines raw remove_red
python run_ocr.py --dir samples_noisy --pipelines raw remove_red
```

## Cấu trúc
| File | Vai trò |
|---|---|
| `app.py` | Giao diện web, pipeline nhiễu → tiền xử lý → OCR → gộp dòng → sửa chính tả |
| `vn_postprocess.py` | Gộp dòng, từ điển, sửa chính tả tiếng Việt |
| `pick_samples.py` | Chọn ảnh test VinText + đọc nhãn |
| `add_noise.py` | Thêm hoa văn, watermark, dấu đỏ |
| `run_ocr.py` | Benchmark word recall theo từng cách tiền xử lý |

## Nhật ký sửa lỗi
1. Thiếu chữ / đọc ra rác → chế độ chính xác cao + lọc rác
2. Chữ cùng hàng bị xuống dòng → gộp ô chữ theo dòng
3. Sai chính tả / mất kiểu in hoa → sửa bằng từ điển VinText, giữ IN HOA
4. Chế độ chính xác cao làm mất chữ, gộp nhầm dòng, từ điển sửa sai → quét 2 lần rồi gộp; gộp dòng theo ô gần nhất; sửa chính tả thận trọng

## Nhận xét
- Tiền xử lý không phải lúc nào cũng tốt: "xóa màu đỏ" xóa luôn chữ đỏ thật trên biển hiệu.
- Chữ viết tay / font nghiêng nghệ thuật vẫn là điểm yếu của EasyOCR.
