"""Giao dien web OCR tieng Viet: tai anh -> them nhieu (tuy chon) -> tien xu ly (tuy chon) -> Scan.
Chay: python app.py   roi mo http://127.0.0.1:7860

Lich su sua loi:
  Fix 1 - thieu chu / doc ra rac : che do "Chinh xac cao" (phong to anh, ha nguong phat hien, beam search)
                                   + loc rac (chu <=2 ky tu, do tin cay thap)
  Fix 2 - sai dong              : gop cac o chu cung hang thanh 1 dong, sap xep trai -> phai (vn_postprocess.group_lines)
  Fix 4 - HQ lam mat chu        : chay ca che do thuong + HQ roi gop (HQ chi THEM chu, khong lam mat);
                                   gop dong so voi o gan nhat (bien nghieng); tu dien sua than trong hon
  Fix 3 - chinh ta / in hoa     : sua bang tu dien VinText + thong ke cap tu, giu nguyen kieu IN HOA nhu tren anh
"""
import os, time
import cv2, numpy as np, easyocr, torch
import gradio as gr
from vn_postprocess import group_lines, Lexicon, correct_line

torch.set_num_threads(os.cpu_count() or 4)
reader = easyocr.Reader(["vi"], gpu=False)
VINTEXT_DIR = os.path.join("data", "vintext", "vietnamese")

_lex = None
def get_lexicon():
    global _lex
    if _lex is None:
        _lex = Lexicon(VINTEXT_DIR)
    return _lex


# ---------------- NHIEU (lam kho anh) ----------------
def add_pattern(img):
    h, w = img.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    p = ((np.sin(xx / 6) + np.sin(yy / 9) + np.sin((xx + yy) / 15)) * 40 + 128).astype(np.uint8)
    return cv2.addWeighted(img, 0.75, cv2.applyColorMap(p, cv2.COLORMAP_PINK), 0.25, 0)

def add_watermark(img):
    h, w = img.shape[:2]; wm = img.copy()
    for y in range(0, h, max(h // 4, 40)):
        cv2.putText(wm, "BAN SAO - MAU THU", (10, y), cv2.FONT_HERSHEY_SIMPLEX, w / 600, (180, 180, 180), 3)
    wm = cv2.warpAffine(wm, cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1), (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(img, 0.8, wm, 0.2, 0)

def add_stamp(img):
    h, w = img.shape[:2]; c = (w // 2, h // 2); r = min(h, w) // 6; out = img.copy()
    cv2.circle(out, c, r, (30, 30, 220), max(r // 10, 3))
    cv2.circle(out, c, int(r * 0.75), (30, 30, 220), max(r // 20, 2))
    cv2.putText(out, "DA DUYET", (c[0] - r // 2, c[1] + r // 10), cv2.FONT_HERSHEY_SIMPLEX, r / 90, (30, 30, 220), 3)
    return out

def add_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def add_gauss_noise(img):
    n = np.random.default_rng(0).normal(0, 25, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

NOISE = {"Nền hoa văn": add_pattern, "Watermark chéo": add_watermark, "Con dấu đỏ": add_stamp,
         "Làm mờ (blur)": add_blur, "Nhiễu hạt": add_gauss_noise}

# ---------------- TIEN XU LY (go nhieu) ----------------
def remove_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def denoise(img):
    return cv2.bilateralFilter(img, 9, 50, 50)

def sharpen(img):
    return cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

def binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

PRE = {"Xóa màu đỏ (dấu)": remove_red, "Khử nhiễu": denoise, "Tăng tương phản (CLAHE)": clahe,
       "Làm nét": sharpen, "Nhị phân hóa (đen/trắng)": binarize}

# ---------------- FIX 1 + 4: OCR chinh xac cao = chay 2 lan roi GOP + loc rac ----------------
def _area(b):
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    return min(xs), min(ys), max(xs), max(ys)

def _overlap(a, b):
    """Ti le giao nhau so voi o NHO hon (0..1)."""
    ax1, ay1, ax2, ay2 = _area(a); bx1, by1, bx2, by2 = _area(b)
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    small = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)) or 1
    return iw * ih / small

def run_ocr(img, hq):
    base = reader.readtext(img)                              # lan 1: cai dat mac dinh
    if not hq:
        return base
    # lan 2: phong to nhe (mag_ratio) + nguong phat hien thap hon -> bat them chu nho / chu nghieng
    extra = reader.readtext(img, mag_ratio=1.5, text_threshold=0.6, low_text=0.35,
                            decoder="beamsearch", beamWidth=5)
    merged = list(base)
    for r in extra:                                          # gop: vung trung nhau -> giu ban tu tin hon
        dup = [i for i, m in enumerate(merged) if _overlap(m[0], r[0]) > 0.5]
        if not dup:
            merged.append(r)
        elif all(r[2] > merged[i][2] for i in dup) and len(dup) == 1:
            merged[dup[0]] = r
    return merged

def is_junk(text, conf):
    core = "".join(ch for ch in text if ch.isalnum())
    if not core:
        return True
    if len(core) <= 2 and conf < 0.5:                        # vd "OE (25%)", "77"
        return True
    return False

# ---------------- PIPELINE ----------------
def resize(img, m):
    h, w = img.shape[:2]
    if m and max(h, w) > m:
        f = m / max(h, w); img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img

MIN_CONF = 0.2   # bo cac o chu ma model tu tin < 20% (thuong la rac)

def scan(image, noises, pres, max_side, keyword, hq=True, spell=True):
    if image is None:
        return None, None, "⚠️ Hãy tải ảnh lên trước.", ""
    img = resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), int(max_side))
    for n in NOISE:
        if n in noises: img = NOISE[n](img)
    noisy = img.copy()
    for p in PRE:
        if p in pres: img = PRE[p](img)

    t = time.time()
    raw = run_ocr(img, hq)
    res = [r for r in raw if r[2] >= MIN_CONF and not is_junk(r[1], r[2])]

    # Fix 2: gop theo dong
    groups = group_lines(res)

    vis = img.copy(); lines, changes = [], []
    for g in groups:
        for box, _, _ in g:
            cv2.polylines(vis, [np.array(box, dtype=np.int32)], True, (0, 200, 0), 2)
        text = " ".join(r[1] for r in g)
        # Fix 3: chinh ta
        if spell:
            words, ch = correct_line(text.split(), get_lexicon())
            text = " ".join(words); changes += ch
        if keyword and keyword.lower() not in text.lower():
            continue
        lines.append(text)
    dt = time.time() - t

    to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    info = f"⏱️ {dt:.1f}s · {len(res)}/{len(raw)} vùng chữ (sau lọc) · {len(lines)} dòng"
    if changes:
        info += "\n\n**Đã sửa chính tả:** " + " · ".join(f"`{a}` → `{b}`" for a, b in changes[:15])
    return to_rgb(noisy), to_rgb(vis), "\n".join(lines) or "(không đọc được chữ nào)", info

def scan_batch(files, noises, pres, max_side, keyword, hq, spell, progress=gr.Progress()):
    if not files:
        return [], "⚠️ Hãy tải ảnh lên trước.", None
    import csv, tempfile
    gallery, blocks, rows = [], [], []
    for f in progress.tqdm(files, desc="Đang scan"):
        path = f if isinstance(f, str) else f.name
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            blocks.append(f"=== {name} ===\n(không đọc được file ảnh)"); continue
        _, vis, text, info = scan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), noises, pres, max_side,
                                  keyword, hq, spell)
        gallery.append((vis, name))
        blocks.append(f"=== {name} ===  {info.splitlines()[0]}\n{text}")
        rows.append({"image": name, "text": text.replace("\n", " | "), "info": info.splitlines()[0]})
    out = os.path.join(tempfile.gettempdir(), "ocr_results.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "text", "info"]); w.writeheader(); w.writerows(rows)
    return gallery, "\n\n".join(blocks), out

# ---------------- GIAO DIEN ----------------
with gr.Blocks(title="OCR tiếng Việt - ảnh nhiễu", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔎 OCR tiếng Việt trên ảnh có hoa văn / vật cản\n"
                "Tải ảnh → (tuỳ chọn) thêm nhiễu → (tuỳ chọn) tiền xử lý → **Scan**")

    with gr.Accordion("🧠 Tuỳ chọn đọc chữ (dùng chung cho cả 2 tab)", open=True):
        with gr.Row():
            hq = gr.Checkbox(True, label="Chế độ chính xác cao (quét 2 lần rồi gộp, chậm hơn ~2-3x)")
            spell = gr.Checkbox(True, label="Sửa chính tả (từ điển tiếng Việt)")
        max_side = gr.Slider(400, 2000, 1280, step=64, label="Thu nhỏ ảnh lớn (cạnh dài tối đa, px)")

    with gr.Tab("🖼️ Một ảnh"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Ảnh đầu vào", type="numpy", height=320)
                noises = gr.CheckboxGroup(list(NOISE), label="1️⃣ Thêm nhiễu (để thử độ khó)")
                pres = gr.CheckboxGroup(list(PRE), label="2️⃣ Tiền xử lý (để gỡ nhiễu)")
                keyword = gr.Textbox(label="Chỉ lấy dòng chứa từ khoá (để trống = lấy hết)")
                with gr.Row():
                    btn = gr.Button("🚀 Scan", variant="primary", size="lg", scale=3)
                    reset = gr.Button("🗑️ Reset", variant="secondary", size="lg", scale=1)
            with gr.Column(scale=1):
                out_noisy = gr.Image(label="Ảnh sau khi thêm nhiễu", height=250)
                out_vis = gr.Image(label="Ảnh đưa vào OCR + khung chữ phát hiện", height=250)
                info = gr.Markdown()
                out_text = gr.Textbox(label="📝 Kết quả chữ", lines=10, show_copy_button=True)
        btn.click(scan, [inp, noises, pres, max_side, keyword, hq, spell],
                  [out_noisy, out_vis, out_text, info])
        reset.click(lambda: (None, [], [], "", None, None, "", ""),
                    None, [inp, noises, pres, keyword, out_noisy, out_vis, out_text, info])

    with gr.Tab("📚 Nhiều ảnh"):
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Chọn nhiều ảnh (giữ Ctrl để chọn nhiều)", file_count="multiple", file_types=["image"])
                b_noises = gr.CheckboxGroup(list(NOISE), label="1️⃣ Thêm nhiễu")
                b_pres = gr.CheckboxGroup(list(PRE), label="2️⃣ Tiền xử lý")
                b_keyword = gr.Textbox(label="Chỉ lấy dòng chứa từ khoá (để trống = lấy hết)")
                with gr.Row():
                    b_btn = gr.Button("🚀 Scan tất cả", variant="primary", size="lg", scale=3)
                    b_reset = gr.Button("🗑️ Reset", size="lg", scale=1)
            with gr.Column(scale=1):
                b_gallery = gr.Gallery(label="Ảnh + khung chữ", columns=3, height=300)
                b_text = gr.Textbox(label="📝 Kết quả từng ảnh", lines=12, show_copy_button=True)
                b_csv = gr.File(label="⬇️ Tải kết quả CSV")
        b_btn.click(scan_batch, [files, b_noises, b_pres, max_side, b_keyword, hq, spell],
                    [b_gallery, b_text, b_csv])
        b_reset.click(lambda: (None, [], [], "", [], "", None), None,
                      [files, b_noises, b_pres, b_keyword, b_gallery, b_text, b_csv])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
