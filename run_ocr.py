"""So sanh OCR tho vs OCR sau tien xu ly.
Dung: python run_ocr.py --dir samples_noisy
Metric: word recall = % tu dung (theo ground truth) ma OCR doc ra dung y het (ca dau).
"""
import argparse, csv, json, os, unicodedata
import cv2, numpy as np, easyocr
from pathlib import Path

def norm(s):  # chuan hoa unicode + chu thuong
    return unicodedata.normalize("NFC", s).lower().strip(".,:;!?\"'()")

def remove_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    out = img.copy(); out[m > 0] = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)[m > 0]
    return out

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def denoise(img):
    return cv2.bilateralFilter(img, 9, 50, 50)

PIPELINES = {
    "raw": lambda x: x,
    "remove_red": remove_red,
    "red+clahe": lambda x: clahe(remove_red(x)),
    "red+denoise+clahe": lambda x: clahe(denoise(remove_red(x))),
}

def recall(gt_words, ocr_words):
    pool = [norm(w) for t in ocr_words for w in t.split()]
    hit = 0
    for w in (norm(g) for g in gt_words):
        if w in pool:
            hit += 1; pool.remove(w)
    return hit / len(gt_words)

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="samples_noisy")
ap.add_argument("--pipelines", nargs="*", default=list(PIPELINES), help="chon cach xu ly, vd: --pipelines raw remove_red")
ap.add_argument("--max-side", type=int, default=1024, help="thu nho anh de canh dai nhat <= so nay (0 = giu nguyen)")
args = ap.parse_args()
PIPELINES = {k: PIPELINES[k] for k in args.pipelines}
import torch; torch.set_num_threads(os.cpu_count() or 4)

def resize(img, m):
    h, w = img.shape[:2]
    if m and max(h, w) > m:
        f = m / max(h, w); img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img
d = Path(args.dir)
gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
reader = easyocr.Reader(["vi"], gpu=False)
Path("debug").mkdir(exist_ok=True)

rows, totals = [], {k: [] for k in PIPELINES}
import time; t0 = time.time()
for name, words in gt.items():
    img = resize(cv2.imread(str(d / name)), args.max_side)
    row = {"image": name, "n_words": len(words)}
    for pname, fn in PIPELINES.items():
        proc = fn(img)
        cv2.imwrite(f"debug/{Path(name).stem}_{pname}.jpg", proc)
        ocr = reader.readtext(proc, detail=0)
        r = recall(words, ocr)
        row[pname] = round(r, 3); totals[pname].append(r)
        if pname == "raw":
            row["gt"] = " | ".join(words); row["ocr_raw"] = " | ".join(ocr)
    row.setdefault("gt", " | ".join(words))
    rows.append(row); print(f"[{time.time()-t0:.0f}s]", name, {k: row[k] for k in PIPELINES}, flush=True)

with open(f"results_{d.name}.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print("\n=== Word recall trung binh ===")
for k, v in totals.items():
    print(f"{k:20s} {np.mean(v)*100:.1f}%")
