"""Chon N anh (ngau nhien, hoac tu chon bang --ids) tu tap test VinText + doc nhan (ground truth).
VinText label: moi dong = x1,y1,...,x4,y4,TRANSCRIPT ; '###' = vung khong doc duoc (bo qua).
Dung: python pick_samples.py --vintext data/vintext --n 10
"""
import argparse, json, re, shutil, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--vintext", default="data/vintext", help="thu muc da giai nen (chua test_image/ va labels/)")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", default="samples")
ap.add_argument("--ids", nargs="*", help="so thu tu anh muon chon, vd: --ids 1201 1250 1300")
ap.add_argument("--seed", type=int, default=42, help="doi so nay de ra bo anh ngau nhien khac")
args = ap.parse_args()

root = Path(args.vintext)
imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg") and "test" in str(p.parent).lower()])
labels = {re.findall(r"\d+", p.stem)[-1].lstrip("0"): p for p in root.rglob("gt_*.txt")}
assert imgs, "Khong tim thay anh test - kiem tra lai duong dan --vintext"

random.seed(args.seed)
random.shuffle(imgs)
if args.ids:
    want = {i.lstrip("0") for i in args.ids}
    imgs = [p for p in imgs if re.findall(r"\d+", p.stem)[-1].lstrip("0") in want]
    args.n = len(imgs)
out = Path(args.out); out.mkdir(exist_ok=True)
gt = {}
for img in imgs:
    num = re.findall(r"\d+", img.stem)[-1].lstrip("0")
    if num not in labels:
        continue
    words = []
    for line in labels[num].read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",", 8)
        if len(parts) == 9 and parts[8] != "###":
            words.append(parts[8])
    if len(words) < 2:
        continue
    shutil.copy(img, out / img.name)
    gt[img.name] = words
    if len(gt) >= args.n:
        break
(out / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Da chon {len(gt)} anh -> {out}/")
