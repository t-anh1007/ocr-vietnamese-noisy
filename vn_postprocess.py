"""Hau xu ly ket qua OCR tieng Viet:
  - group_lines : gop cac o chu cung hang thanh 1 dong, sap xep trai -> phai (fix loi 2)
  - Lexicon     : tu dien + thong ke tu/cap tu lay tu VinText (chi tap train, id 1..1200)
  - correct_line: sua chinh ta tung tu theo tu dien + ngu canh, giu kieu IN HOA/Hoa dau (fix loi 3)
"""
import os, re, pickle, unicodedata
from collections import Counter

# ---------------------------------------------------------------- gop dong
def group_lines(results):
    """results: list (box, text, conf) cua EasyOCR -> list cac dong, moi dong = list (box, text, conf).
    So o chu moi voi o GAN NHAT (theo chieu ngang) trong dong -> chiu duoc bien bi nghieng.
    Nguong lech doc dua tren chieu cao o NHO hon (de khong gop 2 dong co co chu khac nhau),
    nhung khong nho hon 1/2 o lon (de dau "-" rat thap van vao dung dong)."""
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        items.append({"y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys),
                      "x": min(xs), "x2": max(xs), "r": (box, text, conf)})
    items.sort(key=lambda d: (d["y"], d["x"]))
    lines = []
    for it in items:
        best, best_dy = None, None
        for ln in lines:
            near = min(ln, key=lambda o: min(abs(o["x"] - it["x2"]), abs(it["x"] - o["x2"]), abs(o["x"] - it["x"])))
            dy = abs(it["y"] - near["y"])
            thr = 0.5 * max(min(it["h"], near["h"]), 0.5 * max(it["h"], near["h"]))
            if dy < thr and (best is None or dy < best_dy):
                best, best_dy = ln, dy
        if best is None:
            lines.append([it])
        else:
            best.append(it)
    lines.sort(key=lambda l: sum(i["y"] for i in l) / len(l))
    return [[i["r"] for i in sorted(l, key=lambda d: d["x"])] for l in lines]

# ---------------------------------------------------------------- tu dien
def nfc(s):
    return unicodedata.normalize("NFC", s)

def strip_accents(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

PUNCT = "\"'“”‘’.,:;!?()[]{}-–…*/\\"

def clean(w):
    return nfc(w.strip(PUNCT)).lower()

class Lexicon:
    def __init__(self, vintext_dir, cache="lexicon.pkl", train_max_id=1200):
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                self.uni, self.bi = pickle.load(f)
        else:
            self.uni, self.bi = Counter(), Counter()
            for fn in ("vn_dictionary.txt", "general_dict.txt"):
                p = os.path.join(vintext_dir, fn)
                if os.path.exists(p):
                    for line in open(p, encoding="utf-8"):
                        w = clean(line.strip())
                        if w.isalpha():
                            self.uni[w] += 1          # co trong tu dien = it nhat 1
            lab = os.path.join(vintext_dir, "labels")
            for i in range(1, train_max_id + 1):       # CHI dung nhan tap train -> khong "nhin trom" dap an test
                p = os.path.join(lab, f"gt_{i}.txt")
                if not os.path.exists(p):
                    continue
                prev = None
                for line in open(p, encoding="utf-8"):
                    parts = line.strip().split(",", 8)
                    if len(parts) < 9 or parts[8] == "###":
                        prev = None; continue
                    for w in parts[8].split():
                        w = clean(w)
                        if not w.isalpha():
                            prev = None; continue
                        self.uni[w] += 5
                        if prev:
                            self.bi[(prev, w)] += 1
                        prev = w
            with open(cache, "wb") as f:
                pickle.dump((self.uni, self.bi), f)
        self.vocab = set(self.uni)
        self.by_skel = {}                               # "tra" -> {"tra","trà","trá",...}
        for w in self.vocab:
            self.by_skel.setdefault(strip_accents(w), set()).add(w)

    def __len__(self):
        return len(self.vocab)

# ---------------------------------------------------------------- sua chinh ta
def edit1(a, b):
    """True neu khoang cach Levenshtein giua a va b <= 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diff += 1
            if diff > 1:
                return False
            if len(a) == len(b):
                i += 1
            j += 1
        else:
            i += 1; j += 1
    return diff + (len(b) - j) <= 1

def apply_case(src, w):
    letters = [c for c in src if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(letters) > 1:
        return w.upper()
    if letters and letters[0].isupper():
        return w[:1].upper() + w[1:]
    return w

def candidates(w, lex):
    """Tu ung vien cho tu w (da lower). Tra ve list (word, cost)."""
    out = {}
    for c in lex.by_skel.get(strip_accents(w), ()):     # cung bo khung, chi khac dau: LAT -> lạt/lát
        out[c] = 0.5
    skel = strip_accents(w)
    if len(w) >= 3:                                     # sai 1 ky tu (them/bot/thay)
        for s, ws in lex.by_skel.items():
            if edit1(skel, s):
                for c in ws:
                    out.setdefault(c, 1.0)
    return list(out.items())

def score(c, cost, prev, nxt, lex):
    import math
    s = math.log(lex.uni[c] + 1) - 3 * cost
    if prev: s += 3 * math.log(lex.bi[(prev, c)] + 1)
    if nxt:  s += 3 * math.log(lex.bi[(c, nxt)] + 1)
    return s

MIN_FREQ = 5          # ung vien phai tung xuat hien trong nhan train (uni >= 5), khong chi co trong tu dien

def correct_line(words, lex):
    """Sua chinh ta THAN TRONG. Tra ve (list tu da sua, list (cu, moi)).
    Quy tac:
      - Tu da co trong tu dien: chi doi dau (cung khung chu) khi ngu canh ung ho ro rang
        (vd TRA + SỮA -> TRÀ vi cap "trà sữa" hay gap, con "tra sữa" chua gap bao gio).
      - Tu khong co trong tu dien: sua dau thoai mai (loai -> loại);
        sua sai ky tu (them/bot/thay 1 chu) chi khi tu dai >= 5 ky tu.
      - Tu <= 3 ky tu khong co trong tu dien, so, ky hieu: giu nguyen (vd "Qat" khong doan bua thanh "Mặt").
    """
    import math
    def opts(x):                                        # (tu, so loi) cho 1 manh khi tach
        if x in lex.vocab and lex.uni[x] >= MIN_FREQ:
            return [(x, 0)]
        return [(c, 1) for c, cost in candidates(x, lex) if cost == 1.0 and lex.uni[c] >= MIN_FREQ]

    toks = []
    for w in words:                                     # tach dinh lien: "TRAASỮA" -> "TRA" + "SỮA"
        core = clean(w)
        if core.isalpha() and core not in lex.vocab and len(core) >= 6:
            best = None
            for k in range(2, len(core) - 1):
                oa, ob = opts(core[:k]), opts(core[k:])
                if oa and ob:
                    # uu tien: it loi nhat -> loi "thua ky tu" (OCR hay nhan doi chu khi o dinh nhau)
                    # -> hay di cung manh kia (thong ke cap tu) -> pho bien
                    def key(o, other, left):
                        shorter = len(strip_accents(o[0])) < len(strip_accents(core[:k] if left else core[k:]))
                        pair = lex.bi[(o[0], other)] if left else lex.bi[(other, o[0])]
                        return (-o[1], shorter, pair, lex.uni[o[0]])
                    b0 = max(ob, key=lambda o: (-o[1], lex.uni[o[0]]))[0]
                    (a, ea) = max(oa, key=lambda o: key(o, b0, True))
                    (b, eb) = max(ob, key=lambda o: key(o, a, False))
                    sc = -1000 * (ea + eb) + lex.uni[a] + lex.uni[b]
                    if ea + eb <= 1 and (not best or sc > best[0]):
                        best = (sc, apply_case(w[:k], a), apply_case(w[k:], b))
            if best:
                toks += [best[1], best[2]]; continue
        toks.append(w)

    def ctx(c, prev, nxt):
        return (lex.bi[(prev, c)] if prev else 0) + (lex.bi[(c, nxt)] if nxt else 0)

    out, changes = [], []
    low = [clean(t) for t in toks]
    for i, t in enumerate(toks):
        core = low[i]
        prev = clean(out[-1]) if out else None
        nxt = low[i + 1] if i + 1 < len(toks) else None
        new = None
        if not core.isalpha() or len(core) < 2:
            pass
        elif core in lex.vocab:
            alts = [c for c in lex.by_skel.get(strip_accents(core), ()) if c != core and lex.uni[c] >= MIN_FREQ]
            if alts and ctx(core, prev, nxt) == 0:
                b = max(alts, key=lambda c: ctx(c, prev, nxt))
                if ctx(b, prev, nxt) >= 2:
                    new = b
        else:
            cands = [(c, cost) for c, cost in candidates(core, lex)
                     if lex.uni[c] >= MIN_FREQ and (cost < 1.0 or len(core) >= 5)]
            if cands:
                new = max(cands, key=lambda cc: math.log(lex.uni[cc[0]]) + 3 * math.log(ctx(cc[0], prev, nxt) + 1) - 3 * cc[1])[0]
        if new and new != core:
            m = re.match(r"^(\W*)(.*?)(\W*)$", t)
            fixed = m.group(1) + apply_case(m.group(2), new) + m.group(3)
            out.append(fixed); changes.append((t, fixed))
        else:
            out.append(t)
    return out, changes
