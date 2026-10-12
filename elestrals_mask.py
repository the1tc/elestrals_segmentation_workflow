#!/usr/bin/env python3
"""No-foil mask for a whole card picture with text and a frame printed over the art (Elestrals
full arts; made on one card, Aeolus BS1-097, 1468 x 2048). The matte models fail on such a card
as it is: the text block reads as foreground. So the card is taken apart first:

  1. text and black border by colour rules, no model (`frame_masks`);
  2. the text is painted out of the picture (inpaint) and the picture is regraded (`VARIANTS`);
  3. the rembg models matte the character on that (`RECIPE`), and the parts are joined;
  4. the bottom of the card is the same on every card and never has a character on it, so there
     the layout's fixed mask (`FIXED`) is the last word: bar, gem, inset and the black band with
     the credits are no foil, the slivers of art around the bar are foil.

Output: 8-bit grey, white = no foil (text, border, bar, character), black = foil.

Usage: elestrals_mask.py frame <card.png> [--out DIR]   steps 1 and 4 only, with overlays; no model, fast
       elestrals_mask.py sweep <card.png> [--out DIR]   every variant x every model, one sheet per model
       elestrals_mask.py build <card.png> [--out DIR]   the mask, with RECIPE
       elestrals_mask.py fixed <card.png>               write FIXED again from a card and the bar trace (BAR)

sweep and build run the segmentation models on the CPU; birefnet-general takes about 10 GB of RAM.
The models are downloaded to ~/.u2net on first use.
"""
import argparse
import gc
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_MODELS = ["isnet-anime", "isnet-general-use", "u2net", "birefnet-general-lite",
                "birefnet-general", "birefnet-portrait", "birefnet-massive"]
# (variant, model, threshold) triples whose mattes are joined; chosen from the sweep sheets.
# Aeolus, 2026-10-12, 14 variants x 7 models: the full birefnet-general on the card without its
# text has the whole figure, legs behind the rules text included. Gamma, saturation and contrast
# move its matte by 1 to 2%; a median blur makes every model worse. birefnet-general-lite and
# -massive lose the legs under the text, isnet-anime half the cape, birefnet-portrait everything.
RECIPE = [("notext", "birefnet-general", 128)]

# The fixed mask of the layout: 255 no foil, 0 foil, 128 = not fixed, decided per card.
FIXED = os.path.join(ROOT, "frames", "ElestralsFullArt.png")
# The bottom bar with its gem, white on black: traced by hand by klappec on Aeolus (2026-10-12).
BAR = os.path.join(ROOT, "frames", "ElestralsBarTrace.png")


def disc(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def grow(mask, r):
    return cv2.dilate(mask.astype(np.uint8), disc(r)).astype(bool)


def fixed_mask(rgb):
    """FIXED from one card: the traced bar, and from where the bar reaches its full width down
    the card's black (credits filled in)."""
    bar = np.array(Image.open(BAR).convert("L")) > 127
    side = np.argmax(bar[:, np.flatnonzero(bar.any(0))[0]])
    black = rgb.max(2) < 20
    lab, n = ndi.label(~black)                                    # letters are small holes in the black
    black |= np.isin(lab, np.flatnonzero(ndi.sum(~black, lab, np.arange(1, n + 1)) < 3000) + 1)
    out = np.full(bar.shape, 128, np.uint8)
    out[side:] = np.where(black[side:], 255, 0)
    out[bar] = 255
    return out


def frame_masks(rgb):
    """Text, border and the area to paint out for the models (the letters whole, 9 px around
    their fill) as boolean masks. Numbers are for a 1468 x 2048 card."""
    mx = rgb.max(2).astype(int)
    sat = mx - rgb.min(2).astype(int)

    # text: a white patch whose rim is black. Clouds and stars are as white, but sit on colour.
    white = (rgb.min(2) > 160) & (sat < 60)
    dark = (mx < 70) & (sat < 60)
    lab, n = ndi.label(white)
    idx = np.arange(1, n + 1)
    rim = ndi.grey_dilation(lab, footprint=disc(3)) * ~white      # each patch's 3 px rim, labelled
    rim_n = np.maximum(ndi.sum(rim > 0, rim, idx), 1)
    size = ndi.sum(white, lab, idx)
    keep = (size > 25) & (size < 8000) & (ndi.sum(dark, rim, idx) / rim_n > 0.35)
    # with the outline and its glow: the dark within 7 px. What shows through a letter (the
    # middle of an o) is art and stays foil (klappec, 2026-10-12).
    fill = np.isin(lab, idx[keep])
    text = cv2.morphologyEx((fill | (grow(fill, 7) & (mx < 110))).astype(np.uint8), cv2.MORPH_CLOSE, disc(2)).astype(bool)

    # border: black that reaches the edge of the picture
    lab, _ = ndi.label(mx < 20)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return text, np.isin(lab, edge[edge > 0]), grow(fill, 9)


def painted_out(rgb, area):
    """The card without its text, for the models."""
    return cv2.inpaint(rgb, area.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)


def gamma(g):
    return lambda a: (255 * (a / 255.0) ** g).astype(np.uint8)


def contrast(c):
    return lambda a: np.clip((a.astype(float) - 128) * c + 128, 0, 255).astype(np.uint8)


def smooth(a):
    """Median blur: takes the printed foil specks and small stars out of the background."""
    return cv2.medianBlur(a, 15)


def saturate(s):
    def f(a):
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(float)
        hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return f


def clahe(a):
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# every variant starts from the card with the text painted out, except "raw"
VARIANTS = {
    "raw": None,
    "notext": lambda a: a,
    "smooth": smooth,
    "gamma0.6": gamma(0.6),
    "gamma1.6": gamma(1.6),
    "gamma2.2": gamma(2.2),
    "contrast1.5": contrast(1.5),
    "smooth+gamma1.6": lambda a: gamma(1.6)(smooth(a)),
    "smooth+contrast1.5": lambda a: contrast(1.5)(smooth(a)),
    "saturate2": saturate(2.0),
    "clahe": clahe,
    # "noframe...": border and bar painted over with the art's median colour first
    "noframe": lambda a: a,
    "noframe+smooth": smooth,
    "noframe+gamma1.6": gamma(1.6),
}


def variant(name, rgb, clean, frame):
    if VARIANTS[name] is None:
        return rgb
    if name.startswith("noframe"):
        clean = np.where(frame[..., None], np.median(clean[~frame], axis=0).astype(np.uint8), clean)
    return VARIANTS[name](clean)


def overlay(rgb, mask):
    """The card with the foil area tinted magenta."""
    out = rgb.astype(float)
    foil = mask < 128
    out[foil] = out[foil] * 0.35 + np.array([255, 0, 255]) * 0.65
    return Image.fromarray(out.astype(np.uint8))


def model_masks(pictures, models):
    """{(picture name, model): matte}. One model in memory at a time: together they need
    more RAM than most machines have."""
    from rembg import new_session, remove
    out = {}
    for m in models:
        session = new_session(m, providers=["CPUExecutionProvider"])
        for name, pic in pictures.items():
            matte = np.array(remove(Image.fromarray(pic), session=session, only_mask=True).convert("L"))
            if m == "isnet-anime":                                # it answers frames and floors with grey
                matte[matte < 245] = 0
            out[name, m] = matte
            print(m, name, flush=True)
        del session
        gc.collect()
    return out


def sheet(tiles, cols, path, width=367):
    """Contact sheet of (label, PIL image)."""
    w, h = width, round(width * tiles[0][1].height / tiles[0][1].width)
    out = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * (h + 22)), "black")
    draw = ImageDraw.Draw(out)
    for i, (label, im) in enumerate(tiles):
        x, y = i % cols * w, i // cols * (h + 22)
        draw.text((x + 4, y + 4), label, fill="white")
        out.paste(im.resize((w, h)), (x, y + 22))
    out.save(path, quality=88)
    print(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["frame", "sweep", "build", "fixed"])
    p.add_argument("card")
    p.add_argument("--out", default=os.path.join(ROOT, "out"))
    p.add_argument("--models", nargs="+", default=SWEEP_MODELS)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.card))[0]
    rgb = np.array(Image.open(args.card).convert("RGB"))
    if args.mode == "fixed":
        os.makedirs(os.path.dirname(FIXED), exist_ok=True)
        Image.fromarray(fixed_mask(rgb)).save(FIXED)
        return print(FIXED)
    layout = np.array(Image.open(FIXED).convert("L"))
    layout = np.where(layout > 191, 255, np.where(layout < 64, 0, 128))   # a hand edit may leave soft edges
    text, border, letters = frame_masks(rgb)
    frame = border | (layout == 255)
    fixed = text | frame
    clean = painted_out(rgb, letters)
    print(f"text {text.mean():.1%}  frame {frame.mean():.1%}")

    def final(m):                                                 # step 4
        return np.where(layout == 128, m, layout).astype(np.uint8)

    if args.mode == "frame":
        overlay(rgb, final(fixed * 255)).save(os.path.join(args.out, f"{name}_frame.jpg"))
        Image.fromarray(clean).save(os.path.join(args.out, f"{name}_notext.jpg"))
        return

    if args.mode == "sweep":
        pics = {v: variant(v, rgb, clean, frame) for v in args.variants}
        masks = model_masks(pics, args.models)
        np.savez_compressed(os.path.join(args.out, f"{name}_sweep.npz"),
                            **{f"{v}|{m}": k for (v, m), k in masks.items()})
        for m in args.models:
            tiles = [(v, overlay(rgb, final(np.maximum(masks[v, m], fixed * 255)))) for v in args.variants]
            sheet(tiles, 6, os.path.join(args.out, f"{name}_sweep_{m}.jpg"))
        return

    pics = {v: variant(v, rgb, clean, frame) for v in {v for v, _, _ in RECIPE}}
    masks = model_masks(pics, sorted({m for _, m, _ in RECIPE}))
    char = np.zeros(fixed.shape, bool)
    for v, m, t in RECIPE:
        char |= masks[v, m] >= t
    # the model takes the bar for foreground and its matte runs a few px over the bar's edge onto
    # the art (a 15 px wedge beside the gem on Aeolus). Of the matte off the fixed mask, keep only
    # the pieces that reach further than 16 px from it: a boot does, the spill does not.
    lab, n = ndi.label(char & (layout == 128))
    far = np.unique(lab[~grow(layout == 255, 16)])
    char &= np.isin(lab, far[far > 0])
    lab, n = ndi.label(~char)                                     # pinholes between rope strands
    char |= np.isin(lab, np.flatnonzero(ndi.sum(~char, lab, np.arange(1, n + 1)) < 0.001 * char.size) + 1)
    mask = final((char | fixed) * 255)
    Image.fromarray(mask).save(os.path.join(args.out, f"{name}.png"))
    overlay(rgb, mask).save(os.path.join(args.out, f"{name}_overlay.jpg"))
    print(f"no foil {(mask > 0).mean():.1%}")


if __name__ == "__main__":
    main()
