#!/usr/bin/env python3
"""No-foil mask for a whole card picture with text and a frame printed over the art (Elestrals
full arts; made on one card, Aeolus BS1-097, 1468 x 2048). The matte models fail on such a card
as it is: the text block reads as foreground. So the card is taken apart first:

  1. text and black border by colour rules, no model (`frame_masks`);
  2. the text is painted out of the picture (inpaint) and the picture is regraded (`VARIANTS`);
  3. the rembg models matte the character on that (`RECIPE`), and the parts are joined;
  4. OpenCV matches the bottom frame on this card, including its one, two or three rune
     stones. The detected frame and black credits band are no foil; surrounding art is foil.

Output: 8-bit grey, white = no foil (text, border, bar, character), black = foil.

Usage: elestrals_mask.py frame <card.png> [--out DIR]   steps 1 and 4 only, with overlays; no model, fast
       elestrals_mask.py sweep <card.png> [--out DIR]   every variant x every model, one sheet per model
       elestrals_mask.py build <card.png> [--out DIR]   the mask, with RECIPE
       elestrals_mask.py fixed <card.png>               export the detected layout to FIXED

sweep and build run the segmentation models on the CPU; birefnet-general takes about 10 GB of RAM.
The models are downloaded to ~/.u2net on first use.
"""
import argparse
import gc
import os
import json

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


def detected_frame(rgb):
    """Match original metal surfaces at normalized size, ignoring printed symbols.

    Separate silhouettes cover classic and stat bars, each with 1/2/3 stones.
    Searching a small neighborhood permits shifts without binding the mask to
    the input resolution. A weak match fails explicitly instead of applying an
    unrelated layout. No learned model or additional dependency is required.
    """
    directory = os.path.join(ROOT, 'frames', 'detector')
    with open(os.path.join(directory, 'templates.json'), encoding='utf-8') as stream:
        templates = json.load(stream)
    canonical = cv2.resize(rgb, (1468, 2048), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(canonical, cv2.COLOR_RGB2GRAY)
    search = cv2.copyMakeBorder(gray[1470:2020], 0, 0, 40, 40, cv2.BORDER_REPLICATE)
    search = cv2.resize(search, None, fx=.5, fy=.5)
    results = []
    for item in templates:
        name = item['name']
        ref = cv2.imread(os.path.join(directory, name + '-reference.png'), cv2.IMREAD_GRAYSCALE)
        weight = cv2.imread(os.path.join(directory, name + '-match.png'), cv2.IMREAD_GRAYSCALE)
        gem = cv2.imread(os.path.join(directory, name + '-gem.png'), cv2.IMREAD_GRAYSCALE)
        if ref is None or weight is None or gem is None:
            raise ValueError('Missing frame detector assets: ' + name)
        ref = cv2.resize(ref, None, fx=.5, fy=.5)
        weight = cv2.resize(weight, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        gem = cv2.resize(gem, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Normalized intensity matching retains surface shading while excluding
        # the variable numbers and elemental glyphs through the weight mask.
        scores = cv2.matchTemplate(search, ref, cv2.TM_CCORR_NORMED, mask=weight)
        def edges(picture):
            smooth = cv2.GaussianBlur(picture, (5, 5), 0)
            return (np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0)) +
                    np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1)))
        gem_scores = cv2.matchTemplate(edges(search), edges(ref), cv2.TM_CCORR_NORMED, mask=gem)
        if name in ('classic-2', 'classic-3'):
            scores = .8 * scores + .2 * gem_scores
        scores[~np.isfinite(gem_scores) | (gem_scores < .50)] = -1
        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        _, score, _, location = cv2.minMaxLoc(scores)
        results.append((score, item, location))
    score, item, location = max(results, key=lambda result: result[0])
    if score < .88:
        raise ValueError(f'No supported bottom frame matched reliably (score {score:.3f}).')
    dx, dy = location[0] * 2 - 40, location[1] * 2 - 40
    mask = cv2.imread(os.path.join(directory, item['name'] + '-mask.png'), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError('Missing frame silhouette: ' + item['name'])
    correction_name = None
    contour_entry = None
    correction_directory = os.path.join(ROOT, 'corrections')
    correction_index = os.path.join(correction_directory, 'frames.json')
    if os.path.isfile(correction_index):
        with open(correction_index, encoding='utf-8') as stream:
            frame_entries = json.load(stream)
        for entry in frame_entries:
            with Image.open(os.path.join(correction_directory, entry['reference'])) as reference:
                sample = np.array(Image.fromarray(rgb).resize(reference.size)).astype(float)
                difference = np.abs(sample - np.array(reference.convert('RGB')).astype(float)).mean()
            if difference <= entry['max_mean_absolute_error']:
                with Image.open(os.path.join(correction_directory, entry['mask'])) as contour:
                    mask = np.array(contour.convert('L').resize((1468, 2048), Image.Resampling.NEAREST))
                correction_name = entry['name']
                contour_entry = entry
                break
    mask = cv2.warpAffine(mask, np.float32([[1, 0, dx], [0, 1, dy]]), (1468, 2048), flags=cv2.INTER_NEAREST)
    mask = refine_frame_boundary(canonical, mask > 127, item['name'], dx, dy,
                                 margin=2 if correction_name else 32)
    if contour_entry is not None:
        # Explicitly measured exterior bevel edges prevent similarly colored
        # artwork from returning along straight sides and rounded corners.
        for side in ('left', 'right'):
            profile = contour_entry.get(side + '_edge_profile')
            if not profile:
                continue
            points = np.asarray(profile, dtype=float)
            for row in range(round(points[0, 0]) + dy, round(points[-1, 0]) + dy + 1):
                if not 0 <= row < mask.shape[0]:
                    continue
                edge = int(round(np.interp(row - dy, points[:, 0], points[:, 1]))) + dx
                if side == 'left':
                    mask[row, :max(0, edge)] = False
                else:
                    mask[row, min(mask.shape[1], edge + 1):] = False
        bottom = contour_entry.get('side_bottom')
        if bottom is not None:
            mask[bottom + dy + 1:, :265 + dx] = False
            mask[bottom + dy + 1:, 1200 + dx:] = False
        # Remove the colored artwork fringe left by mixed edge pixels. This
        # operation is confined to the measured side edges of this correction.
        for row in range(max(0, 1700 + dy), min(mask.shape[0], 1963 + dy)):
            columns = np.flatnonzero(mask[row])
            if not len(columns):
                continue
            for edge, step in ((int(columns[0]), 1), (int(columns[-1]), -1)):
                if not (edge < 200 + dx or edge > 1250 + dx):
                    continue
                for _ in range(18):
                    if not 0 <= edge < mask.shape[1] or not mask[row, edge]:
                        break
                    red, green, blue = canonical[row, edge].astype(int)
                    colored_art = max(red, green, blue) - min(red, green, blue) > 24
                    colored_art |= green - red > 12 or red - blue > 10
                    if not colored_art:
                        break
                    mask[row, edge] = False
                    edge += step
    mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    print(f"frame {item['name']}  runes {item['runes']}  match {score:.3f}  offset {dx},{dy}", flush=True)
    return mask, {'style': item['name'], 'runes': item['runes'], 'score': score,
                  'offset_at_1468x2048': [dx, dy], 'boundary_method': 'GrabCut with template seeds',
                  'contour_correction': correction_name,
                  'candidate_scores': {candidate['name']: value for value, candidate, _ in results}}


def refine_frame_boundary(rgb, template, style, dx, dy, margin=32):
    """Use the matched shape as seeds, then follow this card's actual perimeter.

    Only a bounded band around the template is allowed to change. Interior
    numbers, black stat circles and rune symbols remain part of the frame.
    """
    yy, xx = np.nonzero(template)
    y0, y1 = max(0, yy.min() - 40), min(template.shape[0], yy.max() + 41)
    x0, x1 = max(0, xx.min() - 40), min(template.shape[1], xx.max() + 41)
    seed = template[y0:y1, x0:x1].astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    inside = cv2.erode(seed, kernel) > 0
    if style.startswith('stats-'):
        # The dark interiors of the stat circles can resemble dark artwork.
        # Seed them explicitly so GrabCut cannot cut a notch through a number.
        circles = np.zeros(seed.shape, np.uint8)
        for cx in (160, 1306):
            cv2.circle(circles, (cx + dx - x0, 1710 + dy - y0), 85, 1, -1)
        inside |= (circles > 0) & (seed > 0)
    outside = cv2.dilate(seed, kernel) == 0
    labels = np.full(seed.shape, cv2.GC_PR_BGD, np.uint8)
    labels[seed > 0] = cv2.GC_PR_FGD
    labels[inside] = cv2.GC_FGD
    labels[outside] = cv2.GC_BGD
    cv2.setRNGSeed(0)
    cv2.grabCut(cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR), labels, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                4, cv2.GC_INIT_WITH_MASK)
    refined = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, disc(2))
    refined = ndi.binary_fill_holes(refined)
    components, _ = ndi.label(refined)
    keep = np.unique(components[inside])
    refined &= np.isin(components, keep[keep > 0])
    # The lower metal lip is similar in color to dark artwork beneath it.
    # Do not let segmentation grow a large strip of that artwork into the frame.
    limit = cv2.dilate(seed, disc(min(3, margin) if margin <= 6 else 6)) > 0
    lower = max(0, round(rgb.shape[0] * .88) - y0)
    refined[lower:] &= limit[lower:]
    result = np.zeros_like(template)
    result[y0:y1, x0:x1] = refined
    return result


def detected_layout(rgb, bar):
    # The frame mask controls only the metal itself. The footer's black band
    # remains no-foil, and exposed art immediately below the frame remains foil.
    h, w = bar.shape
    layout = np.full((h, w), 128, np.uint8)
    start = stat_bar_top(bar)
    footer = footer_mask(rgb)
    footer |= (rgb.max(2) < 20) & card_perimeter((h, w))
    footer = footer[start:]
    layout[start:] = np.where(footer, 255, 0)
    layout[bar] = 255
    return layout


def stat_bar_top(bar):
    rows = np.flatnonzero(bar.any(axis=1))
    if not len(rows):
        raise ValueError('Cannot establish the artwork boundary: empty stat frame.')
    return int(rows[0])


def footer_mask(rgb):
    """Trace the printed footer from below, without flooding into the artwork.

    Enclosed artist/credits lettering is filled before the scan. Each column
    stops at its first artwork pixel, so the foil strip follows the real footer
    curve rather than a rectangle. Isolated black artwork above it is excluded.
    """
    h, w = rgb.shape[:2]
    start = round(h * .84)
    black = rgb.max(2) < 8
    filled = ndi.binary_fill_holes(black[start:])
    run = np.cumprod(filled[::-1].astype(np.uint8), axis=0).sum(axis=0)
    top = h - run
    window = max(3, round(31 * w / 1468) | 1)
    top = ndi.median_filter(top, size=window, mode='nearest')
    top = np.maximum(top, round(h * .90))
    footer = np.arange(h)[:, None] >= top[None, :]
    footer |= black & card_perimeter((h, w))
    footer[:start] = False
    return footer


def card_perimeter(shape):
    """Physical outer margin only; do not flood from it into dark artwork."""
    h, w = shape
    margin = max(1, round(w * .033))
    edge = np.zeros((h, w), bool)
    edge[:margin] = edge[-margin:] = True
    edge[:, :margin] = edge[:, -margin:] = True
    return edge


def disc(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def grow(mask, r):
    return cv2.dilate(mask.astype(np.uint8), disc(r)).astype(bool)


def fixed_mask(rgb):
    """Compatibility export of the layout detected on this particular card."""
    bar, _ = detected_frame(rgb)
    return detected_layout(rgb, bar)


def title_mask(rgb):
    """Bright, outlined card names in the full-art title band, including their outline.

    Detect individual outlined glyphs rather than protecting a solid rectangle:
    the artwork between letters and inside their counters stays available for foil.
    Coordinates and sizes follow the original 1468 x 2048 full-art layout.
    """
    h, w = rgb.shape[:2]
    scale = w / 1468
    r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
    band = np.zeros((h, w), bool)
    band[round(h * .045):round(h * .125), round(w * .10):round(w * .90)] = True
    bright = band & ((.299 * r + .587 * g + .114 * b) > 120) & (rgb.max(2) > 170)
    # Dark red strokes are not neutral black and were missed by frame_masks.
    outline = ((.299 * r + .587 * g + .114 * b) < 130) | ((r > g * 1.35) & (r > b * 1.25) & (g < 140))
    fill = np.zeros((h, w), bool)
    candidates = [bright, bright & ((.299 * r + .587 * g + .114 * b) > 150),
                  band & (r > 190) & (g > 170) & (r - b > 25) & (g - b > 15),
                  band & ndi.binary_fill_holes(cv2.morphologyEx(outline.astype(np.uint8), cv2.MORPH_CLOSE, disc(2))) & ~outline]
    accepted = []
    for candidate in candidates:
        lab, n = ndi.label(candidate)
        radius = max(1, round(5 * scale))
        for label, bounds in enumerate(ndi.find_objects(lab), 1):
            if bounds is None:
                continue
            ys, xs = bounds
            height, width = ys.stop - ys.start, xs.stop - xs.start
            if not (35 * scale <= height <= 130 * scale and 5 * scale <= width <= 150 * scale
                    and width <= height * 2.8 and .09 * h <= ys.stop <= .12 * h
                    and ys.start >= .058 * h
                    and .078 * h <= (ys.start + ys.stop) / 2 <= .102 * h):
                continue
            ys = slice(max(0, ys.start - radius), min(h, ys.stop + radius))
            xs = slice(max(0, xs.start - radius), min(w, xs.stop + radius))
            glyph = lab[ys, xs] == label
            rim = grow(glyph, radius) & ~glyph
            if glyph.sum() < 65 * scale ** 2 or outline[ys, xs][rim].mean() < .45:
                continue
            accepted.append((xs.start, xs.stop, ys.start, ys.stop, lab, label))
    # A title is a row of neighboring glyphs. Discard isolated stars/props in
    # the title band rather than extending the protection to nearby artwork.
    accepted.sort(key=lambda item: item[0])
    groups = []
    for item in accepted:
        if not groups or item[0] - max(x[1] for x in groups[-1]) > 28 * scale:
            groups.append([])
        groups[-1].append(item)
    if groups:
        group = max(groups, key=lambda items: max(x[1] for x in items) - min(x[0] for x in items))
        for x0, x1, y0, y1, lab, label in group:
            fill[y0:y1, x0:x1] |= lab[y0:y1, x0:x1] == label
    near = grow(fill, max(1, round(5 * scale)))
    return fill | (near & outline)


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
    title = title_mask(rgb)
    text |= title

    # border: black that reaches the edge of the picture
    lab, _ = ndi.label(mx < 20)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = np.isin(lab, edge[edge > 0]) & card_perimeter(rgb.shape[:2])
    return text, border, grow(fill, 9) | grow(title, 2)


def painted_out(rgb, area):
    """The card without its text, for the models."""
    return cv2.inpaint(rgb, area.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)


def subject_correction(rgb):
    """Explicit artwork-specific subject selections, guarded by image matching.

    This is a saved user correction, not automatic recognition of foil motifs.
    A filename alone never triggers it on a different illustration.
    """
    directory = os.path.join(ROOT, 'corrections')
    index = os.path.join(directory, 'subjects.json')
    if not os.path.isfile(index):
        return None, None
    with open(index, encoding='utf-8') as stream:
        entries = json.load(stream)
    for entry in entries:
        with Image.open(os.path.join(directory, entry['reference'])) as reference:
            target = np.array(reference.convert('RGB')).astype(float)
            sample = np.array(Image.fromarray(rgb).resize(reference.size)).astype(float)
        error = np.abs(sample - target).mean()
        if error <= entry['max_mean_absolute_error']:
            with Image.open(os.path.join(directory, entry['mask'])) as image:
                mask = np.array(image.convert('L').resize((rgb.shape[1], rgb.shape[0]),
                                                        Image.Resampling.NEAREST)) > 127
            print(f"subject correction: {entry['name']} (artwork difference {error:.2f})", flush=True)
            return mask, entry['name']
    return None, None


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


def model_masks(pictures, models, art_height):
    """{(picture name, model): matte}. One model in memory at a time: together they need
    more RAM than most machines have."""
    from rembg import new_session, remove
    out = {}
    for m in models:
        session = new_session(m, providers=["CPUExecutionProvider"])
        for name, pic in pictures.items():
            if not 0 < art_height <= pic.shape[0]:
                raise ValueError('Invalid artwork boundary for character model.')
            # Neither the stat bar nor any pixels below it enter the model.
            crop = Image.fromarray(pic[:art_height])
            upper = np.array(remove(crop, session=session, only_mask=True).convert("L"))
            matte = np.zeros(pic.shape[:2], np.uint8)
            matte[:art_height] = upper
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
    bar, detection = detected_frame(rgb)
    art_height = stat_bar_top(bar)
    detection['artwork_height'] = art_height
    print(f'Character model boundary: row {art_height}; stat bar and everything below are excluded.', flush=True)
    layout = detected_layout(rgb, bar)
    Image.fromarray(footer_mask(rgb).astype(np.uint8) * 255).save(os.path.join(args.out, f'{name}_footer_mask.png'))
    Image.fromarray(bar.astype(np.uint8) * 255).save(os.path.join(args.out, f'{name}_stat_frame_mask.png'))
    rgba = np.dstack([rgb, bar.astype(np.uint8) * 255])
    rgba[~bar, :3] = 0
    Image.fromarray(rgba).save(os.path.join(args.out, f'{name}_stat_frame.png'))
    with open(os.path.join(args.out, f'{name}_frame_detection.json'), 'w', encoding='utf-8') as stream:
        json.dump(detection, stream, indent=2)
    correction, correction_name = subject_correction(rgb)
    text, border, letters = frame_masks(rgb)
    if correction_name in ('Trifernal center dog', 'Emphyrix complete tail and wings', 'Stratomoth wings and body'):
        # Speed lines and round foil glints can fool the text detector.
        # Restrict these artwork-specific corrections to printed text bands.
        h, w = text.shape
        print_band = np.zeros((h, w), bool)
        print_band[round(h * .045):round(h * .125)] = True
        rules_start = .605 if correction_name == 'Trifernal center dog' else .585
        print_band[round(h * rules_start):round(h * .75)] = True
        text &= print_band
        letters &= print_band
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
        masks = model_masks(pics, args.models, art_height)
        np.savez_compressed(os.path.join(args.out, f"{name}_sweep.npz"),
                            **{f"{v}|{m}": k for (v, m), k in masks.items()})
        for m in args.models:
            tiles = [(v, overlay(rgb, final(np.maximum(masks[v, m], fixed * 255)))) for v in args.variants]
            sheet(tiles, 6, os.path.join(args.out, f"{name}_sweep_{m}.jpg"))
        return

    if correction is None:
        pics = {v: variant(v, rgb, clean, frame) for v in {v for v, _, _ in RECIPE}}
        masks = model_masks(pics, sorted({m for _, m, _ in RECIPE}), art_height)
        char = np.zeros(fixed.shape, bool)
        for v, m, t in RECIPE:
            char |= masks[v, m] >= t
    else:
        char = correction.copy()
    # the model takes the bar for foreground and its matte runs a few px over the bar's edge onto
    # the art (a 15 px wedge beside the gem on Aeolus). Of the matte off the fixed mask, keep only
    # the pieces that reach further than 16 px from it: a boot does, the spill does not.
    lab, n = ndi.label(char & (layout == 128))
    far = np.unique(lab[~grow(layout == 255, 16)])
    char &= np.isin(lab, far[far > 0])
    lab, n = ndi.label(~char)                                     # pinholes between rope strands
    char |= np.isin(lab, np.flatnonzero(ndi.sum(~char, lab, np.arange(1, n + 1)) < 0.001 * char.size) + 1)
    if correction is not None:
        # Preserve the explicit selection; global pinhole filling must not add
        # background pixels back into the corrected subject.
        char = correction.copy()
        char[art_height:] = False
        Image.fromarray(char.astype(np.uint8) * 255).save(os.path.join(args.out, f'{name}_subject_mask.png'))
    char[art_height:] = False
    mask = final((char | fixed) * 255)
    Image.fromarray(mask).save(os.path.join(args.out, f"{name}.png"))
    overlay(rgb, mask).save(os.path.join(args.out, f"{name}_overlay.jpg"))
    print(f"no foil {(mask > 0).mean():.1%}")


if __name__ == "__main__":
    main()
