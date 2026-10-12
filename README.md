# Elestrals segmentation workflow

Makes the no-foil mask of an Elestrals full-art card picture: text, border, bottom bar and
character stay flat, the rest of the art is foil.

Output is an 8-bit grey PNG the size of the card: **white = no foil, black = foil**.
`examples/Aeolus_mask.png` is the result for Aeolus (BS1-097).

Status: made and checked on **one card** (Aeolus, 1468 x 2048), on overlays only. Not yet run on
a second card and not yet seen in game.

## Install

    pip install -r requirements.txt

The matte model (`birefnet-general`, about 970 MB) is downloaded to `~/.u2net` on first use. It
runs on the CPU and takes about 10 GB of RAM.

## Use

    python3 elestrals_mask.py build <card.png>     the mask: out/<card>.png and out/<card>_overlay.jpg
    python3 elestrals_mask.py frame <card.png>     text, border and bar only; no model, fast
    python3 elestrals_mask.py sweep <card.png>     every regrade x every model, one contact sheet per model
    python3 elestrals_mask.py fixed <card.png>     rebuild frames/ElestralsFullArt.png from the bar trace

`--out DIR` changes the output folder. On the overlay the foil area is tinted magenta.

Card pictures are not in this repo; bring your own.

## How it works

Matte models fail on a whole card: the rules text reads as foreground. So the card is taken apart
first.

1. **Text and black border by colour rules, no model.** Text is any white patch with a black rim.
   The outline and its dark glow count as text; what shows through a letter (the middle of an
   "o" in the title) is art and stays foil. The border is black that reaches the picture's edge.
2. **The text is painted out** (inpaint), so the model sees a card without its text.
3. **`birefnet-general` mattes the character** on that picture.
4. **The bottom of the card is a fixed mask.** It is the same on every card and never has a
   character on it, so `frames/ElestralsFullArt.png` is the last word there: bar, gem, inset line
   and the black band with the credits are no foil, the slivers of art around the bar are foil.

Two clean-ups on the character matte: pieces that stay within 16 px of the fixed mask are dropped
(the model takes the bar for foreground and spills a few px over its edge onto the art), and
pinholes inside the character are filled.

## The fixed mask

`frames/ElestralsFullArt.png` has three values:

| Value | Meaning |
|---|---|
| 255 white | no foil |
| 0 black | foil |
| 128 grey | not fixed, decided per card (everything above the bar) |

It is built by `fixed` from `frames/ElestralsBarTrace.png`, a hand trace of the bar and its gem
(white on black), plus the black of the card it is run on. To change the bar's outline, edit the
trace and run `fixed` again. Soft brush edges are snapped to the nearest of the three values.

## What was tried

14 regrades of the card x 7 rembg models (`sweep`), judged on overlay sheets:

- **`birefnet-general` on the text-free card** has the whole figure, including the legs behind the
  rules text.
- **Gamma, contrast, saturation, CLAHE** move its matte by 1 to 2%. None is used.
- **A median blur** makes every model worse.
- **Blanking the frame** before the model does not help.
- **`birefnet-general-lite` and `birefnet-massive`** lose the legs under the text.
- **`isnet-anime`** loses half the white cape; **`birefnet-portrait`** finds nothing.
- **Colour rules for the bar** could not part its grey bevel and gem from grey clouds behind them,
  and SAM on click points came out speckled. Hence the hand trace.

## Limits

- The colour thresholds and pixel distances are tuned to a 1468 x 2048 picture of this layout.
- Whether the frame sits on the same pixels on other cards is not checked. If it does not, the
  fixed mask needs an offset or scale.
- Other layouts (a different bar, no bar) need their own fixed mask.
- One card per run; there is no batch mode.
- Props count as character (the sack on Aeolus).
