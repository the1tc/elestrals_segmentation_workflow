# Elestrals segmentation workflow

Creates a no-foil mask for full-art Elestrals card images. The output is an 8-bit
PNG: **white = no foil; black = foil**. The overlay tints foil magenta.

## Install and run

```sh
pip install -r requirements.txt
python elestrals_mask.py build card.png
python launcher.py
```

`launcher.py` adds an optional Tkinter desktop interface with image/output-folder
pickers, the four workflows, a live log, and a busy indicator. Processing runs in
a separate worker process. A Python installation with Tkinter is required for
this source launcher; the Windows executable is not committed to this repository.
Keep the processing script, `frames/`, and `corrections/` alongside the launcher.

```sh
python elestrals_mask.py build card.png --out out
python elestrals_mask.py frame card.png --out out
python elestrals_mask.py sweep card.png --out out
python elestrals_mask.py fixed card.png
python -m unittest discover -s tests -v
```

- `build`: final mask and overlay; uses the original `birefnet-general` recipe
  unless the artwork matches an explicit saved subject correction.
- `frame`: text/frame preview and the text-painted-out image, without a model.
- `sweep`: diagnostic comparisons of the selected models and variants.
- `fixed`: compatibility export of this card's detected layout to
  `frames/ElestralsFullArt.png`. Normal runs detect their own layout and no longer
  depend on that shared output file.

The model is downloaded to `~/.u2net` on first use (approximately 970 MB), runs on
the CPU, and originally required approximately 10 GB RAM. Sweep can download
additional models. No new runtime dependencies were added for detection.

## Processing and the stat-bar boundary

1. Detect neutral printed body text by its white fill and dark rim. A separate
   title detector handles the colored, outlined card names in the full-art title
   band, preserving the spaces between letters.
2. Match the bottom frame using OpenCV and reference surfaces. Separate templates
   cover classic and circular-stat frames with one, two, or three rune stones.
   Variable numbers and elemental symbols are excluded from matching weights.
3. Refine the frame perimeter locally with GrabCut. Lower-edge growth is limited
   so dark artwork beneath the inset stripe does not become part of the metal.
4. Paint out text for the character model. **The model receives only the crop
   above the highest detected stat-bar pixel.** Its matte is padded with zero
   below that row. This applies to Build and Sweep, including models other than
   BirefNet. Saved subject selections and subsequent hole filling cannot override
   the lower layout either.
5. From that row down, only the frame, printed footer, and physical card perimeter
   determine the mask. The footer is traced upward from the bottom black band,
   with enclosed credits filled and isolated dark artwork excluded. Exposed art
   between the frame and footer remains foil.

The physical perimeter is bounded spatially: a dark patch connected to the card
edge is not automatically treated as border throughout the artwork.

## Additional output files

Each normal workflow also writes:

| File | Content |
|---|---|
| `<card>_stat_frame.png` | Original frame pixels on transparency; hidden RGB cleared |
| `<card>_stat_frame_mask.png` | White frame, black elsewhere |
| `<card>_footer_mask.png` | Detected printed footer protection |
| `<card>_frame_detection.json` | Style, rune count, similarity, offset, artwork cutoff and contour correction |
| `<card>_subject_mask.png` | Saved subject selection, when a corrected Build is used |

## Explicit artwork corrections

`corrections/subjects.json` includes saved selections for Trifernal (middle dog,
shoulders and front legs; rear dogs treated as background), Emphyrix (tail and
pale wing sections), and Stratomoth (body, antennae and wing boundaries).
`corrections/frames.json` contains Ohmperial's tighter frame contour and measured
exterior side profiles. Corrections require a match to a small RGB artwork
reference; filenames alone do not activate them.

These are user-directed artwork selections, **not a model trained to recognize
foil stars or E symbols**. Subject corrections apply to Build; Sweep remains a
comparison of model results. Text protection on corrected artwork is constrained
to its title/rules bands to avoid treating glints and speed lines as letters.

## Validation and limits

Frame style/rune-count selection was checked on 15 available Base Full Art
images, including mixed-element three-rune stat frames. Resized images, small
translations, blank-image rejection, and representative frame interiors were
checked during development. Several contours and subjects were refined after
visual review; that does not establish pixel-perfect accuracy on unseen cards.
The original `examples/Aeolus_mask.png` predates these changes.

Classic two/three-rune references are composed from the original classic stone.
They passed composed-image checks, but real cards of those two layouts were not
available. Other layouts can fail the similarity threshold or require new
references. Similarity scores are not calibrated probabilities.

Frame matching normalizes to 1468 x 2048 and searches a small neighborhood for
translation. The original body-text thresholds, morphology distances and subject
cleanup remain tuned to that size. Corrections rescale with the image but are not
intended for different artwork crops or card layouts. Inspect frame/subject
exports and overlays before using a newly encountered card in game.

The model input crop intentionally changes the model's context compared with the
original whole-card run. Segmentation above the stat bar may therefore differ.

The checked-in reference crops/thumbnails come from the inspected original card
images; masks are traced/refined detection assets. No generated illustration is
used as a frame-matching reference. The original hand-traced frame assets remain
available in `frames/`.
