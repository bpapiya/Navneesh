# First page: mobile intro animation

Open `index.html` on a phone (or any browser). It runs about 34 seconds and then repeats (times below are from the moment screen one appears, about 2.8s in):

1. **Gate** (0–3.6s): the view walks through the flowered gate towards the book, with a gentle step. The arch passes out of frame, and the blurry book in the distance sharpens as it gets closer. When it reaches screen one's book, the scene cuts straight in.
2. **Scene** (from the cut): sky, town, tree, pot and the closed book are already in place.
3. **Book** (0.8–1.5s): the closed green book rises into place.
4. **Characters** (1.3–2.3s): the woman slides in from the left and the man from the right.
5. **Logo** (2.3–3.0s): the NP monogram grows from its own centre with a soft gold halo, then a light sweep.
6. **Into page 2** (3.7–6.1s): the two characters leave, the front cover swings open around the spine, the open book fades in, and the seated man slides in from the left.
7. **Man's screen** (4.25–12.5s): his open book fades in as the cover opens. He slides in, slides out, the page turns, he slides back in; then he slides out again and the page turns once more, and as it settles his screen hands over to the seated couple's (the same book, drawn separately).
8. **Seated couple's screen** (12.4–18.2s): they slide in, slide out while the page turns, slide back in; then slide out again while the page turns once more.
9. **Standing couple's screen** (19.55–24.6s): their open book fades in as the previous page turn settles; they slide in, stay a moment and slide out, then the same again. No page turn on this screen.
10. **Book closes** (24.6–26.8s): the reverse of the opening: the green cover swings back in around the spine and the closed-book scene fades in underneath. Then the woman slides in from the left and the man from the right.
11. **Credit** (26.9–28.0s): "Created by" and "Joy & Paps" fade in, in gold, on the closed book's cover.
12. **Repeat** (30.4–31.2s): the gate fades back in over the closed book and the walk starts again.

With reduced motion turned on, the closed-book page with the couple on either side shows straight away and nothing loops.

## Files

- `assets/gate.webp`, `gate-nobook.webp`: the opening gate from `source_gate.png`, and a copy with its book painted out (`build_gate.py`).
- `assets/bg.webp`, `book.webp`, `logo.webp`, `pot.webp`, `woman.webp`, `man.webp`: page 1 layers cut from `source.png`.
- `assets/bg2c.webp`, `bg2cext.webp`, `man2.webp`: the seated man's screen from `source_page2b.png`, with him cut out and the book behind him rebuilt (`build_page2b.py`).
- `assets/bg3.webp`, `bg3ext.webp`, `couple.webp`: the seated couple's screen from `source_page3.png`, with them cut out and the book behind them rebuilt (`build_page3.py`).
- `assets/bg4.webp`, `bg4ext.webp`, `couple2.webp`: the standing couple's screen from `source_page4.png`, with the couple cut out and the book behind them rebuilt.
- `assets/bg5.webp`, `man5.webp`, `woman5.webp`, `bg5ext.webp`: the closed-book scene from `source_page5.png`.
- `assets/bgext.webp`, `bg2ext.webp`: copies with the tree and pot painted out, used for the mirrored sky and floor strips on tall phones.
- `segment.py`, `build_layers.py`, `build_page2.py`, `build_page3.py`, `build_page4.py`, `build_page5.py`, `make_ext.py`: the scripts that produced the layers (`pip install rembg onnxruntime opencv-python-headless pillow numpy`).
