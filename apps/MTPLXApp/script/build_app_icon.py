#!/usr/bin/env python3
"""Build the MTPLX macOS app icon (.icns) from a source PNG/JPG.

Steps:
  1. Detect the squircle's bounding box by scanning for the first
     non-black row / column in from each edge (the source has solid
     black padding around the squircle).
  2. Crop to that bounding box and square it on a tight black canvas
     so the squircle is perfectly centred regardless of any asymmetry
     in the source padding (AI-generated source art is rarely
     pixel-centred).
  3. Flood-fill the rounded corners with transparency so the icon
     doesn't render with black square corners against a white Finder
     window or against the Dock background.
  4. Scale the squircle down and centre it on a 1024×1024 transparent
     canvas using Apple's standard macOS-icon template (≈824×824
     content area, ≈100px transparent padding on each side). This is
     non-negotiable — without it the icon renders ~25% larger than
     every other Dock app and looks visually broken.
  5. Emit every macOS app-icon size into a `.iconset/` folder and run
     `iconutil` to produce `AppIcon.icns`.

Run once whenever the source art changes. Idempotent — overwrites the
existing icon set + .icns each run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


# All sizes Apple expects in a macOS iconset; keys must match the names
# `iconutil` looks for inside the .iconset folder.
ICONSET_SIZES = {
    "icon_16x16.png":      16,
    "icon_16x16@2x.png":   32,
    "icon_32x32.png":      32,
    "icon_32x32@2x.png":   64,
    "icon_128x128.png":    128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png":    256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png":    512,
    "icon_512x512@2x.png": 1024,
}

# "Pure black" threshold for the source's outer padding. The source's
# outside-the-squircle pixels measure under (8,8,8); the squircle itself
# starts at the subtle chrome highlight which is far above 30 sum.
NEAR_BLACK_SUM = 18

# Apple's macOS app-icon template (Big Sur onward). The squircle
# content area is 824×824 inside a 1024×1024 canvas, leaving 100px of
# transparent padding on each side. Every native macOS app icon —
# Mail, Safari, Music, Finder, Photos — follows this. An icon that
# fills the full 1024×1024 looks ~25% oversized next to them in the
# Dock and Launchpad, which is the bug we hit on the first pass.
ICON_CANVAS_SIZE = 1024
ICON_CONTENT_SIZE = 824
ICON_PADDING = (ICON_CANVAS_SIZE - ICON_CONTENT_SIZE) // 2


def find_squircle_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Scan from each edge inward until we hit a non-black pixel.

    The source image has solid-black padding around the squircle. Edge
    detection therefore reduces to "first row / column that contains any
    pixel brighter than `NEAR_BLACK_SUM`".
    """
    rgb = img.convert("RGB")
    px = rgb.load()
    W, H = rgb.size

    def row_has_content(y: int) -> bool:
        for x in range(W):
            r, g, b = px[x, y]
            if (r + g + b) > NEAR_BLACK_SUM:
                return True
        return False

    def col_has_content(x: int) -> bool:
        for y in range(H):
            r, g, b = px[x, y]
            if (r + g + b) > NEAR_BLACK_SUM:
                return True
        return False

    top = next(y for y in range(H) if row_has_content(y))
    bottom = next(y for y in range(H - 1, -1, -1) if row_has_content(y))
    left = next(x for x in range(W) if col_has_content(x))
    right = next(x for x in range(W - 1, -1, -1) if col_has_content(x))
    return (left, top, right + 1, bottom + 1)


def crop_to_square(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop to the bbox, then pad to a perfect square so x/y aspect is 1:1.

    Any extra rows/columns added to square the canvas stay pure black so
    `punch_transparent_corners` can flood-fill straight through them.
    """
    cropped = img.crop(bbox)
    w, h = cropped.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    canvas.paste(cropped.convert("RGBA"), ((side - w) // 2, (side - h) // 2))
    return canvas


def fit_to_apple_template(squircle: Image.Image) -> Image.Image:
    """Centre the cropped squircle on Apple's 1024×1024 macOS app-icon
    template with 100px transparent padding on every side.

    Without this step a tight-cropped icon renders ~25% larger than
    every native Dock app, which reads as "wrong size / not a macOS
    icon" no matter how clean the underlying artwork is.
    """
    resized = squircle.resize(
        (ICON_CONTENT_SIZE, ICON_CONTENT_SIZE),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (ICON_CANVAS_SIZE, ICON_CANVAS_SIZE), (0, 0, 0, 0))
    canvas.paste(resized, (ICON_PADDING, ICON_PADDING), resized)
    return canvas


def punch_transparent_corners(img: Image.Image) -> Image.Image:
    """Flood-fill from the four corners with transparency.

    The squircle's rounded corners leave small black triangles in each
    corner of the cropped bbox. Flood-filling from the corner inward
    with a generous tolerance turns those triangles transparent without
    touching the squircle's chrome highlight (which is far brighter).
    """
    rgba = img.convert("RGBA")
    transparent = (0, 0, 0, 0)
    for corner in ((0, 0), (rgba.width - 1, 0), (0, rgba.height - 1), (rgba.width - 1, rgba.height - 1)):
        # Tolerance 24 catches the slight per-pixel jitter inside the
        # outer black padding without bleeding past the squircle edge,
        # which jumps to ~RGB 40+ for the chrome highlight.
        ImageDraw.floodfill(rgba, corner, transparent, thresh=24)
    return rgba


def render_iconset(master: Image.Image, iconset_dir: Path) -> None:
    iconset_dir.mkdir(parents=True, exist_ok=True)
    # Master is already 1024×1024 (Apple template). For every smaller
    # target, scale directly from the 1024 master in one Lanczos pass —
    # cascading downscales blur fine detail at 16/32px sizes where the
    # chrome edges already barely have room to render.
    assert master.size == (ICON_CANVAS_SIZE, ICON_CANVAS_SIZE)
    for name, size in ICONSET_SIZES.items():
        out = master.resize((size, size), Image.Resampling.LANCZOS)
        out.save(iconset_dir / name, "PNG")


def build_icns(iconset_dir: Path, icns_path: Path) -> None:
    subprocess.run(
        ["iconutil", "--convert", "icns", "--output", str(icns_path), str(iconset_dir)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the source PNG/JPG of the icon (with black outer padding).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write AppIcon.icns + the .iconset folder.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iconset_dir = args.output_dir / "AppIcon.iconset"
    if iconset_dir.exists():
        shutil.rmtree(iconset_dir)
    icns_path = args.output_dir / "AppIcon.icns"
    master_path = args.output_dir / "AppIcon-master.png"

    src = Image.open(args.source)
    bbox = find_squircle_bbox(src)
    print(f"detected squircle bbox: {bbox}  ({bbox[2] - bbox[0]}x{bbox[3] - bbox[1]})")

    squared = crop_to_square(src, bbox)
    transparent = punch_transparent_corners(squared)
    master = fit_to_apple_template(transparent)
    master.save(master_path, "PNG")
    print(
        f"wrote master: {master_path}  ({master.size[0]}x{master.size[1]}, "
        f"{ICON_CONTENT_SIZE}x{ICON_CONTENT_SIZE} content + {ICON_PADDING}px padding)"
    )

    render_iconset(master, iconset_dir)
    build_icns(iconset_dir, icns_path)
    print(f"wrote icns: {icns_path}")


if __name__ == "__main__":
    main()
