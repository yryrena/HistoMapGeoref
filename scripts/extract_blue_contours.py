#!/usr/bin/env python
"""
Title: Extract Blue Contours from Georeferenced Maps
Date: 2025-09-27
Author: Ruiyi Yang 

Description: This script extracts "blue" linework (e.g., isohyets, hydrological features) 
from a georeferenced RGB GeoTIFF and outputs vector polylines as GeoJSON.

Features:
- HSV-based thresholding (with CLI overrides for fine-tuning color detection)
- Morphological cleanup (opening, closing, small-object removal)
- Optional Gaussian blur for noise reduction
- Contour tracing → LineStrings in EPSG:4326 (lon/lat coordinates)
- Adjustable vertex thinning and geometric simplification
- Debug outputs: RGB preview and binary mask (PNG)
 
Notes: 
- skimage HSV convention: H, S, V ∈ [0, 1].
- If no contours are detected, try widening the hue range and lowering S/V thresholds.
- Adjust morphology and blur parameters to match the scan quality.
"""


import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.transform import xy

from shapely.geometry import LineString, mapping
from shapely.ops import unary_union

from skimage.color import rgb2hsv
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk


def imwrite_png(path, arr_uint8):
    """Write a grayscale or RGB uint8 numpy array to PNG (no extra deps)."""
    try:
        from imageio.v2 import imwrite
        imwrite(path, arr_uint8)
    except Exception:
        ## fallback via Pillow
        from PIL import Image
        Image.fromarray(arr_uint8).save(path)


def normalize_rgb(r, g, b):
    """Normalize bands to [0,1] float."""
    def _norm(x):
        x = x.astype(np.float32)
        mx = x.max()
        if mx == 0:
            return x
        ## if 8-bit or 16-bit, map to [0,1]
        if mx > 1.0:
            return x / (255.0 if mx <= 255 else mx)
        return x
    return _norm(r), _norm(g), _norm(b)


def threshold_blue_hsv(rgb, h_min, h_max, s_min, v_min, blur_sigma=0.0):
    """
    Threshold 'blue' in HSV.
    Parameters are in [0,1] (skimage convention).
    """
    if blur_sigma and blur_sigma > 0:
        ## light blur to stabilize thresholds
        rgb = gaussian(rgb, sigma=blur_sigma, multichannel=True, preserve_range=True)

    hsv = rgb2hsv(np.clip(rgb, 0, 1))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    ## handle wrap-around if h_min > h_max (rare, for red across 0/1 boundary)
    if h_min <= h_max:
        mask = (H >= h_min) & (H <= h_max) & (S >= s_min) & (V >= v_min)
    else:
        mask = ((H >= h_min) | (H <= h_max)) & (S >= s_min) & (V >= v_min)

    return mask.astype(np.uint8)


def trace_contours(mask_uint8, level=0.5):
    """find_contours expects float/uint; returns list of Nx2 arrays in (row, col)."""
    return find_contours(mask_uint8.astype(np.uint8), level=level)


def to_lines(contours, ds, step=1, simplify_tol=0.0):
    """
    Convert pixel-space contours to lon/lat LineStrings.
    - step: keep every Nth vertex to thin lines (>=1).
    - simplify_tol: simplify in degrees (0 disables).
    """
    features = []
    for arr in contours:
        if arr.shape[0] < 2:
            continue
        arr = arr[::max(1, int(step))]  ## thin vertices
        rows = arr[:, 0]
        cols = arr[:, 1]
        xs, ys = xy(ds.transform, rows, cols, offset='center')  ## lon, lat for EPSG:4326
        line = LineString(zip(xs, ys))
        if simplify_tol and simplify_tol > 0:
            line = line.simplify(simplify_tol, preserve_topology=False)
        if line.is_valid and not line.is_empty and len(line.coords) >= 2:
            features.append(line)
    return features


def save_geojson(lines, out_path):
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(ln)} for ln in lines]
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    print(f"Wrote {len(lines)} polylines → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract 'blue' contours to GeoJSON from a georeferenced RGB GeoTIFF.")
    ap.add_argument("geotiff", help="Input georeferenced RGB GeoTIFF (EPSG:4326).")
    ap.add_argument("out_geojson", help="Output GeoJSON path.")

    ## HSV thresholds (0..1)
    ap.add_argument("--h-min", type=float, default=0.55, help="Hue min for blue (default 0.55)")
    ap.add_argument("--h-max", type=float, default=0.75, help="Hue max for blue (default 0.75)")
    ap.add_argument("--s-min", type=float, default=0.25, help="Saturation min (default 0.25)")
    ap.add_argument("--v-min", type=float, default=0.15, help="Value (brightness) min (default 0.15)")

    ## pre/post processing
    ap.add_argument("--min-size", type=int, default=50, help="Remove connected components smaller than this (pixels).")
    ap.add_argument("--open", type=int, default=0, dest="open_r", help="Binary opening radius (0=off).")
    ap.add_argument("--close", type=int, default=0, dest="close_r", help="Binary closing radius (0=off).")
    ap.add_argument("--blur", type=float, default=0.0, help="Gaussian blur sigma before HSV (0=off).")

    ## vectorization tuning
    ap.add_argument("--step", type=int, default=1, help="Keep every Nth vertex (>=1).")
    ap.add_argument("--simplify", type=float, default=0.0, help="Shapely simplify tolerance in degrees (0=off).")

    ## debug outputs
    ap.add_argument("--debug-dir", default=None, help="Write debug PNGs (RGB preview + mask) into this folder.")

    args = ap.parse_args()

    if not os.path.exists(args.geotiff):
        sys.exit(f"Input GeoTIFF not found: {args.geotiff}")

    ## read raster
    with rasterio.open(args.geotiff) as ds:
        if ds.count < 3:
            sys.exit("Need an RGB raster (>=3 bands).")
        r = ds.read(1)
        g = ds.read(2)
        b = ds.read(3)
        r, g, b = normalize_rgb(r, g, b)
        rgb = np.dstack([r, g, b])

    ## write downscaled RGB preview if requested
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        ## quick downscale for viewing if huge
        H, W, _ = rgb.shape
        max_side = 2000
        scale = min(1.0, max_side / max(H, W))
        if scale < 1.0:
            newH, newW = int(H * scale), int(W * scale)
            ## simple area-based downscale
            from skimage.transform import resize
            rgb_prev = resize(rgb, (newH, newW), order=1, preserve_range=True, anti_aliasing=True)
        else:
            rgb_prev = rgb
        rgb8 = np.clip(rgb_prev * 255.0, 0, 255).astype(np.uint8)
        imwrite_png(os.path.join(args.debug_dir, "debug_rgb_preview.png"), rgb8)

    ## threshold in HSV
    mask = threshold_blue_hsv(
        rgb,
        h_min=args.h_min, h_max=args.h_max,
        s_min=args.s_min, v_min=args.v_min,
        blur_sigma=args.blur
    )

    ## morphological cleanup
    if args.min_size and args.min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=args.min_size).astype(np.uint8)
    if args.open_r and args.open_r > 0:
        mask = binary_opening(mask.astype(bool), selem=disk(args.open_r)).astype(np.uint8)
    if args.close_r and args.close_r > 0:
        mask = binary_closing(mask.astype(bool), selem=disk(args.close_r)).astype(np.uint8)

    ## save mask
    if args.debug_dir:
        mask8 = (mask * 255).astype(np.uint8)
        imwrite_png(os.path.join(args.debug_dir, "debug_blue_mask.png"), mask8)

    ## trace contours
    contours = trace_contours(mask, level=0.5)
    if not contours:
        print("No contours found. Try widening hue range and/or lowering s_min/v_min, e.g.:")
        print("  --h-min 0.50 --h-max 0.80 --s-min 0.15 --v-min 0.10")
        print("Also try: --min-size 10 --open 1 --close 1 --blur 0.5")
        save_geojson([], args.out_geojson)
        return

    ## convert to lines (lon/lat)
    with rasterio.open(args.geotiff) as ds:
        lines = to_lines(contours, ds, step=max(1, args.step), simplify_tol=max(0.0, args.simplify))

    ## optionally merge tiny segments (disabled by default; uncomment if needed)
    # merged = unary_union(lines)
    # if merged.geom_type == "LineString":
    #     lines = [merged]
    # elif merged.geom_type == "MultiLineString":
    #     lines = list(merged.geoms)

    save_geojson(lines, args.out_geojson)


if __name__ == "__main__":
    main()