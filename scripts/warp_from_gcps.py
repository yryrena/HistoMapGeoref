#!/usr/bin/env python
"""
Title: Warp Raster from GCPs (Thin Plate Spline, EPSG:4326)
Date: 2025-09-28
Author: Ruiyi Yang 

Description: This script takes a scanned historical map and a set of Ground Control Points (GCPs) 
(JSON format from `pick_gcps_qt.py`), applies them with GDAL, and warps the raster 
into geographic coordinates (WGS84, EPSG:4326).

Workflow: 
- Reads pixel ↔ lon/lat GCP pairs from a JSON file.
- Creates a VRT (Virtual Raster) with embedded GCPs.
- Warps the raster with Thin Plate Spline (TPS) transformation.
- Saves output as a GeoTIFF aligned to EPSG:4326.

Requirements: GDAL with Python bindings (`conda install -c conda-forge gdal`) 
"""


import json, sys, os
from osgeo import gdal, osr

def main():
    if len(sys.argv) < 3:
        print("Usage: python warp_from_gcps.py <image> <gcps.json> [out_geotiff]")
        sys.exit(1)
    in_img = sys.argv[1]
    in_gcps = sys.argv[2]
    out_tif = sys.argv[3] if len(sys.argv) > 3 else "map_georef.tif"

    ## read GCPs
    gcps_json = json.loads(open(in_gcps, "r", encoding="utf-8").read())
    if len(gcps_json) < 4:
        raise SystemExit("Need at least 4 GCPs (more is better).")

    ## build GDAL GCP list (lon/lat are in degrees → WGS84)
    gcp_list = []
    for g in gcps_json:
        col, row = g["pixel"]   ## x = column, y = row (pixels)
        lon, lat = g["lonlat"]  ## in degrees
        gcp_list.append(gdal.GCP(float(lon), float(lat), 0.0, float(col), float(row)))

    ## write a VRT with the GCPs
    vrt_path = os.path.splitext(out_tif)[0] + ".vrt"
    src_ds = gdal.Open(in_img, gdal.GA_ReadOnly)
    if src_ds is None:
        raise SystemExit(f"cannot open image: {in_img}")

    ## older GDAL does not support GCPSpatialRef argument 
    gdal.Translate(vrt_path, src_ds, GCPs=gcp_list, format="VRT")

    ## warp (TPS) to a georeferenced GeoTIFF in EPSG:4326
    gdal.Warp(
        out_tif, vrt_path,
        dstSRS="EPSG:4326",   ## force WGS84 output
        tps=True,
        multithread=True,
        resampleAlg="cubicspline",
        format="GTiff"
    )

    print(f"Done → {out_tif}")

if __name__ == "__main__":
    main()
