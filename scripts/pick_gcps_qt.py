#!/usr/bin/env python
"""
Title: Interactive GCP Picker (Qt + Matplotlib)
Date: 2025-09-27
Author: Ruiyi Yang 

Description: This script provides an interactive tool to place Ground Control Points (GCPs) 
on scanned historical maps with visible graticules. It uses PyQt for dialogs 
and Matplotlib for visualization, allowing full create/edit/delete workflow.

Features:
- Click map pixels → enter corresponding lon/lat (decimal or DMS).
- Undo last point (U), delete nearest (D), edit nearest (E).
- Save (S) or Save-As (A) to JSON with pixel + lon/lat coordinates.
- Auto-loads existing `gcps.json` if present in working directory.
- Prevents accidental duplicate clicks near existing GCPs.
- Provides numbered labels on all placed GCPs for easy reference.
 
Controls: 
- Left Click → Add GCP at clicked pixel, prompt for lon/lat
- U → Undo last point
- D → Delete nearest point to mouse cursor
- E → Edit nearest point (re-enter lon/lat)
- S → Save to JSON (default: gcps.json)
- A → Save-As JSON with new name
- Q → Quit (with confirmation)
"""


import json, re, os, sys, math, pathlib

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

DEFAULT_JSON = "gcps.json"

def parse_angle(s):
    if isinstance(s, (int, float)): return float(s)
    text = str(s).strip().upper().replace(" ", "")
    sign = 1.0
    if text.endswith(("W","S")): sign, text = -1.0, text[:-1]
    elif text.endswith(("E","N")): text = text[:-1]
    text = (text.replace("DEG","°").replace("D","°")
                 .replace("MIN","′").replace("M","′")
                 .replace("SEC","″").replace("S","″")
                 .replace("'", "′").replace('"', "″"))
    try: return sign * float(text)
    except ValueError: pass
    d=m=sec=0.0
    mobj = re.search(r"(-?\d+(?:\.\d+)?)°", text)
    if mobj: d = float(mobj.group(1))
    else:
        parts = re.split(r"[,:]", text)
        if len(parts)==2 and parts[0] and parts[1]:
            return sign*(float(parts[0]) + float(parts[1])/60.0)
    mobj = re.search(r"(\d+(?:\.\d+)?)′", text);  m = float(mobj.group(1)) if mobj else 0.0
    sobj = re.search(r"(\d+(?:\.\d+)?)″", text);  sec = float(sobj.group(1)) if sobj else 0.0
    return sign*(d + m/60.0 + sec/3600.0)

def ask_text(title, prompt, preset="", parent=None):
    ok = False
    while not ok:
        text, ok = QInputDialog.getText(parent, title, prompt, text=preset)
        if not ok: return None
        text = text.strip()
        if text: return text

def ask_lon_lat(parent=None, preset=None):
    plon = "" if not preset else str(preset[0])
    plat = "" if not preset else str(preset[1])
    while True:
        lon_s = ask_text("Longitude", "Enter longitude (decimal or DMS, e.g. 113.333 or 113°20′E):", plon, parent)
        if lon_s is None: return None
        try: lon = parse_angle(lon_s); break
        except Exception: QMessageBox.critical(parent, "Invalid", "Please enter a valid longitude.")
    while True:
        lat_s = ask_text("Latitude", "Enter latitude (decimal or DMS, e.g. 34.25 or 34°15′N):", plat, parent)
        if lat_s is None: return None
        try: lat = parse_angle(lat_s); break
        except Exception: QMessageBox.critical(parent, "Invalid", "Please enter a valid latitude.")
    return lon, lat

def choose_image(parent=None, initial=None):
    dlg = QFileDialog(parent, "Select map image", initial or os.path.expanduser("~"))
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilters(["Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)", "All files (*)"])
    return dlg.selectedFiles()[0] if dlg.exec_() and dlg.selectedFiles() else ""

def save_as_dialog(parent=None, initial_name=DEFAULT_JSON):
    dlg = QFileDialog(parent, "Save GCPs as…", str(pathlib.Path(initial_name).absolute()))
    dlg.setAcceptMode(QFileDialog.AcceptSave)
    dlg.setNameFilters(["JSON (*.json)", "All files (*)"])
    if dlg.exec_() and dlg.selectedFiles():
        path = dlg.selectedFiles()[0]
        if not path.lower().endswith(".json"): path += ".json"
        return path
    return None

def nearest_index(gcps, x, y):
    if not gcps: return None
    best_i, best_d = None, float("inf")
    for i,g in enumerate(gcps):
        px,py = g["pixel"]
        d = (px-x)**2 + (py-y)**2
        if d < best_d: best_d, best_i = d, i
    return best_i

def main():
    app = QApplication.instance() or QApplication(sys.argv)

    ## image path
    img_path = sys.argv[1] if len(sys.argv)>1 else choose_image()
    if not img_path or not os.path.exists(img_path):
        print("No image selected. Exiting."); return
    img = mpimg.imread(img_path)

    ## state
    gcps, markers, labels = [], [], []
    out_json = DEFAULT_JSON

    ## auto-load existing gcps.json if present
    if os.path.exists(out_json):
        try:
            loaded = json.loads(open(out_json, "r", encoding="utf-8").read())
            for d in loaded:
                if "pixel" in d and "lonlat" in d:
                    gcps.append({"pixel": [float(d["pixel"][0]), float(d["pixel"][1])],
                                 "lonlat": [float(d["lonlat"][0]), float(d["lonlat"][1])]})
            print(f"Loaded {len(gcps)} points from {out_json}")
        except Exception as e:
            print("Could not load existing gcps.json:", e)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title("Click → enter lon/lat | Keys: U=undo, D=delete-nearest, E=edit-nearest, S=save, A=save-as, Q=quit")
    ax.set_axis_off()

    def redraw():
        ## clear all markers & labels, then redraw from gcps
        for m in markers: m.remove()
        markers.clear()
        for t in labels: t.remove()
        labels.clear()
        for i,g in enumerate(gcps, 1):
            col,row = g["pixel"]
            markers.append(ax.scatter([col],[row], s=30, c="red"))
            labels.append(ax.text(col,row,str(i), color="yellow", fontsize=10,
                                  ha="left", va="bottom",
                                  bbox=dict(facecolor="black", alpha=0.3, pad=1)))
        fig.canvas.draw_idle()

    ## initial draw if we loaded points
    redraw()

    def onclick(ev):
        if ev.inaxes != ax or ev.xdata is None or ev.ydata is None or ev.button != 1:
            return
        col, row = float(ev.xdata), float(ev.ydata)

        ## prevent accidental duplicate at same pixel
        if any(abs(col-g["pixel"][0])<0.5 and abs(row-g["pixel"][1])<0.5 for g in gcps):
            print("Skipped: looks like a duplicate click at nearly the same pixel.")
            return

        ans = ask_lon_lat()
        if ans is None: return
        lon, lat = ans

        gcps.append({"pixel": [col, row], "lonlat": [lon, lat]})
        print(f"Added GCP #{len(gcps)}  pixel=({col:.1f},{row:.1f})  lon/lat=({lon:.6f},{lat:.6f})")
        redraw()

    def onkey(ev):
        nonlocal out_json
        if ev.key in ("u","U"):
            if gcps:
                gcps.pop()
                print(f"Undo → {len(gcps)} points remain.")
                redraw()
        elif ev.key in ("d","D"):
            if not gcps: return
            ## find nearest to current mouse position in data coords
            if ev.xdata is None or ev.ydata is None:
                QMessageBox.information(None, "Delete", "Move the mouse near the point you want to delete, then press D.")
                return
            i = nearest_index(gcps, ev.xdata, ev.ydata)
            if i is not None:
                removed = gcps.pop(i)
                print(f"Deleted point #{i+1} at pixel=({removed['pixel'][0]:.1f},{removed['pixel'][1]:.1f})")
                redraw()
        elif ev.key in ("e","E"):
            if not gcps: return
            if ev.xdata is None or ev.ydata is None:
                QMessageBox.information(None, "Edit", "Move the mouse near the point you want to edit, then press E.")
                return
            i = nearest_index(gcps, ev.xdata, ev.ydata)
            if i is not None:
                lon0, lat0 = gcps[i]["lonlat"]
                ans = ask_lon_lat(preset=(lon0, lat0))
                if ans is not None:
                    gcps[i]["lonlat"] = [ans[0], ans[1]]
                    print(f"Edited point #{i+1} → lon/lat=({ans[0]:.6f},{ans[1]:.6f})")
                    redraw()
        elif ev.key in ("s","S"):
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(gcps, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(gcps)} points to {out_json}")
            QMessageBox.information(None, "Saved", f"Saved {len(gcps)} points to {out_json}")
        elif ev.key in ("a","A"):
            path = save_as_dialog(initial_name=out_json)
            if path:
                out_json = path
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(gcps, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(gcps)} points to {out_json}")
                QMessageBox.information(None, "Saved", f"Saved {len(gcps)} points to {out_json}")
        elif ev.key in ("q","Q"):
            if QMessageBox.question(None, "Quit", "Quit without saving? (Press S first to save)") == QMessageBox.Yes:
                plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()

if __name__ == "__main__":
    main()
