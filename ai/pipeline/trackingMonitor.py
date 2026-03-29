import os
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np


class PipelineTracker:
    def __init__(self, outputRoot="output"):
        self.framesDir = os.path.join(outputRoot, "frames")
        self.corneaDir = os.path.join(outputRoot, "segmentedCornea")
        self.barDir = os.path.join(outputRoot, "segmentedBar")
        self.intersectDir = os.path.join(outputRoot, "intersection")
        self.spanDir = os.path.join(outputRoot, "horizontalSpans")

        for dirPath in [self.framesDir, self.corneaDir, self.barDir, self.intersectDir, self.spanDir]:
            os.makedirs(dirPath, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("Tracking Monitor")
        self.root.geometry("1200x800")
        self.buildMainMenu()
        self.root.mainloop()


    def sortedFiles(self, folder, prefix, ext=".json"):
        files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(ext)]
        return sorted(files, key=lambda f: int(f.replace(prefix, "").replace(ext, "")))

    def frameFiles(self):
        files = [f for f in os.listdir(self.framesDir) if f.startswith("img_")]
        return sorted(files, key=lambda f: int(f.replace("img_", "").replace(".jpg", "")))

    def loadFrame(self, frameFile):
        return Image.open(os.path.join(self.framesDir, frameFile)).convert("RGB")

    def loadMask(self, jsonPath):
        with open(jsonPath) as f:
            return json.load(f)["coordinates"]

    def overlayMask(self, img, coords, color=(0, 255, 0, 100)):
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for r, c in coords:
            draw.point((c, r), fill=color)
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def toTk(self, img, maxW=1100, maxH=650):
        img.thumbnail((maxW, maxH))
        return ImageTk.PhotoImage(img)

    def clearWindow(self):
        for w in self.root.winfo_children():
            w.destroy()

    def buildMainMenu(self):
        self.clearWindow()
        self.root.title("Pipeline Tracker – Main Menu")
        tk.Label(self.root, text="Pipeline Tracker", font=("Arial", 20, "bold")).pack(pady=30)

        modes = [
            ("Cornea Segmentation", lambda: self.buildMaskViewer("cornea")),
            ("Bar Segmentation", lambda: self.buildMaskViewer("bar")),
            ("Intersection Segmentation", lambda: self.buildMaskViewer("intersection")),
            ("Horizontal Spans", self.buildSpanViewer),
        ]
        for label, cmd in modes:
            tk.Button(self.root, text=label, width=30, height=2, command=cmd).pack(pady=8)
