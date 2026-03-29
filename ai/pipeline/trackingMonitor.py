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

    def buildMaskViewer(self, mode):
        self.clearWindow()

        cfg = {
            "cornea": ("segmentedCornea_", self.corneaDir, (0, 255, 0, 120)),
            "bar": ("segmentedBar_", self.barDir, (255, 165, 0, 120)),
            "intersection": ("intersection_", self.intersectDir, (255, 0, 0, 120)),
        }
        prefix, folder, color = cfg[mode]

        frameFiles = self.frameFiles()
        maskFiles = self.sortedFiles(folder, prefix)
        total = min(len(frameFiles), len(maskFiles))

        if total == 0:
            tk.Label(self.root, text="No files found.", font=("Arial", 14)).pack(pady=20)
            tk.Button(self.root, text="← Back", command=self.buildMainMenu).pack()
            return

        state = {"count": 0, "streaming": False, "afterId": None}

        topBar = tk.Frame(self.root)
        topBar.pack(fill="x", padx=10, pady=5)
        tk.Button(topBar, text="← Back", command=lambda: self.stopStream(state) or self.buildMainMenu()).pack(side="left")
        tk.Label(topBar, text=f"{mode.capitalize()} Viewer", font=("Arial", 14, "bold")).pack(side="left", padx=15)

        infoLabel = tk.Label(self.root, text="", font=("Arial", 11))
        infoLabel.pack()
        imgLabel = tk.Label(self.root)
        imgLabel.pack(pady=5, expand=True)

        ctrlBar = tk.Frame(self.root)
        ctrlBar.pack(pady=5)
        tk.Button(ctrlBar, text="◀ Prev", command=lambda: self.step(state, -1, total, showFn)).pack(side="left", padx=4)
        tk.Button(ctrlBar, text="Next ▶", command=lambda: self.step(state, 1, total, showFn)).pack(side="left", padx=4)

        jumpFrame = tk.Frame(self.root)
        jumpFrame.pack()
        tk.Label(jumpFrame, text="Jump to frame:").pack(side="left")
        jumpEntry = tk.Entry(jumpFrame, width=6)
        jumpEntry.pack(side="left", padx=4)
        tk.Button(jumpFrame, text="Go", command=lambda: self.jump(state, jumpEntry, total, showFn)).pack(side="left")

        streamBtn = tk.Button(self.root, text="▶ Start Stream")
        streamBtn.pack(pady=4)
        streamBtn.config(command=lambda: self.toggleStream(state, total, showFn, streamBtn))

        def showFn():
            i = state["count"]
            frame = self.loadFrame(frameFiles[i])
            coords = self.loadMask(os.path.join(folder, maskFiles[i]))
            img = self.overlayMask(frame, coords, color)
            photo = self.toTk(img)
            imgLabel.config(image=photo)
            imgLabel.image = photo
            infoLabel.config(text=f"Frame {i + 1} / {total}   |   {maskFiles[i]}")

        showFn()

    def buildSpanViewer(self):
        self.clearWindow()

        frameFiles = self.frameFiles()
        spanFiles = self.sortedFiles(self.spanDir, "horizontalSpans_")
        total = min(len(frameFiles), len(spanFiles))

        if total == 0:
            tk.Label(self.root, text="No span files found.", font=("Arial", 14)).pack(pady=20)
            tk.Button(self.root, text="← Back", command=self.buildMainMenu).pack()
            return

        state = {"framecount": 0, "linecount": 0}

        def loadSpans(fi):
            with open(os.path.join(self.spanDir, spanFiles[fi])) as f:
                return json.load(f)["spans"]

        topBar = tk.Frame(self.root)
        topBar.pack(fill="x", padx=10, pady=5)
        tk.Button(topBar, text="← Back", command=self.buildMainMenu).pack(side="left")
        tk.Label(topBar, text="Horizontal Span Viewer", font=("Arial", 14, "bold")).pack(side="left", padx=15)

        infoLabel = tk.Label(self.root, text="", font=("Arial", 11))
        infoLabel.pack()
        imgLabel = tk.Label(self.root)
        imgLabel.pack(pady=5, expand=True)

        spanLabel = tk.Label(self.root, text="", font=("Arial", 12, "bold"), fg="blue")
        spanLabel.pack()

        ctrlBar = tk.Frame(self.root)
        ctrlBar.pack(pady=4)
        tk.Button(ctrlBar, text="◀ Prev Frame", command=lambda: stepFrame(-1)).pack(side="left", padx=4)
        tk.Button(ctrlBar, text="Next Frame ▶", command=lambda: stepFrame(1)).pack(side="left", padx=4)
        tk.Button(ctrlBar, text="◀ Prev Line", command=lambda: stepLine(-1)).pack(side="left", padx=8)
        tk.Button(ctrlBar, text="Next Line ▶", command=lambda: stepLine(1)).pack(side="left", padx=4)

        jumpRow = tk.Frame(self.root)
        jumpRow.pack(pady=3)
        tk.Label(jumpRow, text="Jump frame:").pack(side="left")
        frameEntry = tk.Entry(jumpRow, width=5)
        frameEntry.pack(side="left", padx=3)
        tk.Button(jumpRow, text="Go", command=lambda: jumpToFrame(frameEntry)).pack(side="left", padx=2)
        tk.Label(jumpRow, text="  Jump line:").pack(side="left")
        lineEntry = tk.Entry(jumpRow, width=5)
        lineEntry.pack(side="left", padx=3)
        tk.Button(jumpRow, text="Go", command=lambda: jumpToLine(lineEntry)).pack(side="left")

        def show():
            fi = state["framecount"]
            li = state["linecount"]
            spans = loadSpans(fi)
            frame = self.loadFrame(frameFiles[fi])

            spanVal = spans[li] if spans else 0
            img = self.highlightSpanLine(frame, fi, li)
            photo = self.toTk(img)
            imgLabel.config(image=photo)
            imgLabel.image = photo

            infoLabel.config(text=f"Frame {fi + 1}/{total}  |  Line entry {li + 1}/{len(spans)}  |  {spanFiles[fi]}")
            spanLabel.config(text=f"Span value (end − start) = {spanVal} px")

        def stepFrame(d):
            fi = max(0, min(total - 1, state["framecount"] + d))
            state["framecount"] = fi
            state["linecount"] = 0
            show()

        def stepLine(d):
            spans = loadSpans(state["framecount"])
            li = max(0, min(len(spans) - 1, state["linecount"] + d))
            state["linecount"] = li
            show()

        def jumpToFrame(entry):
            v = int(entry.get()) - 1
            state["framecount"] = max(0, min(total - 1, v))
            state["linecount"] = 0
            show()


        def jumpToLine(entry):
            try:
                spans = loadSpans(state["framecount"])
                v = int(entry.get()) - 1
                state["linecount"] = max(0, min(len(spans) - 1, v))
                show()
            except ValueError:
                pass

        show()

    def highlightSpanLine(self, frame, framecount, linecount):
        intersectFiles = self.sortedFiles(self.intersectDir, "intersection_")
        if framecount >= len(intersectFiles):
            return frame

        coords = self.loadMask(os.path.join(self.intersectDir, intersectFiles[framecount]))
        if not coords:
            return frame

        mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
        for r, c in coords:
            if r < frame.height and c < frame.width:
                mask[r, c] = 1

        activeRows = []
        for row in range(mask.shape[0]):
            cols = np.where(mask[row] == 1)[0]
            if len(cols) >= 2 and int(cols[-1] - cols[0]) > 0:
                activeRows.append((row, int(cols[0]), int(cols[-1])))

        img = frame.copy()
        draw = ImageDraw.Draw(img)
        if linecount < len(activeRows):
            row, start, end = activeRows[linecount]
            draw.line([(start, row), (end, row)], fill=(0, 100, 255), width=2)
        return img

    def step(self, state, d, total, showFn):
        state["count"] = max(0, min(total - 1, state["count"] + d))
        showFn()

    def jump(self, state, entry, total, showFn):
        try:
            v = int(entry.get()) - 1
            state["count"] = max(0, min(total - 1, v))
            showFn()
        except ValueError:
            pass

    def toggleStream(self, state, total, showFn, btn):
        if state["streaming"]:
            self.stopStream(state)
            btn.config(text="▶ Start Stream")
        else:
            state["streaming"] = True
            btn.config(text="⏹ Stop Stream")
            self.streamNext(state, total, showFn, btn)

    def streamNext(self, state, total, showFn, btn):
        if not state["streaming"] or state["count"] >= total - 1:
            self.stopStream(state)
            btn.config(text="▶ Start Stream")
            return
        state["count"] += 1
        showFn()
        state["afterId"] = self.root.after(100, lambda: self.streamNext(state, total, showFn, btn))

    def stopStream(self, state):
        state["streaming"] = False
        if state.get("afterId"):
            self.root.after_cancel(state["afterId"])
            state["afterId"] = None



if __name__ == "__main__":
    PipelineTracker(outputRoot="output")