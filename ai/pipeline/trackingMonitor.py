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