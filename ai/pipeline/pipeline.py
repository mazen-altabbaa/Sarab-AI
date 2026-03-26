import os
import cv2
import json
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VideoPipeline:
    def __init__(self, videoPath, outputDir="output"):
        self.videoPath = videoPath
        self.framesDir = os.path.join(outputDir, "frames")
        self.corneaDir = os.path.join(outputDir, "segmentedCornea")
        self.barDir = os.path.join(outputDir, "segmentedBar")
        self.intersectionDir = os.path.join(outputDir, "intersection")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for d in [self.framesDir, self.corneaDir, self.barDir, self.intersectionDir]:
            os.makedirs(d, exist_ok=True)

    def extractFrames(self):
        cap = cv2.VideoCapture(self.videoPath)
        count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(self.framesDir, f"img_{count}.jpg"), frame)
            count += 1
        cap.release()
        
        print(f"extracted {count-1} frames")
