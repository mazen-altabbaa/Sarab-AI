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

    def segmentCornea(self, modelPath, threshold=0.5, imgSize=512):
        model = SegformerForSemanticSegmentation.from_pretrained(modelPath)
        model.to(self.device).eval()

        transform = A.Compose([
            A.Resize(imgSize, imgSize),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        for fname in sorted(os.listdir(self.framesDir)):
            imgBGR = cv2.imread(os.path.join(self.framesDir, fname))
            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            h, w = imgBGR.shape[:2]

            tensor = transform(image=imgRGB)["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(pixel_values=tensor).logits
                if logits.shape[1] > 1:
                    logits = logits[:, 1:2]
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

            mask = (prob > threshold).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            count = os.path.splitext(fname)[0].split("_")[1]
            self.saveMaskJson(mask, self.corneaDir, f"segmentedCornea_{count}.json")


    def computeIntersections(self):
        corneaFiles = sorted(os.listdir(self.corneaDir))

        for cf in corneaFiles:
            count = cf.replace("segmentedCornea_", "").replace(".json", "")
            bf = f"segmentedBar_{count}.json"

            corneaPath = os.path.join(self.corneaDir, cf)
            barPath    = os.path.join(self.barDir, bf)

            if not os.path.exists(barPath):
                print(f"there is no matching bar file for frame {count}!!!!")
                continue

            corneaMask = self.loadMaskJson(corneaPath)
            barMask    = self.loadMaskJson(barPath)

            intersection = np.logical_and(corneaMask, barMask).astype(np.uint8)
            self.saveMaskJson(intersection, self.intersectionDir, f"intersection_{count}.json")
