import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import os


def createCustomColormap():
    colorMapping = [
        (170, '#f5cecf'),
        (200, '#ffa6b0'),
        (230, '#fc8b8c'),
        (260, '#f97d7b'),
        (290, '#f4605d'),
        (320, '#fb4548'),
        (350, '#fc180d'),
        (380, '#fe3300'),
        (410, '#f47504'),
        (440, '#f9d500'),
        (470, '#fff710'),
        (500, '#b1fa0c'),
        (530, '#00fe07'),
        (560, '#03be41'),
        (590, '#00a855'),
        (620, '#098c6a'),
        (650, '#0546b4'),
        (680, '#0216e6'),
        (710, '#0000fa'),
        (740, '#0000e7'),
        (770, '#0100dc'),
        (800, '#0300c6'),
        (830, '#00019f'),
        (860, '#00006c'),
        (890, '#000030'),
        (920, '#000022'),
        (950, '#000015'),
        (980, '#000008')
    ]
    
    colorMapping.sort(key=lambda x: x[0])
    
    thicknessValues = [cm[0] for cm in colorMapping]
    colors = [cm[1] for cm in colorMapping]
    
    boundaries = [thicknessValues[0] - 15]
    for i in range(len(thicknessValues) - 1):
        midpoint = (thicknessValues[i] + thicknessValues[i+1]) / 2
        boundaries.append(midpoint)
    boundaries.append(thicknessValues[-1] + 15)
    
    cmap = LinearSegmentedColormap.from_list('custom_cornea', colors, N=len(colors))
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    
    return cmap, norm, thicknessValues, colors, boundaries

def getClosestColorIndex(value, thicknessValues):
    if value <= thicknessValues[0]:
        return 0
    if value >= thicknessValues[-1]:
        return len(thicknessValues) - 1
    
    for i in range(len(thicknessValues) - 1):
        if thicknessValues[i] <= value <= thicknessValues[i+1]:
            if value - thicknessValues[i] < thicknessValues[i+1] - value:
                return i
            else:
                return i+1
    return len(thicknessValues) - 1


def loadCorneaFile(filename):
    fileExtension = os.path.splitext(filename)[1].lower()    
    with open(filename, 'r') as file:
        content = file.read()
    
    return parseCornealData(content)

def parseCornealData(fileContent):
    lines = [line.strip() for line in fileContent.strip().split('\n')]
    
    blocks = []
    currentBlock = []
    
    for lineNum, line in enumerate(lines):
        if line.startswith('CornealThickness') or line.startswith('"CornealThickness"'):
            continue
            
        if not line:
            if currentBlock:
                blocks.append(currentBlock)
                currentBlock = []
        else:
            line = line.replace('"', '').replace(',', ';')
            
            values = []
            for x in line.split(';'):
                x = x.strip()
                if x:
                    try:
                        x = x.replace(',', '.')
                        values.append(float(x))
                    except ValueError:
                        continue
            
            if values:
                currentBlock.append(values)
    
    if currentBlock:
        blocks.append(currentBlock)
    
    return blocks

