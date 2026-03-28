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

def createCircularMap(dataBlock, cmap=None, norm=None, figsize=(10, 10)):
    data = np.array(dataBlock)
    nRings, nMeridians = data.shape

    maskedData = np.ma.masked_where(data <= -900, data)
    
    theta = np.linspace(0, 2*np.pi, nMeridians, endpoint=False)
    r = np.linspace(0, 1, nRings)
    
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1, projection='polar')
    
    mesh = ax.pcolormesh(Theta, R, maskedData, cmap=cmap, norm=norm, shading='auto')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    for i in range(1, min(nRings, 6)):
        circleRadius = i / nRings
        circle = plt.Circle((0, 0), circleRadius, transform=ax.transData._b, 
                           fill=False, color='gray', alpha=0.5, linewidth=0.5)
        ax.add_patch(circle)
    
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    plt.tight_layout()
    return fig

def createCompositeMap(blocks, cmap=None, norm=None, figsize=(15, 5)):
    nBlocks = len(blocks)
    fig, axes = plt.subplots(1, nBlocks, figsize=figsize, 
                             subplot_kw={'projection': 'polar'})
    
    if nBlocks == 1:
        axes = [axes]
    
    for idx, (ax, block) in enumerate(zip(axes, blocks)):
        data = np.array(block)
        nRings, nMeridians = data.shape
        
        maskedData = np.ma.masked_where(data <= -900, data)
        
        theta = np.linspace(0, 2*np.pi, nMeridians, endpoint=False)
        r = np.linspace(0, 1, nRings)
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        
        mesh = ax.pcolormesh(Theta, R, maskedData, cmap=cmap, norm=norm, shading='auto')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
    
    plt.tight_layout()
    return fig
