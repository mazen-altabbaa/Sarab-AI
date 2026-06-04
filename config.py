import os
from pathlib import Path

rootDir = Path(__file__).parent
dataPath = rootDir / "data"
aiModelsPath = rootDir / "ai" / "models"
backendPath = rootDir / "backend"
frontendPath = rootDir / "frontend"
testsPath = rootDir / "tests"

colors = {
    'primary': '#3B82F6'
}

# AI configs
maxBatchSize = 32
imageWidth = 224
imageHeight = 224
timeoutSeconds = 30
confidenceThreshold = 0.75
maxRetries = 3

defaultModelName = ""


# Frontend configs
frontendPort = None
defaultTheme = ""

# Backend configs
backendPort = 8000
backendHost = "0.0.0.0"
corsOrigins = ["http://localhost:3000", "http://localhost:8000"]
