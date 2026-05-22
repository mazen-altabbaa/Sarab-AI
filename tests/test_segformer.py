import pytest
import requests
import os

segformerUrl = "http://localhost:10000"

samplesDir = os.path.join(os.path.dirname(__file__), "Tests", "Samples")
lrVideoPath = os.path.join(samplesDir, "lr.MOV")
rlVideoPath = os.path.join(samplesDir, "rl.MOV")


class TestUnitSegformer:
    def testSegformerHealth(self):
        response = requests.get(f"{segformerUrl}/docs")
        assert response.status_code == 200

    def testSegformerDocsAvailable(self):
        response = requests.get(f"{segformerUrl}/openapi.json")
        assert response.status_code == 200

    def testSegformerMapsEndpointExists(self):
        response = requests.get(f"{segformerUrl}/openapi.json")
        paths = response.json().get("paths", {})
        assert "/api/Samples/maps" in paths


class TestComponentSegformer:
    def testSegformerProcessesVideos(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{segformerUrl}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        assert response.status_code == 200

    def testSegformerReturnsMapKeys(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{segformerUrl}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        data = response.json()
        assert "maps" in data
        assert "trackingVideos" in data

    def testSegformerRejectsNoFiles(self):
        response = requests.post(f"{segformerUrl}/api/Samples/maps")
        assert response.status_code == 422


class TestIntegrationSegformer:
    def testSegformerServiceReachable(self):
        response = requests.get(f"{segformerUrl}/docs")
        assert response.status_code == 200

    def testSegformerFullPipeline(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{segformerUrl}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestApiSegformer:
    def testSegformerRejectsSingleVideo(self):
        with open(lrVideoPath, "rb") as lr:
            response = requests.post(
                f"{segformerUrl}/api/Samples/maps",
                files={"left2right": ("lr.MOV", lr, "video/quicktime")}
            )
        assert response.status_code == 422

    def testSegformerMapsResponseStructure(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{segformerUrl}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        data = response.json()
        assert "maps" in data
        assert "left2right" in data["maps"]
        assert "right2left" in data["maps"]
        assert "fullMap" in data["maps"]
        assert "trackingVideos" in data
        assert "left2right" in data["trackingVideos"]
        assert "right2left" in data["trackingVideos"]

    def testSegformerDocsSchema(self):
        response = requests.get(f"{segformerUrl}/openapi.json")
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
