import pytest
import requests
import os

vgg16Url = "http://localhost:10000"

samplesDir = os.path.join(os.path.dirname(__file__), "Tests", "Samples")
lrVideoPath = os.path.join(samplesDir, "lr.MOV")
rlVideoPath = os.path.join(samplesDir, "rl.MOV")


class TestUnitVGG16:
    def testVGG16Health(self):
        response = requests.get(f"{vgg16Url}/docs")
        assert response.status_code == 200

    def testVGG16DocsAvailable(self):
        response = requests.get(f"{vgg16Url}/openapi.json")
        assert response.status_code == 200

    def testVGG16MapsEndpointExists(self):
        response = requests.get(f"{vgg16Url}/openapi.json")
        paths = response.json().get("paths", {})
        assert "/api/Samples/maps" in paths


class TestComponentVGG16:
    def testVGG16ProcessesVideos(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{vgg16Url}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        assert response.status_code == 200

    def testVGG16ReturnsMapKeys(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{vgg16Url}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        data = response.json()
        assert "maps" in data
        assert "trackingVideos" in data

    def testVGG16RejectsNoFiles(self):
        response = requests.post(f"{vgg16Url}/api/Samples/maps")
        assert response.status_code == 422


class TestIntegrationVGG16:
    def testVGG16ServiceReachable(self):
        response = requests.get(f"{vgg16Url}/docs")
        assert response.status_code == 200

    def testVGG16FullPipeline(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{vgg16Url}/api/Samples/maps",
                files={
                    "left2right": ("lr.MOV", lr, "video/quicktime"),
                    "right2left": ("rl.MOV", rl, "video/quicktime")
                },
                timeout=600
            )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestApiVGG16:
    def testVGG16RejectsSingleVideo(self):
        with open(lrVideoPath, "rb") as lr:
            response = requests.post(
                f"{vgg16Url}/api/Samples/maps",
                files={"left2right": ("lr.MOV", lr, "video/quicktime")}
            )
        assert response.status_code == 422

    def testVGG16MapsResponseStructure(self):
        with open(lrVideoPath, "rb") as lr, open(rlVideoPath, "rb") as rl:
            response = requests.post(
                f"{vgg16Url}/api/Samples/maps",
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

    def testVGG16DocsSchema(self):
        response = requests.get(f"{vgg16Url}/openapi.json")
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
