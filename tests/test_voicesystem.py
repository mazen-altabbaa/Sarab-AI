import pytest
import requests
import os

whisperUrl = "http://localhost:9000"
llmUrl = "http://localhost:8000"

samplesDir = os.path.join(os.path.dirname(__file__), "Tests", "Samples")
audioPath = os.path.join(samplesDir, "sample.wav")


class TestUnitWhisper:
    def testWhisperHealth(self):
        response = requests.get(f"{whisperUrl}/docs")
        assert response.status_code == 200

    def testWhisperDocsTitle(self):
        response = requests.get(f"{whisperUrl}/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert "title" in data["info"]

    def testWhisperRedirectToDoc(self):
        response = requests.get(f"{whisperUrl}/", allow_redirects=False)
        assert response.status_code in [301, 302, 307, 308]

    def testWhisperAsrEndpointExists(self):
        response = requests.get(f"{whisperUrl}/openapi.json")
        paths = response.json().get("paths", {})
        assert "/api/Samples/asr" in paths


class TestUnitLlm:
    def testLlmHealth(self):
        response = requests.get(f"{llmUrl}/health")
        assert response.status_code == 200
        assert response.json().get("status") == "ok"

    def testLlmDocsAvailable(self):
        response = requests.get(f"{llmUrl}/docs")
        assert response.status_code == 200

    def testLlmAnalyzeEndpointExists(self):
        response = requests.get(f"{llmUrl}/openapi.json")
        paths = response.json().get("paths", {})
        assert any("/api/Samples/" in p for p in paths)

    def testLlmAnalyzeGetMethod(self):
        response = requests.get(f"{llmUrl}/api/Samples/test123/analyze")
        assert response.status_code == 200
        data = response.json()
        assert "sample_id" in data


class TestComponentWhisper:
    def testWhisperTranscribesAudio(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{whisperUrl}/api/Samples/asr",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"language": "ar", "output": "txt"}
            )
        assert response.status_code == 200
        assert len(response.text.strip()) > 0

    def testWhisperReturnsTextOutput(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{whisperUrl}/api/Samples/asr",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"output": "txt"}
            )
        assert "text/plain" in response.headers.get("content-type", "")

    def testWhisperRejectsNoFile(self):
        response = requests.post(f"{whisperUrl}/api/Samples/asr")
        assert response.status_code == 422


class TestComponentLlm:
    def testLlmAnalyzesAudio(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{llmUrl}/api/Samples/sample1/analyze",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"language": "ar"}
            )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def testLlmReturnsExpectedFields(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{llmUrl}/api/Samples/sample1/analyze",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"language": "ar"}
            )
        data = response.json()
        expectedFields = {"EyeSide", "Gender", "Age", "City", "Status", "Profession", "Notes"}
        assert any(field in data for field in expectedFields)

    def testLlmRejectsNoFile(self):
        response = requests.post(f"{llmUrl}/api/Samples/sample1/analyze")
        assert response.status_code == 422


class TestIntegration:
    def testWhisperToLlmPipeline(self):
        with open(audioPath, "rb") as f:
            whisperResponse = requests.post(
                f"{whisperUrl}/api/Samples/asr",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"language": "ar", "output": "txt"}
            )
        assert whisperResponse.status_code == 200
        assert len(whisperResponse.text.strip()) > 0

        with open(audioPath, "rb") as f:
            llmResponse = requests.post(
                f"{llmUrl}/api/Samples/sample1/analyze",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"language": "ar"}
            )
        assert llmResponse.status_code == 200
        assert isinstance(llmResponse.json(), dict)

    def testBothServicesReachable(self):
        whisperResp = requests.get(f"{whisperUrl}/docs")
        llmResp = requests.get(f"{llmUrl}/health")
        assert whisperResp.status_code == 200
        assert llmResp.status_code == 200

    def testLlmDependsOnWhisper(self):
        whisperResp = requests.get(f"{whisperUrl}/docs")
        assert whisperResp.status_code == 200

        with open(audioPath, "rb") as f:
            llmResponse = requests.post(
                f"{llmUrl}/api/Samples/sample1/analyze",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"language": "ar"}
            )
        assert llmResponse.status_code == 200


class TestApi:
    def testWhisperAsrReturnsContentDisposition(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{whisperUrl}/api/Samples/asr",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"output": "txt"}
            )
        assert "content-disposition" in response.headers

    def testWhisperAsrEngineHeader(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{whisperUrl}/api/Samples/asr",
                files={"audioFile": ("sample.wav", f, "audio/wav")},
                params={"output": "txt"}
            )
        assert "asr-engine" in response.headers

    def testLlmHealthEndpointSchema(self):
        response = requests.get(f"{llmUrl}/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def testLlmAnalyzeInvalidSampleId(self):
        with open(audioPath, "rb") as f:
            response = requests.post(
                f"{llmUrl}/api/Samples/@@invalid!!/analyze",
                files={"audioFile": ("sample.wav", f, "audio/wav")}
            )
        assert response.status_code in [200, 404, 422]
