import importlib.metadata
import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

asrModel = ASRModelFactory.create_asr_model()
asrModel.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
)

assetsPath = os.getcwd() + "/swagger-ui-assets"
if path.exists(assetsPath + "/swagger-ui.css") and path.exists(assetsPath + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assetsPath), name="static")

    def swaggerMonkeyPatch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swaggerMonkeyPatch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/api/Samples/asr", tags=["Endpoints"])
async def getSamplesAsr(
    audioFile: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initialPrompt: Union[str, None] = Query(default=None),
    vadFilter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    wordTimestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    minSpeakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    maxSpeakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    result = asrModel.transcribe(
        load_audio(audioFile.file, encode),
        task,
        language,
        initialPrompt,
        vadFilter,
        wordTimestamps,
        {"diarize": diarize, "min_speakers": minSpeakers, "max_speakers": maxSpeakers},
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audioFile.filename)}.{output}"',
        },
    )


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audioFile: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initialPrompt: Union[str, None] = Query(default=None),
    vadFilter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    wordTimestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    minSpeakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    maxSpeakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    result = asrModel.transcribe(
        load_audio(audioFile.file, encode),
        task,
        language,
        initialPrompt,
        vadFilter,
        wordTimestamps,
        {"diarize": diarize, "min_speakers": minSpeakers, "max_speakers": maxSpeakers},
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audioFile.filename)}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detectLanguage(
    audioFile: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detectedLangCode, confidence = asrModel.language_detection(load_audio(audioFile.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detectedLangCode],
        "language_code": detectedLangCode,
        "confidence": confidence,
    }


@click.command()
@click.option("-h", "--host", metavar="HOST", default="0.0.0.0")
@click.option("-p", "--port", metavar="PORT", default=9000)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()