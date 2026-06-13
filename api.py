# OpenAI API Spec. Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech

from contextlib import asynccontextmanager
from io import BytesIO
import os
import time

import soundfile as sf
import torchaudio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from g2pw import G2PWConverter
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from cosyvoice.utils.file_utils import load_wav
from single_inference import CustomCosyVoice, get_bopomofo_rare


class Settings(BaseSettings):
    api_key: str = Field(
        default="", description="Specifies the API key used to authenticate the user."
    )

    model_path: str = Field(
        default="MediaTek-Research/BreezyVoice",
        description="Specifies the model used for speech synthesis.",
    )
    speaker_prompt_audio_path: str = Field(
        default="./data/example.wav",
        description="Specifies the path to the prompt speech audio file of the speaker.",
    )
    speaker_prompt_text_transcription: str = Field(
        default="在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。",
        description="Specifies the transcription of the speaker prompt audio.",
    )


class SpeechRequest(BaseModel):
    model: str = ""
    input: str = Field(
        description="The content that will be synthesized into speech. You can include phonetic symbols if needed, though they should be used sparingly.",
        examples=["今天天氣真好"],
    )
    response_format: str = ""
    speed: float = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_started = time.perf_counter()
    app.state.settings = Settings()
    app.state.cosyvoice = CustomCosyVoice(app.state.settings.model_path)
    app.state.bopomofo_converter = G2PWConverter()
    app.state.prompt_speech_16k = load_wav(
        app.state.settings.speaker_prompt_audio_path, 16000
    )
    app.state.prompt_text_normalized = app.state.cosyvoice.frontend.text_normalize_new(
        app.state.settings.speaker_prompt_text_transcription, split=False
    )
    app.state.prompt_text_bopomo = get_bopomofo_rare(
        app.state.prompt_text_normalized, app.state.bopomofo_converter
    )
    prompt_cache_started = time.perf_counter()
    app.state.prompt_cache = app.state.cosyvoice.build_zero_shot_prompt_cache(
        app.state.prompt_text_bopomo, app.state.prompt_speech_16k
    )
    print(
        "BreezyVoice prompt cache complete:",
        {
            "prompt_cache_seconds": round(time.perf_counter() - prompt_cache_started, 3),
        },
    )
    if os.environ.get("BREEZYVOICE_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        warmup_started = time.perf_counter()
        warmup_text = os.environ.get("BREEZYVOICE_WARMUP_TEXT", "你好。")
        warmup_content = app.state.cosyvoice.frontend.text_normalize_new(
            warmup_text, split=False
        )
        warmup_content_bopomo = get_bopomofo_rare(
            warmup_content, app.state.bopomofo_converter
        )
        app.state.cosyvoice.inference_zero_shot_no_normalize_cached_prompt(
            warmup_content_bopomo,
            app.state.prompt_cache,
        )
        print(
            "BreezyVoice warmup complete:",
            {
                "warmup_text": warmup_text,
                "warmup_seconds": round(time.perf_counter() - warmup_started, 3),
            },
        )
    print(
        "BreezyVoice startup complete:",
        {
            "startup_seconds": round(time.perf_counter() - startup_started, 3),
            "model_path": app.state.settings.model_path,
        },
    )
    yield
    del app.state.cosyvoice
    del app.state.bopomofo_converter


app = FastAPI(lifespan=lifespan, root_path="/v1")


@app.get("/models")
async def get_models(request: Request):
    return {
        "object": "list",
        "data": [
            {
                "id": request.app.state.settings.model_path,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/audio/speech")
async def speach_endpoint(request: Request, payload: SpeechRequest):
    request_started = time.perf_counter()
    # normalization
    preprocess_started = time.perf_counter()
    content_to_synthesize = request.app.state.cosyvoice.frontend.text_normalize_new(
        payload.input, split=False
    )
    content_to_synthesize_bopomo = get_bopomofo_rare(
        content_to_synthesize, request.app.state.bopomofo_converter
    )
    preprocess_seconds = time.perf_counter() - preprocess_started
    inference_started = time.perf_counter()
    output = request.app.state.cosyvoice.inference_zero_shot_no_normalize_cached_prompt(
        content_to_synthesize_bopomo,
        request.app.state.prompt_cache,
    )
    inference_seconds = time.perf_counter() - inference_started
    encode_started = time.perf_counter()
    audio_buffer = BytesIO()
    sf.write(
        audio_buffer,
        output["tts_speech"].squeeze(0).cpu().numpy(),
        22050,
        format="WAV",
    )
    audio_buffer.seek(0)
    encode_seconds = time.perf_counter() - encode_started
    total_seconds = time.perf_counter() - request_started
    print(
        "BreezyVoice request timing:",
        {
            "text": payload.input,
            "preprocess_seconds": round(preprocess_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "encode_seconds": round(encode_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "output_samples": int(output["tts_speech"].shape[1]),
        },
    )
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8080)
