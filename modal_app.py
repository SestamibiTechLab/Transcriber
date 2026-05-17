"""
Podcast Transcriber - Modal Deployment
Serverless transcription service with GPU support and permanent URL
"""

import os
import tempfile
import yt_dlp
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Modal setup
app = modal.App("podcast-transcriber")

# Use GPU-enabled image with Whisper and dependencies pre-installed
image = (
    modal.Image.debian_slim()
    .pip_install(
        "openai-whisper",
        "yt-dlp",
        "torch",
        "numpy",
    )
    .apt_install("ffmpeg")
)

# Shared volume for model caching
cache_vol = modal.Volume.from_name("whisper-cache", create_if_missing=True)

web_app = FastAPI()


class TranscribeRequest(BaseModel):
    url: str
    language: str = "auto-detect"


@app.cls(
    image=image,
    gpu="any",
    volumes={"/cache": cache_vol},
    timeout=3600,  # 1 hour max
)
class WhisperTranscriber:
    def __init__(self):
        import whisper
        print("Loading Whisper model...")
        self.model = whisper.load_model("base", device="cuda")
        print("✅ Model loaded on GPU")

    @modal.method()
    def transcribe(self, url: str, language: str = None) -> dict:
        import whisper

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_path,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "128",
                    }
                ],
                "quiet": True,
                "no_warnings": True,
                "match_filter": yt_dlp.utils.match_filter_func("duration < 7200"),
            }

            print(f"Downloading: {url}")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e:
                raise ValueError(f"Download failed: {str(e)[:200]}")

            # Handle yt-dlp extension appending
            if not os.path.exists(audio_path):
                audio_path = audio_path + ".mp3"

            if not os.path.exists(audio_path):
                raise ValueError("Could not download audio from that URL")

            file_size = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"Downloaded: {file_size:.1f}MB")

            print("Transcribing...")
            result = self.model.transcribe(
                audio_path,
                language=language if language and language != "auto-detect" else None
            )

            segments = [
                {
                    "start": round(s["start"], 1),
                    "end": round(s["end"], 1),
                    "text": s["text"].strip(),
                }
                for s in result.get("segments", [])
            ]

            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": segments,
            }


transcriber = WhisperTranscriber()


@web_app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "base",
        "device": "cuda",
        "service": "Modal Transcriber",
    }


@web_app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="No URL provided")

    try:
        result = transcriber.transcribe.remote(
            request.url.strip(),
            request.language if request.language != "auto-detect" else None
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)[:300]}")


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
