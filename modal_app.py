"""
Podcast Transcriber - Modal Deployment
Serverless transcription service with GPU support and permanent URL
"""

import os
import tempfile
import modal

# Modal setup
app = modal.App("podcast-transcriber")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "openai-whisper",
        "torch",
        "yt-dlp",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pydantic",
        "requests",
    )
    .apt_install("ffmpeg")
)

cache_vol = modal.Volume.from_name("whisper-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="any",
    volumes={"/cache": cache_vol},
    timeout=3600,
)
class WhisperTranscriber:
    @modal.enter()
    def load_model(self):
        import whisper
        print("Loading Whisper model...")
        self.model = whisper.load_model("base", device="cuda")
        print("Model loaded on GPU")

    @modal.method()
    def transcribe(self, url: str, language: str = None) -> dict:
        import yt_dlp
        import requests

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")

            print(f"Downloading: {url}")

            # Check if it's a direct audio file (MP3, WAV, etc.)
            if url.lower().endswith(('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg')):
                try:
                    print("Detected direct audio file - downloading...")
                    response = requests.get(url, timeout=300, stream=True)
                    response.raise_for_status()
                    with open(audio_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                except Exception as e:
                    raise ValueError(f"Failed to download audio file: {str(e)[:200]}")
            else:
                # Use yt-dlp for platform URLs (YouTube, etc.)
                try:
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "outtmpl": audio_path,
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "128",
                        }],
                        "quiet": True,
                        "no_warnings": True,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                except Exception as e:
                    raise ValueError(f"Download failed: {str(e)[:200]}")

            if not os.path.exists(audio_path):
                audio_path = audio_path + ".mp3"

            if not os.path.exists(audio_path):
                raise ValueError("Could not download audio from that URL")

            print(f"Downloaded {os.path.getsize(audio_path) / 1024 / 1024:.1f}MB, transcribing...")

            result = self.model.transcribe(
                audio_path,
                language=language if language and language != "auto-detect" else None,
                fp16=True,
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


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    transcriber = WhisperTranscriber()

    class TranscribeRequest(BaseModel):
        url: str
        language: str = "auto-detect"

    @web_app.get("/")
    async def root():
        return {"status": "Podcast Transcriber API is running", "endpoints": ["/health", "/transcribe"]}

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": "base", "device": "cuda"}

    @web_app.post("/transcribe")
    async def transcribe(request: TranscribeRequest):
        if not request.url.strip():
            raise HTTPException(status_code=400, detail="No URL provided")
        try:
            result = transcriber.transcribe.remote(
                request.url.strip(),
                request.language if request.language != "auto-detect" else None,
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)[:300]}")

    return web_app
