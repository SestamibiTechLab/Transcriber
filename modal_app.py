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
        "google-genai",
    )
    .apt_install("ffmpeg")
)

cache_vol = modal.Volume.from_name("whisper-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="any",
    volumes={"/cache": cache_vol},
    timeout=7200,  # 2 hours for large files
    secrets=[modal.Secret.from_name("google-api")],
)
class WhisperTranscriber:
    @modal.enter()
    def load_model(self):
        import whisper
        print("Loading Whisper model...")
        self.model = whisper.load_model("base", device="cuda")
        print("✅ Base model loaded on GPU")

    def improve_grammar(self, text: str) -> str:
        """Use Gemini to improve grammar and readability of transcription"""
        import google.genai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY - skipping grammar improvement")
            return text

        try:
            print(f"🔧 Improving grammar with Gemini... (text: {len(text)} chars)")
            client = genai.Client(api_key=api_key)

            prompt = f"""Please improve the grammar, punctuation, and readability of this transcribed text while preserving the original meaning and content. Fix any run-on sentences, missing punctuation, and awkward phrasing:

{text}

Return only the improved text, without any explanation or preamble."""

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            improved = response.text.strip()
            print(f"✅ Grammar improvement applied! ({len(improved)} chars)")
            return improved
        except Exception as e:
            print(f"❌ Grammar improvement failed: {type(e).__name__}: {str(e)}")
            print("Returning original text")
            return text

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
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(url, timeout=300, stream=True, headers=headers, allow_redirects=True)
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    if total_size > 500 * 1024 * 1024:  # 500MB limit
                        raise ValueError(f"File too large: {total_size / 1024 / 1024:.0f}MB (max 500MB)")

                    downloaded = 0
                    with open(audio_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if downloaded > 500 * 1024 * 1024:
                                    raise ValueError("File too large - exceeded 500MB limit during download")

                    print(f"Downloaded {downloaded / 1024 / 1024:.1f}MB")
                except Exception as e:
                    print(f"Audio file download error: {str(e)}")
                    raise ValueError(f"Failed to download audio: {str(e)[:200]}")
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

            if not os.path.exists(audio_path):
                raise ValueError("Audio file was not created successfully")

            file_size = os.path.getsize(audio_path) / 1024 / 1024
            print(f"Downloaded {file_size:.1f}MB, transcribing...")

            if file_size == 0:
                raise ValueError("Downloaded file is empty")

            try:
                result = self.model.transcribe(
                    audio_path,
                    language=language if language and language != "auto-detect" else None,
                    fp16=True,
                    initial_prompt="Add proper punctuation and capitalization. Break into complete sentences.",
                )
            except Exception as e:
                print(f"Transcription error: {str(e)}")
                raise ValueError(f"Transcription failed: {str(e)[:200]}")

            # Improve grammar using Gemini (optional)
            full_text = result["text"].strip()
            full_text = self.improve_grammar(full_text)

            # Format text with paragraphs every 5 sentences
            sentences = full_text.split(". ")
            paragraphs = []
            for i in range(0, len(sentences), 5):
                paragraph = ". ".join(sentences[i:i+5])
                if not paragraph.endswith("."):
                    paragraph += "."
                paragraphs.append(paragraph)
            formatted_text = "\n\n".join(paragraphs)

            segments = [
                {
                    "start": round(s["start"], 1),
                    "end": round(s["end"], 1),
                    "text": s["text"].strip(),
                }
                for s in result.get("segments", [])
            ]

            return {
                "text": formatted_text,
                "language": result.get("language", "unknown"),
                "segments": segments,
            }


@app.function(image=image, timeout=7200, secrets=[modal.Secret.from_name("google-api")])
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
