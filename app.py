import os
import whisper
import tempfile
import yt_dlp
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# Load model once at startup (tiny fits in Render free tier RAM)
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")
print(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)
print("Model loaded.")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    url = (data or {}).get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")

            # yt-dlp handles podcast RSS, direct MP3, YouTube, Spotify previews, etc.
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_path,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }],
                "quiet": True,
                "no_warnings": True,
                # Cap download at 60 minutes of audio to protect free tier
                "match_filter": yt_dlp.utils.match_filter_func("duration < 3600"),
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # yt-dlp may append .mp3 extension
            if not os.path.exists(audio_path):
                audio_path = audio_path + ".mp3"

            if not os.path.exists(audio_path):
                return jsonify({"error": "Could not download audio from that URL"}), 400

            result = model.transcribe(audio_path, fp16=False)

            segments = [
                {"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"].strip()}
                for s in result.get("segments", [])
            ]

            return jsonify({
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": segments,
            })

    except yt_dlp.utils.DownloadError as e:
        return jsonify({"error": f"Download failed: {str(e)[:200]}"}), 400
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)[:300]}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_SIZE})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
