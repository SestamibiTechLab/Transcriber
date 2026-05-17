import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# Modal backend URL
MODAL_API = "https://sestamibitechlab--podcast-transcriber-fastapi-app.modal.run"

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/health", methods=["GET"])
def health():
    try:
        response = requests.get(f"{MODAL_API}/health", timeout=5)
        if response.ok:
            data = response.json()
            return jsonify({
                "status": "ok",
                "model": data.get("model"),
                "device": data.get("device"),
                "backend": "Modal GPU"
            })
        else:
            return jsonify({"status": "error", "backend": "Modal unreachable"}), 503
    except:
        return jsonify({"status": "error", "backend": "Modal unreachable"}), 503

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    url = (data or {}).get("url", "").strip()
    language = (data or {}).get("language", "auto-detect")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Proxy request to Modal backend
        response = requests.post(
            f"{MODAL_API}/transcribe",
            json={"url": url, "language": language},
            timeout=3600  # 1 hour timeout
        )

        if response.ok:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            return jsonify({"error": error_data.get("detail", "Transcription failed")}), response.status_code
    except requests.Timeout:
        return jsonify({"error": "Transcription timeout - file may be too large"}), 504
    except Exception as e:
        return jsonify({"error": f"Request failed: {str(e)[:200]}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
