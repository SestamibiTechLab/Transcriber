# Podcast Transcriber

Transcribe any podcast episode from a URL using OpenAI Whisper, hosted free on Render.

## Deploy to Render (free)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/podcast-transcriber.git
git push -u origin main
```

### 2. Create a Render Web Service
1. Go to https://render.com and sign up (free)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` — hit **Deploy**

### 3. Wait for build (~5 min first time)
Render installs ffmpeg, Python deps, and downloads the Whisper model.

### 4. Open your app
Your URL will be `https://podcast-transcriber.onrender.com` (or similar).
Paste it into the "API URL" field in the app if testing locally.

## Notes

- **Free tier limits**: 512MB RAM, spins down after 15 min idle (first request after spin-down takes ~30s)
- **Model**: `base` by default (good accuracy, fits in RAM). Change `WHISPER_MODEL` env var to `tiny` if you hit memory issues, or `small` if you upgrade to a paid plan.
- **Max episode length**: capped at 90 minutes to protect free tier resources
- **Supported URLs**: Direct MP3/M4A links, most podcast RSS episode URLs, YouTube, and many podcast platforms (via yt-dlp)
- **Output formats**: Full text, timestamped segments, .txt download, .srt subtitle download

## Local development

```bash
pip install -r requirements.txt
# Install ffmpeg: brew install ffmpeg (mac) or sudo apt install ffmpeg (linux)
python app.py
```
App runs at http://localhost:5000
