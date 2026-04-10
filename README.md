# Telegram URL Summary Bot (MVP)

Production-oriented MVP Telegram bot that accepts a URL, extracts the underlying content, transcribes media locally with Whisper when needed, and returns a concise text summary.

## Supported inputs
- YouTube
- TikTok
- Instagram posts / reels
- Twitter/X
- VK
- Generic article URLs

## Stack
- `python-telegram-bot` v20+ (async)
- `requests` + `BeautifulSoup4` for generic article extraction
- `yt-dlp` for public media download
- `openai-whisper` for local transcription
- `Groq API` for summarization
- No database
- No paid APIs

## How it works
1. User sends one URL.
2. Bot detects the domain.
3. If the URL is a media platform, it downloads the media audio with `yt-dlp`.
4. Whisper transcribes locally.
5. Groq generates a structured summary.
6. Bot returns the result in Telegram.
7. Temporary files are deleted automatically.

## Prerequisites
- Python 3.10+
- `ffmpeg` installed and available in PATH

### Install ffmpeg
**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**macOS (Homebrew)**
```bash
brew install ffmpeg
```

**Windows (Chocolatey)**
```powershell
choco install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

## Setup in under 10 commands
```bash
git clone <your-repo-url>
cd kviker
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
python bot.py
```

Then:
1. Open your bot in Telegram.
2. Press **Start**.
3. Paste one public URL.

## Environment variables
Copy `.env.example` to `.env` and fill:
- `TELEGRAM_BOT_TOKEN` — free from `@BotFather`
- `GROQ_API_KEY` — free from Groq Console
- `GROQ_MODEL` — defaults to `llama-3.1-8b-instant`
- `WHISPER_MODEL` — `tiny` or `base`

## Example bot output
```text
📌 Title: Example Video Title
🗂 Source: YouTube
⏱ Length: 08:41

Summary:
• Main argument or takeaway
• Important evidence or examples
• Key names, numbers, or events
• Final conclusion or implication
```

## Notes
- This MVP processes **public URLs only**.
- Private, login-gated, DRM-protected, or unsupported pages will fail gracefully.
- Whisper is **lazy-loaded** on first transcription request to keep startup fast.
- Very long videos will work, but latency depends on your CPU and chosen Whisper model.
- `tiny` is the best default for a free-tier MVP.

## Deploy options
This bot runs locally with polling, or on any always-on Docker host such as:
- Railway
- Render
- Fly.io
- a small VPS

## Remote deployment
This repository is ready to deploy as a Dockerized service:
- `Dockerfile` installs Python dependencies and `ffmpeg`
- `bot.py` starts a tiny HTTP health endpoint when `PORT` is present
- `render.yaml` configures Render Blueprint deployment with `/healthz`

### Option A: Render
1. Push this repository to GitHub.
2. Open Render and create a new Blueprint or use the included `render.yaml`.
3. Set:
   - `TELEGRAM_BOT_TOKEN`
   - `GROQ_API_KEY`
4. Deploy.

Notes:
- Render Free web services spin down after 15 minutes without inbound traffic, so they are fine for testing but not ideal for a 24/7 Telegram polling bot.
- For continuous remote runtime, use a paid always-on instance on Render or another always-on Docker host.

### Option B: Railway
1. Create a new Railway project from this GitHub repo.
2. Railway will detect the `Dockerfile` automatically.
3. Set:
   - `TELEGRAM_BOT_TOKEN`
   - `GROQ_API_KEY`
4. Deploy the service.

The bot does not require a public webhook URL because it uses Telegram long polling.

## Troubleshooting
### 1. `ffmpeg` not found
Install `ffmpeg` and confirm `ffmpeg -version` works in the same shell session.

### 2. `ModuleNotFoundError: whisper`
Re-run:
```bash
pip install -r requirements.txt
```

### 3. Groq auth failure
Check that `GROQ_API_KEY` is set correctly in `.env`.

### 4. Telegram auth failure
Check that `TELEGRAM_BOT_TOKEN` is valid and belongs to the bot you started.

### 5. Media extraction fails
Common reasons:
- private post/video
- geo restriction
- login required
- platform-side anti-bot changes

The bot will return a readable error message instead of crashing.

## File overview
- `bot.py` — Telegram handlers, async orchestration, response formatting
- `extractor.py` — URL routing, article extraction, media download, Whisper transcription
- `summarizer.py` — Groq summarization client and prompt logic
