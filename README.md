# MindVault 🧠
### Your AI-Powered Learning Assistant on Telegram

MindVault is a Telegram bot that teaches anyone — from students to developers — about any subject. It searches the web for free resources, downloads and sends real PDF books, summarizes everything with AI, and tutors users through documents they upload — all inside Telegram.

---

## What It Does

- **Topic Search** — Type any subject and MindVault finds free learning resources from Internet Archive and LearnAnything.xyz
- **AI Summary** — Groq (Llama 3.3 70B) reads the resources and explains the topic in clear, beginner-friendly language
- **Free PDF Delivery** — The bot downloads a real PDF book from Internet Archive and sends it directly to you in Telegram
- **Visual Context** — A relevant image is fetched from Unsplash to go with every topic
- **Document Tutor** — Send any PDF and MindVault reads it and teaches you through it like a real tutor, section by section
- **Always Online** — A built-in ping server keeps the bot alive 24/7 on Render's free tier via UptimeRobot

---

## How to Use

**Learn a topic:**
Just type any subject in the chat
```
Python programming
World War 2
Photosynthesis
How to build APIs
```

**Get tutored from your own document:**
Send any PDF file to the bot and it will:
1. Read the document
2. Give you an overview
3. Teach you section by section
4. Answer your questions about it

**During a tutoring session:**
- Type `next` to move to the next section
- Ask any question about the document
- Type `/stop` to end the session

---

## Commands

| Command | Description |
|--------|-------------|
| `/start` | Welcome message and introduction |
| `/help` | Show all features and how to use them |
| `/stop` | End an active tutoring session |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| python-telegram-bot | Telegram bot framework |
| Groq (llama-3.3-70b-versatile) | AI summarization and tutoring |
| Internet Archive API | Free books, videos, and PDFs |
| LearnAnything.xyz API | Curated learning paths |
| Unsplash API | Topic images |
| pypdf | Reading uploaded PDF documents |
| httpx | Async HTTP requests |
| Python http.server | Ping endpoint for UptimeRobot |

---

## Environment Variables

Create these on Render (or in a `.env` file for local testing):

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
UNSPLASH_ACCESS_KEY=your_unsplash_access_key
PYTHON_VERSION=3.11.9
```

---

## Deployment (Render)

1. Fork or clone this repository
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repo
4. Set the environment variables listed above
5. Set the **Start Command** to:
```
python main.py
```
6. Deploy

---

## Keeping It Alive (UptimeRobot)

Render's free tier sleeps after 15 minutes of inactivity. To keep MindVault online 24/7:

1. Go to [uptimerobot.com](https://uptimerobot.com) and create a free account
2. Add a new monitor:
   - Type: `HTTP(s)`
   - Name: `MindVault Bot`
   - URL: `https://your-render-url.onrender.com`
   - Interval: `5 minutes`
3. Save — UptimeRobot will ping your bot every 5 minutes to keep it awake

---

## Project Structure

```
mindvault/
├── main.py          # All bot logic in one file
├── requirements.txt # Python dependencies
├── runtime.txt      # Python version for Render
└── README.md        # This file
```

---
