import os
import io
import asyncio
import logging
import threading
import httpx
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any, Tuple
from groq import Groq
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── ENV VARS ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# In-memory stores
tutor_sessions: Dict[int, Dict[str, Any]] = {}
topic_sessions: Dict[int, Dict[str, Any]] = {}


# ─── SEARCH APIS (reliable and fast) ──────────────────────────────────────

async def search_gutenberg(topic: str) -> List[Dict[str, Optional[str]]]:
    """Search Project Gutenberg via gutendex.com (free, fast, returns downloadable formats)."""
    url = f"https://gutendex.com/books/?search={topic}"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url)
            data = resp.json()
            books = data.get("results", [])
            results = []
            for book in books[:3]:
                title = book.get("title", "Unknown")
                authors = book.get("authors", [])
                author = authors[0]["name"] if authors else "Unknown"
                formats = book.get("formats", {})
                # Prefer PDF, then EPUB, then HTML
                download_url = formats.get("application/pdf") or formats.get("application/epub+zip") or formats.get("text/html")
                book_id = book.get("id")
                page_link = f"https://www.gutenberg.org/ebooks/{book_id}"
                results.append({
                    "title": f"{title} — {author}",
                    "link": page_link,
                    "download": download_url,
                })
            return results
    except Exception as e:
        logger.error(f"Gutenberg error: {e}")
        return []


async def search_open_library(topic: str) -> List[Dict[str, Optional[str]]]:
    """Search Open Library for books with full text."""
    url = f"https://openlibrary.org/search.json?q={topic}&limit=3&has_fulltext=true"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url)
            data = resp.json()
            docs = data.get("docs", [])
            results = []
            for doc in docs[:3]:
                title = doc.get("title", "Unknown")
                author = doc.get("author_name", ["Unknown"])[0]
                olid = doc.get("key", "")
                link = f"https://openlibrary.org{olid}"
                results.append({
                    "title": f"{title} — {author}",
                    "link": link,
                    "download": None,   # Open Library does not provide direct PDF download via API easily
                })
            return results
    except Exception as e:
        logger.error(f"Open Library error: {e}")
        return []


async def fetch_unsplash_image(topic: str) -> Tuple[Optional[str], Optional[str]]:
    """Get a landscape photo from Unsplash."""
    url = "https://api.unsplash.com/search/photos"
    params = {"query": topic, "per_page": 1, "orientation": "landscape"}
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url, params=params, headers=headers)
            data = resp.json()
            results = data.get("results", [])
            if results:
                photo = results[0]
                return photo["urls"]["regular"], photo["user"]["name"]
        return None, None
    except Exception as e:
        logger.error(f"Unsplash error: {e}")
        return None, None


# ─── GROQ SUMMARIZER ─────────────────────────────────────────────────────────

def summarize_with_groq(topic: str, gutenberg: List[Dict], openlibrary: List[Dict]) -> str:
    """Generate a learning summary from the found resources."""
    gutenberg_text = "\n".join([f"- {r['title']}: {r['link']}" for r in gutenberg]) or "None"
    openlibrary_text = "\n".join([f"- {r['title']}: {r['link']}" for r in openlibrary]) or "None"

    prompt = f"""You are MindVault, an expert AI learning assistant. The user wants to learn: "{topic}"

Resources found:
Project Gutenberg: {gutenberg_text}
Open Library: {openlibrary_text}

Write a learning summary with:
1. A clear 2-3 sentence explanation of what "{topic}" is
2. Why these resources are valuable for learning it
3. One practical beginner tip

Be friendly, motivating, and concise. Max 180 words."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
            timeout=15,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq summary error: {e}")
        return f"Here's what I found about '{topic}'. Check the resources below!"


def follow_up_with_groq(topic: str, user_question: str, summary: str) -> str:
    """Answer a follow-up question about the topic using the previous context."""
    prompt = f"""You are MindVault, an expert AI teacher. The user is learning about "{topic}".

Previous summary:
{summary}

Now the user asks: "{user_question}"

Answer clearly and helpfully. If the question is general, give a concise but informative reply. If you don't know, suggest checking the provided resources. Keep it under 200 words."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
            timeout=15,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq follow-up error: {e}")
        return "I'm having trouble generating a response right now. Please try again or ask something else."


# ─── GROQ PDF TUTOR ──────────────────────────────────────────────────────────

def tutor_with_groq(document_text: str, user_message: str, is_first: bool) -> str:
    """Generate tutor responses for PDF sessions."""
    truncated = document_text[:6000]

    if is_first:
        prompt = f"""You are MindVault Tutor — an expert AI teacher. A student uploaded a document.

Document:
---
{truncated}
---

1. Give a warm welcome and brief overview (2-3 sentences)
2. List the KEY topics you will teach
3. Start teaching the FIRST topic with examples
4. End by asking if they are ready for the next section or have questions

Be engaging and use simple language."""
    else:
        prompt = f"""You are MindVault Tutor teaching a student from this document:
---
{truncated}
---

Student says: "{user_message}"

- If they say "next", teach the next concept
- If they ask a question, answer clearly from the document
- If confused, re-explain with a simpler example
- Always end with a prompt to keep them engaged"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
            timeout=15,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Tutor error: {e}")
        return "⚠️ Error generating tutor response. Please try again."


# ─── FORMAT MESSAGE ───────────────────────────────────────────────────────────

def format_message(topic: str, summary: str, gutenberg: List[Dict], openlibrary: List[Dict],
                   photographer: Optional[str]) -> str:
    msg = f"🧠 *MindVault — {topic.title()}*\n\n{summary}\n\n"

    if gutenberg:
        msg += "📜 *Project Gutenberg (Free Books):*\n"
        for r in gutenberg:
            msg += f"📗 [{r['title']}]({r['link']})\n"
        msg += "\n"

    if openlibrary:
        msg += "🏛️ *Open Library:*\n"
        for r in openlibrary:
            msg += f"📘 [{r['title']}]({r['link']})\n"
        msg += "\n"

    if not gutenberg and not openlibrary:
        msg += "⚠️ No free books found for this topic. Try uploading a PDF to get tutored!\n\n"

    if photographer:
        msg += f"📷 _Photo by {photographer} on Unsplash_\n\n"

    msg += "💡 *Ask me anything about this topic* — I'll answer! Or send a PDF for a full tutoring session. /help for more."
    return msg


# ─── HANDLERS ────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *MindVault* — your AI-powered learning assistant!\n\n"
        "🔍 *Learn any topic* — just type it!\n"
        "📄 *Get free books* — from Gutenberg & Open Library\n"
        "📚 *Upload a PDF* — I'll tutor you through it section by section\n"
        "💬 *Follow‑up questions* — after a topic search, just ask anything about it!\n\n"
        "Try: `Python programming` or `Nigerian history` or `Quantum physics`\n\n"
        "Or send me a PDF to start a tutoring session! 🎓",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *MindVault Help*\n\n"
        "*Learn a topic:* Just type any subject\n"
        "*Get tutored:* Send any PDF file\n\n"
        "*During tutoring:*\n"
        "• `next` — move to next section\n"
        "• Ask any question about the document\n"
        "• /stop — end the session\n\n"
        "*Topic follow‑up:* After a topic search, just ask anything about it (e.g., \"Tell me more about X\")\n\n"
        "*Commands:*\n"
        "/start — Welcome\n"
        "/help — This message\n"
        "/stop — End tutoring or clear topic session",
        parse_mode="Markdown"
    )


async def stop_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id in tutor_sessions:
        del tutor_sessions[chat_id]
    if chat_id in topic_sessions:
        del topic_sessions[chat_id]
    await update.message.reply_text("✅ Session ended. Type a topic or upload a PDF to start again!")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.mime_type != "application/pdf":
        await update.message.reply_text("⚠️ Please send a *PDF* file.", parse_mode="Markdown")
        return

    await update.message.reply_text("📖 Reading your document...", parse_mode="Markdown")

    try:
        import pypdf
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()
        reader = pypdf.PdfReader(io.BytesIO(bytes(file_bytes)))
        text = ""
        for page in reader.pages[:20]:
            text += page.extract_text() or ""

        if not text.strip():
            await update.message.reply_text("⚠️ Couldn't extract text. Try a text-based PDF!")
            return

        chat_id = update.message.chat_id
        tutor_sessions[chat_id] = {"text": text, "filename": doc.file_name}
        if chat_id in topic_sessions:
            del topic_sessions[chat_id]

        response = tutor_with_groq(text, "", is_first=True)
        await update.message.reply_text(
            f"🎓 *MindVault Tutor — {doc.file_name}*\n\n{response}\n\n"
            "_Type `next` to continue or ask any question. /stop to end._",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Document error: {e}")
        await update.message.reply_text("⚠️ Error reading document. Please try again.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    text = update.message.text.strip()

    if not text:
        return

    # ── Active PDF tutor session ──
    if chat_id in tutor_sessions:
        session = tutor_sessions[chat_id]
        try:
            response = tutor_with_groq(session["text"], text, is_first=False)
            await update.message.reply_text(
                f"🎓 *MindVault Tutor*\n\n{response}\n\n"
                "_Type `next` to continue or ask a question. /stop to end._",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Tutor error: {e}")
            await update.message.reply_text("⚠️ Error generating response. Please try again.")
        return

    # ── Follow‑up on previous topic search ──
    if chat_id in topic_sessions:
        session = topic_sessions[chat_id]
        try:
            response = follow_up_with_groq(session["topic"], text, session["summary"])
            await update.message.reply_text(
                f"🤔 *Follow‑up on {session['topic'].title()}*\n\n{response}\n\n"
                "_You can keep asking, or type a new topic to start fresh._",
                parse_mode="Markdown"
            )
            return
        except Exception as e:
            logger.error(f"Follow-up error: {e}")
            await update.message.reply_text("⚠️ Error generating follow-up. Please try again.")
            return

    # ── New topic search ──
    topic = text
    await update.message.reply_text(f"🔍 Researching *{topic}*...", parse_mode="Markdown")

    try:
        # Run searches concurrently with timeouts
        gutenberg_task = asyncio.wait_for(search_gutenberg(topic), timeout=10)
        openlibrary_task = asyncio.wait_for(search_open_library(topic), timeout=10)
        image_task = asyncio.wait_for(fetch_unsplash_image(topic), timeout=5)

        gutenberg_results, openlibrary_results, image_data = await asyncio.gather(
            gutenberg_task, openlibrary_task, image_task,
            return_exceptions=True
        )

        # Handle errors
        if isinstance(gutenberg_results, Exception):
            logger.error(f"Gutenberg error: {gutenberg_results}")
            gutenberg_results = []
        if isinstance(openlibrary_results, Exception):
            logger.error(f"Open Library error: {openlibrary_results}")
            openlibrary_results = []
        if isinstance(image_data, Exception) or not isinstance(image_data, tuple):
            image_url, photographer = None, None
        else:
            image_url, photographer = image_data

        # Generate summary
        summary = summarize_with_groq(topic, gutenberg_results, openlibrary_results)

        # Store topic session for follow-ups
        topic_sessions[chat_id] = {
            "topic": topic,
            "summary": summary,
            "resources": {
                "gutenberg": gutenberg_results,
                "openlibrary": openlibrary_results,
            }
        }
        if chat_id in tutor_sessions:
            del tutor_sessions[chat_id]

        # Build and send response
        message = format_message(topic, summary, gutenberg_results, openlibrary_results, photographer)
        if image_url:
            await update.message.reply_photo(photo=image_url, caption=message, parse_mode="Markdown")
        else:
            await update.message.reply_text(message, parse_mode="Markdown")

    except asyncio.TimeoutError:
        await update.message.reply_text(
            "⏰ The search took too long. Please try again with a more specific topic or upload a PDF for tutoring."
        )
    except Exception as e:
        logger.error(f"Topic search error: {e}")
        await update.message.reply_text(
            "⚠️ Something went wrong. Please try again or try a different topic."
        )


# ─── PING SERVER ─────────────────────────────────────────────────────────────

class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"MindVault is alive!")

    def log_message(self, format, *args):
        pass


def run_ping_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), PingHandler)
    logger.info(f"Ping server running on port {port}")
    server.serve_forever()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    ping_thread = threading.Thread(target=run_ping_server, daemon=True)
    ping_thread.start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stop", stop_session))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("MindVault bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
