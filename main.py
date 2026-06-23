import os
import io
import asyncio
import logging
import threading
import httpx
from http.server import HTTPServer, BaseHTTPRequestHandler
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

# In-memory tutor session store
tutor_sessions = {}


# ─── INTERNET ARCHIVE SEARCH (text only) ─────────────────────────────────────
async def search_internet_archive(topic: str) -> list:
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": f"{topic} AND mediatype:texts",
        "fl[]": ["identifier", "title", "mediatype"],
        "rows": 5,
        "page": 1,
        "output": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            docs = data.get("response", {}).get("docs", [])
            results = []
            for doc in docs[:5]:
                identifier = doc.get("identifier", "")
                title = doc.get("title", "No title")
                link = f"https://archive.org/details/{identifier}"
                results.append({
                    "title": title,
                    "link": link,
                    "type": "texts",
                    "identifier": identifier
                })
            return results
    except Exception as e:
        logger.error(f"Internet Archive error: {e}")
        return []


# ─── PROJECT GUTENBERG SEARCH ─────────────────────────────────────────────────
async def search_gutenberg(topic: str) -> list:
    url = f"https://gutendex.com/books/?search={topic}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
            books = data.get("results", [])
            results = []
            for book in books[:3]:
                title = book.get("title", "Unknown")
                authors = book.get("authors", [])
                author = authors[0]["name"] if authors else "Unknown"
                formats = book.get("formats", {})
                # Get PDF or epub link
                pdf_url = formats.get("application/pdf", "")
                epub_url = formats.get("application/epub+zip", "")
                html_url = formats.get("text/html", "")
                read_link = pdf_url or epub_url or html_url
                book_id = book.get("id")
                page_link = f"https://www.gutenberg.org/ebooks/{book_id}"
                results.append({
                    "title": f"{title} — {author}",
                    "link": page_link,
                    "download": pdf_url or epub_url,
                })
            return results
    except Exception as e:
        logger.error(f"Gutenberg error: {e}")
        return []


# ─── OPEN LIBRARY SEARCH ──────────────────────────────────────────────────────
async def search_open_library(topic: str) -> list:
    url = f"https://openlibrary.org/search.json?q={topic}&limit=3&has_fulltext=true"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
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
                })
            return results
    except Exception as e:
        logger.error(f"Open Library error: {e}")
        return []


# ─── DOWNLOAD PDF FROM GUTENBERG ──────────────────────────────────────────────
async def download_gutenberg_pdf(download_url: str) -> bytes | None:
    if not download_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(download_url)
            if resp.status_code == 200:
                return resp.content
        return None
    except Exception as e:
        logger.error(f"Gutenberg download error: {e}")
        return None


# ─── UNSPLASH IMAGE ──────────────────────────────────────────────────────────
async def fetch_unsplash_image(topic: str):
    url = "https://api.unsplash.com/search/photos"
    params = {"query": topic, "per_page": 1, "orientation": "landscape"}
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
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
def summarize_with_groq(topic: str, archive: list, gutenberg: list, openlibrary: list) -> str:
    archive_text = "\n".join([f"- {r['title']}: {r['link']}" for r in archive]) or "None"
    gutenberg_text = "\n".join([f"- {r['title']}: {r['link']}" for r in gutenberg]) or "None"
    openlibrary_text = "\n".join([f"- {r['title']}: {r['link']}" for r in openlibrary]) or "None"

    prompt = f"""You are MindVault, an expert AI learning assistant. The user wants to learn: "{topic}"

Resources found:
Internet Archive: {archive_text}
Project Gutenberg: {gutenberg_text}
Open Library: {openlibrary_text}

Write a learning summary with:
1. A clear 2-3 sentence explanation of what "{topic}" is
2. Why these resources are valuable for learning it
3. One practical beginner tip

Be friendly, motivating, and concise. Max 180 words."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── GROQ PDF TUTOR ──────────────────────────────────────────────────────────
def tutor_with_groq(document_text: str, user_message: str, is_first: bool) -> str:
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

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── FORMAT MESSAGE ───────────────────────────────────────────────────────────
def format_message(topic: str, summary: str, archive: list, gutenberg: list, openlibrary: list, photographer: str | None) -> str:
    msg = f"🧠 *MindVault — {topic.title()}*\n\n{summary}\n\n"

    if archive:
        msg += "📚 *Internet Archive:*\n"
        for r in archive[:3]:
            msg += f"📖 [{r['title']}]({r['link']})\n"
        msg += "\n"

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

    if photographer:
        msg += f"📷 _Photo by {photographer} on Unsplash_\n\n"

    msg += "💡 _Send me a PDF and I'll tutor you through it! /help for more._"
    return msg


# ─── HANDLERS ────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *MindVault* — your AI-powered learning assistant!\n\n"
        "🔍 *Learn any topic* — just type it!\n"
        "📄 *Get free books* — from Gutenberg, Open Library & Internet Archive\n"
        "📚 *Upload a PDF* — I'll tutor you through it section by section\n\n"
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
        "*Commands:*\n"
        "/start — Welcome\n"
        "/help — This message\n"
        "/stop — End tutoring session",
        parse_mode="Markdown"
    )


async def stop_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id in tutor_sessions:
        del tutor_sessions[chat_id]
        await update.message.reply_text("✅ Session ended. Type any topic to keep learning!")
    else:
        await update.message.reply_text("No active session. Type a topic to start learning!")


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

    # Active tutor session
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

    # Topic search
    topic = text
    await update.message.reply_text(f"🔍 Researching *{topic}*...", parse_mode="Markdown")

    try:
        # Run searches concurrently with timeout
        archive_task = search_internet_archive(topic)
        gutenberg_task = search_gutenberg(topic)
        openlibrary_task = search_open_library(topic)
        image_task = fetch_unsplash_image(topic)

        results = await asyncio.gather(
            archive_task, gutenberg_task, openlibrary_task, image_task,
            return_exceptions=True
        )

        archive_results = results[0] if not isinstance(results[0], Exception) else []
        gutenberg_results = results[1] if not isinstance(results[1], Exception) else []
        openlibrary_results = results[2] if not isinstance(results[2], Exception) else []
        image_data = results[3] if not isinstance(results[3], Exception) else (None, None)
        image_url, photographer = image_data if isinstance(image_data, tuple) else (None, None)

        # Summarize
        summary = summarize_with_groq(topic, archive_results, gutenberg_results, openlibrary_results)
        message = format_message(topic, summary, archive_results, gutenberg_results, openlibrary_results, photographer)

        # Send image + message
        if image_url:
            await update.message.reply_photo(photo=image_url, caption=message, parse_mode="Markdown")
        else:
            await update.message.reply_text(message, parse_mode="Markdown")

        # Try to send a PDF from Gutenberg (non-blocking)
        pdf_sent = False
        for book in gutenberg_results:
            if book.get("download"):
                await update.message.reply_text("📥 _Downloading a free book for you..._", parse_mode="Markdown")
                try:
                    pdf_bytes = await asyncio.wait_for(
                        download_gutenberg_pdf(book["download"]), timeout=20
                    )
                    if pdf_bytes:
                        ext = "epub" if "epub" in book["download"] else "pdf"
                        await update.message.reply_document(
                            document=io.BytesIO(pdf_bytes),
                            filename=f"{topic}.{ext}",
                            caption=f"📗 *{book['title']}*\n_Free from Project Gutenberg_",
                            parse_mode="Markdown"
                        )
                        pdf_sent = True
                        break
                except asyncio.TimeoutError:
                    logger.warning("Gutenberg download timed out")
                    break

        if not pdf_sent:
            await update.message.reply_text(
                "📄 _No direct download available for this topic, but check the links above for free books!_\n\n"
                "💡 _You can also send me your own PDF and I'll tutor you through it!_",
                parse_mode="Markdown"
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
