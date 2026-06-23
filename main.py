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

# In-memory tutor session store: {chat_id: {"text": str, "page": int}}
tutor_sessions = {}


# ─── INTERNET ARCHIVE SEARCH ─────────────────────────────────────────────────
async def search_internet_archive(topic: str) -> list:
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": topic,
        "fl[]": ["identifier", "title", "description", "mediatype"],
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
                mediatype = doc.get("mediatype", "")
                link = f"https://archive.org/details/{identifier}"
                results.append({
                    "title": title,
                    "link": link,
                    "type": mediatype,
                    "identifier": identifier
                })
            return results
    except Exception as e:
        logger.error(f"Internet Archive error: {e}")
        return []


# ─── FETCH PDF FROM INTERNET ARCHIVE ─────────────────────────────────────────
async def fetch_pdf_from_archive(identifier: str) -> bytes | None:
    """Try to download a PDF from Internet Archive by identifier."""
    metadata_url = f"https://archive.org/metadata/{identifier}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(metadata_url)
            data = resp.json()
            files = data.get("files", [])

            # Find a PDF file
            pdf_file = None
            for f in files:
                if f.get("name", "").lower().endswith(".pdf"):
                    pdf_file = f.get("name")
                    break

            if not pdf_file:
                return None

            pdf_url = f"https://archive.org/download/{identifier}/{pdf_file}"
            pdf_resp = await client.get(pdf_url, follow_redirects=True, timeout=60)
            if pdf_resp.status_code == 200:
                return pdf_resp.content, pdf_file
            return None, None
    except Exception as e:
        logger.error(f"PDF fetch error: {e}")
        return None, None


# ─── LEARN ANYTHING SEARCH ───────────────────────────────────────────────────
async def search_learn_anything(topic: str) -> list:
    url = f"https://learnanything.xyz/api/search?q={topic}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                items = data if isinstance(data, list) else data.get("results", [])
                for item in items[:3]:
                    title = item.get("title", item.get("name", "Resource"))
                    link = item.get("url", item.get("link", ""))
                    results.append({"title": title, "link": link})
                return results
            return []
    except Exception as e:
        logger.error(f"LearnAnything error: {e}")
        return []


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
def summarize_with_groq(topic: str, archive_results: list, learn_results: list) -> str:
    archive_text = "\n".join(
        [f"- {r['title']} ({r['type']}): {r['link']}" for r in archive_results]
    ) or "No results found."
    learn_text = "\n".join(
        [f"- {r['title']}: {r['link']}" for r in learn_results]
    ) or "No results found."

    prompt = f"""You are MindVault, an AI learning assistant. A user wants to learn about: "{topic}"

Internet Archive resources:
{archive_text}

LearnAnything resources:
{learn_text}

Write a clear, engaging learning summary:
1. A brief 2-3 sentence explanation of what "{topic}" is
2. What they will gain from these resources
3. A beginner tip or encouragement

Friendly, motivating, easy to understand. Max 200 words."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── GROQ PDF TUTOR ──────────────────────────────────────────────────────────
def tutor_with_groq(document_text: str, user_message: str, is_first: bool) -> str:
    # Limit text to avoid token overflow
    truncated = document_text[:6000]

    if is_first:
        prompt = f"""You are MindVault Tutor — an expert AI teacher. A student just uploaded a document for you to teach them from.

Here is the document content:
---
{truncated}
---

Your job:
1. Give a warm welcome and brief overview of what this document is about (2-3 sentences)
2. Break down the KEY topics/chapters you will teach them
3. Start teaching the FIRST topic clearly with examples
4. End by asking: "Ready for the next section? Or do you have questions about this part?"

Be engaging, use simple language, and teach like a real tutor."""
    else:
        prompt = f"""You are MindVault Tutor — an expert AI teacher currently tutoring a student.

Document content (for reference):
---
{truncated}
---

Student says: "{user_message}"

Respond as their tutor:
- If they say "next", teach the next concept from the document
- If they ask a question, answer it clearly using the document
- If they're confused, re-explain with a simpler example
- Always end with a question or prompt to keep them engaged"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── FORMAT TOPIC MESSAGE ─────────────────────────────────────────────────────
def format_message(topic: str, summary: str, archive: list, learn: list, photographer: str | None) -> str:
    msg = f"🧠 *MindVault — {topic.title()}*\n\n"
    msg += f"{summary}\n\n"

    if archive:
        msg += "📚 *Free Resources (Internet Archive):*\n"
        for r in archive[:3]:
            emoji = "🎬" if r["type"] == "movies" else "🎧" if r["type"] == "audio" else "📖"
            msg += f"{emoji} [{r['title']}]({r['link']})\n"
        msg += "\n"

    if learn:
        msg += "🔗 *Learning Paths (LearnAnything):*\n"
        for r in learn:
            msg += f"➡️ [{r['title']}]({r['link']})\n"
        msg += "\n"

    if photographer:
        msg += f"📷 _Photo by {photographer} on Unsplash_\n\n"

    msg += "📄 _Sending you a free PDF on this topic..._"
    return msg


# ─── HANDLERS ────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *MindVault* — your AI-powered learning assistant!\n\n"
        "Here's what I can do:\n\n"
        "🔍 *Learn any topic* — just type it!\n"
        "📄 *Get a free PDF* — I'll find & send one from the Internet Archive\n"
        "📚 *Upload a document* — send me any PDF and I'll tutor you through it\n\n"
        "Try typing: `Python programming` or `Nigerian history` or `Photosynthesis`\n\n"
        "Or send me a PDF to start a tutoring session! 🎓",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *MindVault Help*\n\n"
        "*To learn a topic:*\n"
        "Just type any subject e.g. `Machine Learning`\n\n"
        "*To get tutored from your own document:*\n"
        "Send any PDF file and I'll read it and teach you!\n\n"
        "*During a tutoring session:*\n"
        "• Type `next` to move to the next section\n"
        "• Ask any question about the document\n"
        "• Type `/stop` to end the session\n\n"
        "I find resources, send PDFs, and teach you like a real tutor! 🚀",
        parse_mode="Markdown"
    )


async def stop_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id in tutor_sessions:
        del tutor_sessions[chat_id]
        await update.message.reply_text(
            "✅ Tutoring session ended. Type any topic to learn something new!",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("No active session. Type a topic to start learning!")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle PDF uploads — start a tutor session."""
    doc = update.message.document

    if not doc.mime_type == "application/pdf":
        await update.message.reply_text("⚠️ Please send a *PDF* file for tutoring.", parse_mode="Markdown")
        return

    await update.message.reply_text(
        "📖 Got your document! Reading it now...\n_This might take a few seconds._",
        parse_mode="Markdown"
    )

    try:
        # Download the file from Telegram
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()

        # Extract text from PDF using pypdf
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(bytes(file_bytes)))
        text = ""
        for page in reader.pages[:20]:  # Max 20 pages
            text += page.extract_text() or ""

        if not text.strip():
            await update.message.reply_text(
                "⚠️ I couldn't extract text from this PDF. It might be image-based. "
                "Try a text-based PDF!"
            )
            return

        # Save session
        chat_id = update.message.chat_id
        tutor_sessions[chat_id] = {"text": text, "filename": doc.file_name}

        # Start tutoring
        response = tutor_with_groq(text, "", is_first=True)
        await update.message.reply_text(
            f"🎓 *MindVault Tutor — {doc.file_name}*\n\n{response}",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Document handling error: {e}")
        await update.message.reply_text(
            "⚠️ Something went wrong reading your document. Please try again."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text — either tutor reply or topic search."""
    chat_id = update.message.chat_id
    text = update.message.text.strip()

    if not text:
        return

    # If there's an active tutor session, continue tutoring
    if chat_id in tutor_sessions:
        session = tutor_sessions[chat_id]
        response = tutor_with_groq(session["text"], text, is_first=False)
        await update.message.reply_text(
            f"🎓 *MindVault Tutor*\n\n{response}\n\n"
            "_Type `next` to continue or ask a question. /stop to end session._",
            parse_mode="Markdown"
        )
        return

    # Otherwise do a topic search
    topic = text
    await update.message.reply_text(f"🔍 Researching *{topic}*... please wait!", parse_mode="Markdown")

    archive_task = search_internet_archive(topic)
    learn_task = search_learn_anything(topic)
    image_task = fetch_unsplash_image(topic)

    archive_results, learn_results, (image_url, photographer) = await asyncio.gather(
        archive_task, learn_task, image_task
    )

    summary = summarize_with_groq(topic, archive_results, learn_results)
    message = format_message(topic, summary, archive_results, learn_results, photographer)

    # Send image + summary
    if image_url:
        await update.message.reply_photo(photo=image_url, caption=message, parse_mode="Markdown")
    else:
        await update.message.reply_text(message, parse_mode="Markdown")

    # Try to find and send a PDF
    pdf_sent = False
    for result in archive_results:
        if result["type"] == "texts":
            await update.message.reply_text("📥 _Looking for a free PDF to send you..._", parse_mode="Markdown")
            pdf_data, filename = await fetch_pdf_from_archive(result["identifier"])
            if pdf_data:
                await update.message.reply_document(
                    document=io.BytesIO(pdf_data),
                    filename=filename or f"{topic}.pdf",
                    caption=f"📖 *{result['title']}*\n_Free from Internet Archive_",
                    parse_mode="Markdown"
                )
                pdf_sent = True
                break

    if not pdf_sent:
        await update.message.reply_text(
            "📄 _No downloadable PDF found for this topic, but check the links above!_\n\n"
            "💡 _You can also send me your own PDF and I'll tutor you through it!_",
            parse_mode="Markdown"
        )


# ─── PING SERVER (for UptimeRobot) ───────────────────────────────────────────
class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"MindVault is alive!")

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs


def run_ping_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), PingHandler)
    logger.info(f"Ping server running on port {port}")
    server.serve_forever()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # Start ping server in background thread
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
