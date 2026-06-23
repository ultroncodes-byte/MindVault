import os
import io
import json
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
tutor_sessions = {}


# ─── INTENT UNDERSTANDING ─────────────────────────────────────────────────────
def understand_intent(user_message: str) -> dict:
    prompt = f"""You are an intent classifier for MindVault, an AI learning assistant.

Analyze this user message: "{user_message}"

Respond ONLY with raw JSON, no explanation, no markdown:

If user wants to learn a topic, search resources, understand a concept:
{{"intent": "learn_topic", "topic": "<cleaned topic name>", "reply": ""}}

If user is greeting:
{{"intent": "greeting", "topic": "", "reply": "<friendly greeting mentioning MindVault>"}}

If user is chatting casually (not learning):
{{"intent": "casual_chat", "topic": "", "reply": "<brief helpful response>"}}

Rules:
- "teach me bout python" → topic: "Python programming"
- "what is X", "explain X", "how does X work", "I want to learn X" → learn_topic
- Always return valid JSON only"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Intent error: {e}")
        return {"intent": "learn_topic", "topic": user_message, "reply": ""}


# ─── INTERNET ARCHIVE ─────────────────────────────────────────────────────────
async def search_internet_archive(topic: str) -> list:
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": f"{topic} AND mediatype:texts",
        "fl[]": ["identifier", "title", "mediatype"],
        "rows": 4, "page": 1, "output": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            docs = resp.json().get("response", {}).get("docs", [])
            return [{"title": d.get("title", "No title"),
                     "link": f"https://archive.org/details/{d.get('identifier','')}",
                     "identifier": d.get("identifier", "")} for d in docs[:4]]
    except Exception as e:
        logger.error(f"Archive error: {e}")
        return []


# ─── PROJECT GUTENBERG ────────────────────────────────────────────────────────
async def search_gutenberg(topic: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://gutendex.com/books/?search={topic}")
            books = resp.json().get("results", [])
            results = []
            for book in books[:3]:
                title = book.get("title", "Unknown")
                author = book.get("authors", [{}])[0].get("name", "Unknown") if book.get("authors") else "Unknown"
                formats = book.get("formats", {})
                pdf_url = formats.get("application/pdf", "")
                epub_url = formats.get("application/epub+zip", "")
                results.append({
                    "title": f"{title} — {author}",
                    "link": f"https://www.gutenberg.org/ebooks/{book.get('id')}",
                    "download": pdf_url or epub_url,
                    "source": "Gutenberg"
                })
            return results
    except Exception as e:
        logger.error(f"Gutenberg error: {e}")
        return []


# ─── OPEN LIBRARY ─────────────────────────────────────────────────────────────
async def search_open_library(topic: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://openlibrary.org/search.json?q={topic}&limit=3&has_fulltext=true"
            )
            docs = resp.json().get("docs", [])
            results = []
            for doc in docs[:3]:
                title = doc.get("title", "Unknown")
                author = doc.get("author_name", ["Unknown"])[0]
                olid = doc.get("key", "")
                # Try to get readable/downloadable version
                edition_key = doc.get("edition_key", [])
                download = ""
                if edition_key:
                    download = f"https://openlibrary.org/books/{edition_key[0]}"
                results.append({
                    "title": f"{title} — {author}",
                    "link": f"https://openlibrary.org{olid}",
                    "download": download,
                    "source": "Open Library"
                })
            return results
    except Exception as e:
        logger.error(f"Open Library error: {e}")
        return []


# ─── STANDARD EBOOKS ──────────────────────────────────────────────────────────
async def search_standard_ebooks(topic: str) -> list:
    """Search Standard Ebooks via their OPDS catalog."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://standardebooks.org/opds/all",
                headers={"Accept": "application/atom+xml"}
            )
            # Parse entries from OPDS feed
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            results = []
            topic_lower = topic.lower()
            for entry in entries:
                title_el = entry.find("atom:title", ns)
                title = title_el.text if title_el is not None else ""
                if topic_lower in title.lower():
                    link_el = entry.find("atom:link[@type='application/epub+zip']", ns)
                    if link_el is None:
                        link_el = entry.find("atom:link", ns)
                    link = link_el.get("href", "") if link_el is not None else ""
                    id_el = entry.find("atom:id", ns)
                    page = id_el.text if id_el is not None else link
                    results.append({
                        "title": title,
                        "link": page,
                        "download": link if link.startswith("http") else "",
                        "source": "Standard Ebooks"
                    })
                if len(results) >= 2:
                    break
            return results
    except Exception as e:
        logger.error(f"Standard Ebooks error: {e}")
        return []


# ─── DOAB (Open Access Books) ─────────────────────────────────────────────────
async def search_doab(topic: str) -> list:
    """Directory of Open Access Books — peer-reviewed academic books."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://directory.doabooks.org/rest/search?query={topic}&expand=metadata&limit=3",
                headers={"Accept": "application/json"}
            )
            if resp.status_code != 200:
                return []
            books = resp.json()
            results = []
            for book in books[:3]:
                metadata = book.get("metadata", [])
                title = next((m["value"] for m in metadata if m["key"] == "dc.title"), "Unknown")
                handle = book.get("handle", "")
                link = f"https://directory.doabooks.org/handle/{handle}" if handle else ""
                # Find PDF link
                pdf_link = next(
                    (m["value"] for m in metadata if m["key"] == "dc.identifier.uri" and "pdf" in m["value"].lower()),
                    ""
                )
                results.append({
                    "title": title,
                    "link": link,
                    "download": pdf_link,
                    "source": "DOAB"
                })
            return results
    except Exception as e:
        logger.error(f"DOAB error: {e}")
        return []


# ─── MANYBOOKS ────────────────────────────────────────────────────────────────
async def search_manybooks(topic: str) -> list:
    """ManyBooks free ebook search."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://manybooks.net/api/search.php?q={topic}&type=json"
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            books = data if isinstance(data, list) else data.get("books", [])
            results = []
            for book in books[:3]:
                title = book.get("title", "Unknown")
                author = book.get("author", "Unknown")
                book_id = book.get("pid", "")
                link = f"https://manybooks.net/titles/{book_id}.html" if book_id else ""
                download = book.get("download_url", "")
                results.append({
                    "title": f"{title} — {author}",
                    "link": link,
                    "download": download,
                    "source": "ManyBooks"
                })
            return results
    except Exception as e:
        logger.error(f"ManyBooks error: {e}")
        return []


# ─── DOWNLOAD FILE ────────────────────────────────────────────────────────────
async def download_file(url: str) -> bytes | None:
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        return None
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


# ─── UNSPLASH IMAGE ──────────────────────────────────────────────────────────
async def fetch_unsplash_image(topic: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.unsplash.com/search/photos",
                params={"query": topic, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
            )
            results = resp.json().get("results", [])
            if results:
                return results[0]["urls"]["regular"], results[0]["user"]["name"]
        return None, None
    except Exception as e:
        logger.error(f"Unsplash error: {e}")
        return None, None


# ─── GROQ SUMMARIZER ─────────────────────────────────────────────────────────
def summarize_with_groq(topic: str, all_results: dict) -> str:
    sources_text = ""
    for source, books in all_results.items():
        if books:
            sources_text += f"\n{source}:\n"
            sources_text += "\n".join([f"- {b['title']}: {b['link']}" for b in books])

    prompt = f"""You are MindVault, an expert AI learning assistant. The user wants to learn: "{topic}"

Resources found across multiple free libraries:
{sources_text or 'No resources found'}

Write a learning summary:
1. Clear 2-3 sentence explanation of "{topic}"
2. Why these resources are valuable for learning it
3. One practical beginner tip

Be friendly, motivating, and concise. Max 180 words."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── GROQ TUTOR ──────────────────────────────────────────────────────────────
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
3. Start teaching the FIRST topic with clear examples
4. End by asking if they are ready for the next section

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
- Always end with a prompt to keep them engaged
- Be warm, patient, and encouraging"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600, temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── EXTRACT PDF TEXT ─────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    text = ""
    # Method 1: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:20]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    # Method 2: pypdf fallback
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages[:20]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.warning(f"pypdf failed: {e}")

    return ""


# ─── FORMAT MESSAGE ───────────────────────────────────────────────────────────
def format_message(topic: str, summary: str, all_results: dict, photographer: str | None) -> str:
    source_emojis = {
        "Internet Archive": "📚",
        "Gutenberg": "📜",
        "Open Library": "🏛️",
        "Standard Ebooks": "✨",
        "DOAB": "🎓",
        "ManyBooks": "📕",
    }
    msg = f"🧠 *MindVault — {topic.title()}*\n\n{summary}\n\n"
    for source, books in all_results.items():
        if books:
            emoji = source_emojis.get(source, "📖")
            msg += f"{emoji} *{source}:*\n"
            for b in books[:2]:
                msg += f"• [{b['title']}]({b['link']})\n"
            msg += "\n"
    if photographer:
        msg += f"📷 _Photo by {photographer} on Unsplash_\n\n"
    msg += "💡 _Send me a PDF and I'll tutor you through it! /help for more._"
    return msg


# ─── HANDLERS ────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *MindVault* — your AI-powered learning assistant!\n\n"
        "🔍 *Learn any topic* — just type it naturally!\n"
        "📚 *Free books* — from 6 free libraries including Gutenberg, DOAB & more\n"
        "🎓 *Upload a PDF* — I'll tutor you through it section by section\n\n"
        "Try: `teach me Python` or `what is blockchain` or `explain photosynthesis`\n\n"
        "Or send me a PDF to start a tutoring session! 🎓",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *MindVault Help*\n\n"
        "*Learn a topic:* Type anything naturally\n"
        "• `teach me about AI`\n"
        "• `what is machine learning?`\n"
        "• `I want to understand blockchain`\n\n"
        "*Get tutored:* Send any PDF file\n\n"
        "*During tutoring:*\n"
        "• `next` — move to next section\n"
        "• Ask any question about the document\n"
        "• /stop — end the session\n",
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
        file = await context.bot.get_file(doc.file_id)
        file_bytes = bytes(await file.download_as_bytearray())
        text = extract_pdf_text(file_bytes)

        if not text:
            await update.message.reply_text(
                "⚠️ Couldn't extract text from this PDF.\n\n"
                "This usually happens with scanned PDFs. "
                "Please try a digitally created PDF."
            )
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
    user_message = update.message.text.strip()

    if not user_message:
        return

    # Active tutor session
    if chat_id in tutor_sessions:
        session = tutor_sessions[chat_id]
        try:
            response = tutor_with_groq(session["text"], user_message, is_first=False)
            await update.message.reply_text(
                f"🎓 *MindVault Tutor*\n\n{response}\n\n"
                "_Type `next` to continue or ask a question. /stop to end._",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Tutor error: {e}")
            await update.message.reply_text("⚠️ Error generating response. Please try again.")
        return

    # Intent understanding
    intent_data = understand_intent(user_message)
    intent = intent_data.get("intent", "learn_topic")
    topic = intent_data.get("topic", user_message)
    reply = intent_data.get("reply", "")

    if intent == "greeting":
        await update.message.reply_text(
            reply or "👋 Hello! I'm MindVault. Type any topic to start learning!",
            parse_mode="Markdown"
        )
        return

    if intent == "casual_chat":
        await update.message.reply_text(
            f"{reply}\n\n💡 _Want to learn something? Just type any topic!_",
            parse_mode="Markdown"
        )
        return

    # Learn topic
    await update.message.reply_text(f"🔍 Got it! Researching *{topic}*...", parse_mode="Markdown")

    try:
        results = await asyncio.gather(
            search_internet_archive(topic),
            search_gutenberg(topic),
            search_open_library(topic),
            search_standard_ebooks(topic),
            search_doab(topic),
            search_manybooks(topic),
            fetch_unsplash_image(topic),
            return_exceptions=True
        )

        def safe(r): return r if not isinstance(r, Exception) else []

        all_results = {
            "Internet Archive": safe(results[0]),
            "Gutenberg": safe(results[1]),
            "Open Library": safe(results[2]),
            "Standard Ebooks": safe(results[3]),
            "DOAB": safe(results[4]),
            "ManyBooks": safe(results[5]),
        }
        image_data = results[6] if not isinstance(results[6], Exception) else (None, None)
        image_url, photographer = image_data if isinstance(image_data, tuple) else (None, None)

        summary = summarize_with_groq(topic, all_results)
        message = format_message(topic, summary, all_results, photographer)

        if image_url:
            await update.message.reply_photo(photo=image_url, caption=message, parse_mode="Markdown")
        else:
            await update.message.reply_text(message, parse_mode="Markdown")

        # Try to download and send a book
        pdf_sent = False
        all_books = (
            safe(results[1]) +  # Gutenberg first (most reliable)
            safe(results[3]) +  # Standard Ebooks
            safe(results[5]) +  # ManyBooks
            safe(results[4])    # DOAB
        )
        for book in all_books:
            if book.get("download"):
                await update.message.reply_text(
                    f"📥 _Downloading a free book from {book.get('source', 'library')}..._",
                    parse_mode="Markdown"
                )
                try:
                    file_bytes = await asyncio.wait_for(
                        download_file(book["download"]), timeout=20
                    )
                    if file_bytes:
                        dl_url = book["download"]
                        ext = "epub" if "epub" in dl_url else "pdf"
                        await update.message.reply_document(
                            document=io.BytesIO(file_bytes),
                            filename=f"{topic}.{ext}",
                            caption=f"📗 *{book['title']}*\n_Free from {book.get('source', 'library')}_",
                            parse_mode="Markdown"
                        )
                        pdf_sent = True
                        break
                except asyncio.TimeoutError:
                    logger.warning(f"Download timed out for {book.get('source')}")
                    continue

        if not pdf_sent:
            await update.message.reply_text(
                "📄 _No direct download found for this topic. Check the links above for free books!_\n\n"
                "💡 _You can also send me your own PDF and I'll tutor you through it!_",
                parse_mode="Markdown"
            )

    except Exception as e:
        logger.error(f"Topic search error: {e}")
        await update.message.reply_text("⚠️ Something went wrong. Please try again.")


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
