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
from bs4 import BeautifulSoup

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


# ─── BOOK SEARCH APIS ──────────────────────────────────────────────────────

async def search_gutenberg(topic: str) -> List[Dict[str, Optional[str]]]:
    """Search Project Gutenberg via gutendex.com."""
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
                download_url = formats.get("application/pdf") or formats.get("application/epub+zip") or formats.get("text/html")
                book_id = book.get("id")
                page_link = f"https://www.gutenberg.org/ebooks/{book_id}"
                results.append({
                    "source": "Project Gutenberg",
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
                    "source": "Open Library",
                    "title": f"{title} — {author}",
                    "link": link,
                    "download": None,
                })
            return results
    except Exception as e:
        logger.error(f"Open Library error: {e}")
        return []


async def search_pdf_drive(topic: str) -> List[Dict[str, Optional[str]]]:
    """Scrape PDF Drive for free PDF books (no official API)."""
    url = f"https://www.pdfdrive.com/search?q={topic}"
    results = []
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "lxml")
            # Each result is in a <div class="file-right"> or similar
            # Modern structure: <div class="file-info"> with <a> inside
            items = soup.select("div.file-info")
            for item in items[:3]:
                link_tag = item.find("a", href=True)
                if not link_tag:
                    continue
                title = link_tag.get_text(strip=True) or "Unknown"
                href = link_tag["href"]
                full_link = f"https://www.pdfdrive.com{href}" if href.startswith("/") else href
                # Try to get author from the <span> with class "author"
                author_tag = item.find("span", class_="author")
                author = author_tag.get_text(strip=True) if author_tag else "Unknown"
                results.append({
                    "source": "PDF Drive",
                    "title": f"{title} — {author}",
                    "link": full_link,
                    "download": full_link,  # direct link to the book page (download button)
                })
            return results
    except Exception as e:
        logger.error(f"PDF Drive error: {e}")
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


# ─── GROQ LEARNING PLAN GENERATOR ─────────────────────────────────────────

def generate_learning_plan(topic: str, resources: List[Dict]) -> Dict[str, str]:
    """
    Uses Groq to:
    - Select the best book from the resources.
    - Generate a step-by-step learning plan aligned with that book.
    Returns a dict with 'plan', 'recommended_book_title', 'recommended_book_link'.
    """
    if not resources:
        return {
            "plan": f"I couldn't find any free books for '{topic}'. Try uploading a PDF for tutoring!",
            "recommended_book_title": None,
            "recommended_book_link": None,
        }

    # Format resource list for the prompt
    resource_text = "\n".join([
        f"- [{r['source']}] {r['title']} (Link: {r['link']})"
        for r in resources[:6]  # limit to top 6
    ])

    prompt = f"""You are MindVault, an expert learning assistant. The user wants to learn: "{topic}".

Available books/resources:
{resource_text}

Your tasks:
1. Select the SINGLE best book from the list that would be most suitable for a beginner to learn this topic. Provide the title and the link.
2. Create a clear, step-by-step learning plan using that book. The plan should be a numbered list (e.g., 1. ..., 2. ...) with around 5–8 steps, covering foundational concepts, practice, and real-world application.

Output format (exactly):
Recommended Book: [title] (Link: [link])
Learning Plan:
1. [Step 1]
2. [Step 2]
...
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.6,
            timeout=15,
        )
        output = response.choices[0].message.content.strip()

        # Parse the response to extract book and plan
        lines = output.split("\n")
        plan_lines = []
        recommended_book_title = None
        recommended_book_link = None
        in_plan = False
        for line in lines:
            if line.startswith("Recommended Book:"):
                # Extract title and link
                # Example: "Recommended Book: Python Crash Course (Link: https://..."
                book_part = line.replace("Recommended Book:", "").strip()
                if "(Link:" in book_part:
                    title_part, link_part = book_part.split("(Link:", 1)
                    recommended_book_title = title_part.strip()
                    link_part = link_part.rstrip(")").strip()
                    recommended_book_link = link_part
                else:
                    recommended_book_title = book_part
            elif line.startswith("Learning Plan:"):
                in_plan = True
            elif in_plan and line.strip():
                plan_lines.append(line.strip())

        # If parsing failed, use the whole output as plan
        if not plan_lines and not recommended_book_title:
            # Fallback: treat everything after "Learning Plan:" as plan
            full_plan = output
            # Try to extract a book link from the first line if possible
            return {
                "plan": full_plan,
                "recommended_book_title": None,
                "recommended_book_link": None,
            }

        plan_text = "\n".join(plan_lines) if plan_lines else "Plan not available."
        return {
            "plan": plan_text,
            "recommended_book_title": recommended_book_title,
            "recommended_book_link": recommended_book_link,
        }
    except Exception as e:
        logger.error(f"Groq learning plan error: {e}")
        return {
            "plan": f"I found some books for '{topic}', but I couldn't generate a plan. Check the resources below!",
            "recommended_book_title": None,
            "recommended_book_link": None,
        }


# ─── FORMAT MESSAGE ───────────────────────────────────────────────────────────

def format_message(topic: str, plan_info: Dict, resources: List[Dict],
                   photographer: Optional[str]) -> str:
    plan = plan_info.get("plan", "No plan generated.")
    rec_title = plan_info.get("recommended_book_title")
    rec_link = plan_info.get("recommended_book_link")

    msg = f"🧠 *MindVault — {topic.title()}*\n\n"
    msg += f"📚 *Recommended Book:* "
    if rec_title and rec_link:
        msg += f"[{rec_title}]({rec_link})\n\n"
    else:
        msg += "None selected.\n\n"

    msg += f"📖 *Step‑by‑Step Learning Plan:*\n{plan}\n\n"

    # List all resources found
    if resources:
        msg += "📌 *All Resources Found:*\n"
        for r in resources[:6]:
            msg += f"• [{r['source']}] [{r['title']}]({r['link']})\n"
        msg += "\n"
    else:
        msg += "⚠️ No free books found for this topic. Try uploading a PDF to get tutored!\n\n"

    if photographer:
        msg += f"📷 _Photo by {photographer} on Unsplash_\n\n"

    msg += "💡 *Ask follow‑up questions* – I'll answer based on this plan! Or send a PDF for a full tutoring session. /help for more."
    return msg


# ─── HANDLERS ────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *MindVault* — your AI learning assistant!\n\n"
        "🔍 *Learn any topic* — just type it!\n"
        "📚 I'll search free books from Gutenberg, Open Library, and PDF Drive.\n"
        "🧠 Then I'll create a *step‑by‑step learning plan* with the best book.\n\n"
        "💬 Ask follow‑up questions about the plan.\n"
        "📄 Upload a PDF for one‑on‑one tutoring.\n\n"
        "Try: `Python programming` or `Nigerian history`",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *MindVault Help*\n\n"
        "*Learn a topic:* Just type any subject\n"
        "*Get a plan:* I'll recommend a book and give you a step‑by‑step learning roadmap.\n"
        "*Follow‑up:* Ask anything about the topic or the plan.\n"
        "*Upload PDF:* Get tutored through the document.\n\n"
        "*Commands:*\n/start — Welcome\n/help — This\n/stop — End sessions",
        parse_mode="Markdown"
    )


async def stop_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id in tutor_sessions:
        del tutor_sessions[chat_id]
    if chat_id in topic_sessions:
        del topic_sessions[chat_id]
    await update.message.reply_text("✅ Session ended. Start a new topic or upload a PDF.")


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


def tutor_with_groq(document_text: str, user_message: str, is_first: bool) -> str:
    """Generate tutor responses for PDF sessions (unchanged)."""
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

    # ── Follow‑up on previous topic (plan or book) ──
    if chat_id in topic_sessions:
        session = topic_sessions[chat_id]
        # If the user asks a short question, treat as follow-up using the stored plan and resources
        try:
            follow_up_prompt = f"""You are MindVault. The user is learning "{session['topic']}".
The learning plan we created is:
{session['plan']}

Recommended book: {session.get('recommended_book') or 'Not specified'}

User asks: "{text}"

Answer clearly, referencing the plan or book if relevant. Keep it under 250 words.
"""
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": follow_up_prompt}],
                max_tokens=400,
                temperature=0.7,
                timeout=15,
            )
            answer = response.choices[0].message.content.strip()
            await update.message.reply_text(
                f"🤔 *Follow‑up on {session['topic'].title()}*\n\n{answer}\n\n"
                "_Keep asking, or type a new topic to start fresh._",
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
        # Search all three sources concurrently
        gutenberg_task = asyncio.wait_for(search_gutenberg(topic), timeout=10)
        openlibrary_task = asyncio.wait_for(search_open_library(topic), timeout=10)
        pdfdrive_task = asyncio.wait_for(search_pdf_drive(topic), timeout=12)
        image_task = asyncio.wait_for(fetch_unsplash_image(topic), timeout=5)

        gutenberg_results, openlibrary_results, pdfdrive_results, image_data = await asyncio.gather(
            gutenberg_task, openlibrary_task, pdfdrive_task, image_task,
            return_exceptions=True
        )

        # Handle errors
        if isinstance(gutenberg_results, Exception):
            logger.error(f"Gutenberg error: {gutenberg_results}")
            gutenberg_results = []
        if isinstance(openlibrary_results, Exception):
            logger.error(f"Open Library error: {openlibrary_results}")
            openlibrary_results = []
        if isinstance(pdfdrive_results, Exception):
            logger.error(f"PDF Drive error: {pdfdrive_results}")
            pdfdrive_results = []
        if isinstance(image_data, Exception) or not isinstance(image_data, tuple):
            image_url, photographer = None, None
        else:
            image_url, photographer = image_data

        # Combine all resources
        all_resources = gutenberg_results + openlibrary_results + pdfdrive_results

        # Generate learning plan
        plan_info = generate_learning_plan(topic, all_resources)

        # Store session for follow-ups
        topic_sessions[chat_id] = {
            "topic": topic,
            "plan": plan_info.get("plan", ""),
            "recommended_book": plan_info.get("recommended_book_title"),
            "resources": all_resources,
        }
        if chat_id in tutor_sessions:
            del tutor_sessions[chat_id]

        # Build and send response
        message = format_message(topic, plan_info, all_resources, photographer)
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

    logger.info("MindVault bot is running with learning plans!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
