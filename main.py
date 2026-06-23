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

groq_client = Groq(api_key=GROQ_API_KEY)


# ─── SAFE MESSAGE SENDER ──────────────────────────────────────────────────────
async def safe_reply(update: Update, text: str):
    """Send with Markdown, fall back to plain text if parse fails."""
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        try:
            plain = text.replace("*", "").replace("_", "").replace("`", "")
            await update.message.reply_text(plain)
        except Exception as e:
            logger.error(f"safe_reply failed: {e}")


TECH_TOPICS = [
    "programming", "python", "javascript", "java", "c++", "rust", "golang",
    "web development", "frontend", "backend", "fullstack", "react", "nodejs",
    "cybersecurity", "ethical hacking", "penetration testing", "networking",
    "data science", "machine learning", "artificial intelligence", "deep learning",
    "ui/ux", "product design", "figma", "cloud computing", "aws", "devops",
    "docker", "kubernetes", "linux", "git", "database", "sql", "mongodb",
    "blockchain", "web3", "solidity", "smart contracts", "mobile development",
    "flutter", "react native", "android", "ios", "swift", "kotlin"
]


# ─── GROQ HELPERS ─────────────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    """Escape characters that break Telegram Markdown."""
    # Replace unmatched backticks and special chars that break parsing
    import re
    # Remove triple backticks (code blocks cause issues in captions)
    text = re.sub(r'```[\w]*\n?', '', text)
    # Escape lone special characters that aren't part of markdown pairs
    text = text.replace("&", "&amp;")
    return text.strip()


def ask_groq(prompt: str, max_tokens: int = 800) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return sanitize(response.choices[0].message.content)


# ─── BOOK SOURCES ─────────────────────────────────────────────────────────────
async def search_gutenberg(topic: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://gutendex.com/books/?search={topic}")
            books = resp.json().get("results", [])
            results = []
            for book in books[:4]:
                title = book.get("title", "Unknown")
                author = book.get("authors", [{}])[0].get("name", "Unknown") if book.get("authors") else "Unknown"
                formats = book.get("formats", {})
                pdf_url = formats.get("application/pdf", "")
                epub_url = formats.get("application/epub+zip", "")
                results.append({
                    "title": f"{title} by {author}",
                    "link": f"https://www.gutenberg.org/ebooks/{book.get('id')}",
                    "download": pdf_url or epub_url,
                    "source": "Project Gutenberg"
                })
            return results
    except Exception as e:
        logger.error(f"Gutenberg error: {e}")
        return []


async def search_open_library(topic: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://openlibrary.org/search.json?q={topic}&limit=4&has_fulltext=true"
            )
            docs = resp.json().get("docs", [])
            results = []
            for doc in docs[:4]:
                title = doc.get("title", "Unknown")
                author = doc.get("author_name", ["Unknown"])[0]
                olid = doc.get("key", "")
                edition_key = doc.get("edition_key", [])
                download = f"https://openlibrary.org/books/{edition_key[0]}" if edition_key else ""
                results.append({
                    "title": f"{title} by {author}",
                    "link": f"https://openlibrary.org{olid}",
                    "download": download,
                    "source": "Open Library"
                })
            return results
    except Exception as e:
        logger.error(f"Open Library error: {e}")
        return []


async def search_internet_archive(topic: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://archive.org/advancedsearch.php",
                params={
                    "q": f"{topic} AND mediatype:texts AND subject:programming",
                    "fl[]": ["identifier", "title"],
                    "rows": 4, "page": 1, "output": "json",
                }
            )
            docs = resp.json().get("response", {}).get("docs", [])
            return [{
                "title": d.get("title", "No title"),
                "link": f"https://archive.org/details/{d.get('identifier','')}",
                "download": f"https://archive.org/download/{d.get('identifier','')}/{d.get('identifier','')}.pdf",
                "source": "Internet Archive"
            } for d in docs[:4]]
    except Exception as e:
        logger.error(f"Archive error: {e}")
        return []


async def search_doab(topic: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://directory.doabooks.org/rest/search?query={topic}&expand=metadata&limit=4",
                headers={"Accept": "application/json"}
            )
            if resp.status_code != 200:
                return []
            books = resp.json()
            results = []
            for book in books[:4]:
                metadata = book.get("metadata", [])
                title = next((m["value"] for m in metadata if m["key"] == "dc.title"), "Unknown")
                handle = book.get("handle", "")
                link = f"https://directory.doabooks.org/handle/{handle}" if handle else ""
                pdf_link = next(
                    (m["value"] for m in metadata if m["key"] == "dc.identifier.uri" and "pdf" in m["value"].lower()),
                    ""
                )
                results.append({
                    "title": title, "link": link,
                    "download": pdf_link, "source": "DOAB"
                })
            return results
    except Exception as e:
        logger.error(f"DOAB error: {e}")
        return []


async def fetch_all_books(topic: str) -> list:
    results = await asyncio.gather(
        search_gutenberg(topic),
        search_open_library(topic),
        search_internet_archive(topic),
        search_doab(topic),
        return_exceptions=True
    )
    all_books = []
    for r in results:
        if not isinstance(r, Exception):
            all_books.extend(r)
    return all_books


async def download_file(url: str) -> bytes | None:
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and len(resp.content) > 5000:
                return resp.content
        return None
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


async def send_books(update: Update, topic: str, books: list):
    """Try to download and send up to 3 books directly."""
    sent = 0
    for book in books:
        if sent >= 3:
            break
        if not book.get("download"):
            continue
        try:
            file_bytes = await asyncio.wait_for(download_file(book["download"]), timeout=20)
            if file_bytes:
                ext = "epub" if "epub" in book["download"] else "pdf"
                safe_title = book["title"][:50]
                await update.message.reply_document(
                    document=io.BytesIO(file_bytes),
                    filename=f"{safe_title}.{ext}",
                    caption=f"📗 *{book['title']}*\n_Source: {book['source']}_",
                    parse_mode="Markdown"
                )
                sent += 1
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Send book error: {e}")
            continue

    if sent == 0:
        # Send links instead
        links_msg = f"📚 *Free {topic.title()} Books:*\n\n"
        for book in books[:5]:
            links_msg += f"• [{book['title']}]({book['link']}) _{book['source']}_\n"
        await update.message.reply_text(links_msg, parse_mode="Markdown")
    
    return sent


# ─── COMMAND HANDLERS ─────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *MindVault* — your free tech school on Telegram!\n\n"
        "Here's how to use me:\n\n"
        "❓ `/what python` — What is Python + roadmap + free books\n"
        "📚 `/learn javascript` — Deep lesson on any tech topic\n"
        "🗺 `/roadmap cybersecurity` — Phase-by-phase learning path\n"
        "📖 `/books machine learning` — Get free downloadable books\n"
        "💬 `/ask how does DNS work?` — Ask me anything tech\n\n"
        "Topics I cover:\n"
        "💻 Programming • 🔐 Cybersecurity • 📊 Data Science\n"
        "🤖 AI/ML • 🎨 UI/UX • ☁️ Cloud • ⛓ Blockchain • 🛠 DevOps\n\n"
        "_Type /help to see all commands_",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *MindVault Commands*\n\n"
        "`/what <topic>` — Explains a topic + roadmap + free books\n"
        "`/learn <topic>` — In-depth lesson on a topic\n"
        "`/roadmap <topic>` — Full phase-based learning path\n"
        "`/books <topic>` — Download free books on a topic\n"
        "`/ask <question>` — Ask any tech question\n"
        "`/topics` — See all topics I can teach\n\n"
        "*Examples:*\n"
        "`/what is react`\n"
        "`/learn docker`\n"
        "`/roadmap data science`\n"
        "`/books python programming`\n"
        "`/ask what is the difference between TCP and UDP`",
        parse_mode="Markdown"
    )


async def topics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 *Topics MindVault Can Teach:*\n\n"
        "💻 *Programming*\n"
        "Python, JavaScript, Java, C++, Rust, Go, Swift, Kotlin\n\n"
        "🌐 *Web Development*\n"
        "HTML/CSS, React, Node.js, Django, FastAPI, Next.js\n\n"
        "🔐 *Cybersecurity*\n"
        "Ethical Hacking, Pen Testing, Networking, Linux Security\n\n"
        "📊 *Data Science & AI*\n"
        "Machine Learning, Deep Learning, Data Analysis, NLP\n\n"
        "🎨 *Design*\n"
        "UI/UX, Figma, Product Design, User Research\n\n"
        "☁️ *Cloud & DevOps*\n"
        "AWS, Docker, Kubernetes, CI/CD, Linux\n\n"
        "⛓ *Blockchain & Web3*\n"
        "Solidity, Smart Contracts, DeFi, NFTs\n\n"
        "📱 *Mobile Development*\n"
        "Flutter, React Native, Android, iOS",
        parse_mode="Markdown"
    )


async def what_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).strip()
    if not topic:
        await update.message.reply_text("Usage: `/what python` or `/what is machine learning`", parse_mode="Markdown")
        return

    # Clean "is" from "what is python"
    topic = topic.lower().replace("is ", "").strip()

    await update.message.reply_text(f"🔍 Looking up *{topic.title()}*...", parse_mode="Markdown")

    try:
        # Explanation
        explanation = ask_groq(
            f"""You are MindVault, a tech education assistant. Explain what "{topic}" is to a complete beginner.

Cover:
1. What it is in simple terms (2-3 sentences)
2. What it is used for (real world examples)
3. Why someone should learn it
4. Who uses it (companies, roles)

Be clear, engaging, and encouraging. Max 250 words.""",
            max_tokens=500
        )
        await safe_reply(update, f"🧠 *What is {topic.title()}?*\n\n{explanation}")

        # Roadmap
        await update.message.reply_text(f"🗺 Generating your *{topic.title()}* roadmap...", parse_mode="Markdown")
        roadmap = ask_groq(
            f"""Create a phase-based learning roadmap for someone who wants to learn "{topic}" from scratch.

Structure it exactly like this:

🟢 *BEGINNER PHASE* (Month 1-2)
• Topic 1
• Topic 2
• Topic 3
• Project idea to build

🟡 *INTERMEDIATE PHASE* (Month 3-4)
• Topic 1
• Topic 2
• Topic 3
• Project idea to build

🔴 *ADVANCED PHASE* (Month 5-6)
• Topic 1
• Topic 2
• Topic 3
• Project idea to build

🏆 *CAREER READY*
• Job roles you can apply for
• Portfolio advice

Keep it practical and achievable. Max 350 words.""",
            max_tokens=700
        )
        await safe_reply(update, f"🗺 *{topic.title()} Learning Roadmap*\n\n{roadmap}")

        # Books
        await update.message.reply_text(f"📚 Fetching free *{topic.title()}* books for you...")
        books = await fetch_all_books(topic)
        if books:
            await send_books(update, topic, books)
        else:
            await update.message.reply_text("📄 No downloadable books found right now. Try /books with a broader term.")

    except Exception as e:
        logger.error(f"/what error: {e}")
        await update.message.reply_text(f"⚠️ Error: {str(e)[:100]}")


async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).strip()
    if not topic:
        await update.message.reply_text("Usage: `/learn javascript`", parse_mode="Markdown")
        return

    await update.message.reply_text(f"📚 Preparing your lesson on *{topic.title()}*...", parse_mode="Markdown")

    try:
        lesson = ask_groq(
            f"""You are MindVault, an expert tech teacher. Teach "{topic}" as a structured lesson.

Structure:
1. 📌 *Introduction* — What it is and why it matters
2. 🔑 *Core Concepts* — The key things to understand (explain each clearly)
3. 💻 *Practical Example* — A simple real code example or use case
4. ✅ *Key Takeaways* — 3-5 bullet points to remember
5. 🚀 *Next Steps* — What to learn next

Use simple language. Be encouraging. Max 400 words.""",
            max_tokens=800
        )
        await safe_reply(update,
            f"📚 *Lesson: {topic.title()}*\n\n{lesson}\n\n"
            f"_Want the full roadmap? Type /roadmap {topic}_"
        )
    except Exception as e:
        logger.error(f"/learn error: {e}")
        await update.message.reply_text(f"⚠️ Error: {str(e)[:100]}")


async def roadmap_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).strip()
    if not topic:
        await update.message.reply_text("Usage: `/roadmap data science`", parse_mode="Markdown")
        return

    await update.message.reply_text(f"🗺 Building your *{topic.title()}* roadmap...", parse_mode="Markdown")

    try:
        roadmap = ask_groq(
            f"""Create a detailed phase-based learning roadmap for "{topic}" from absolute beginner to job-ready.

Structure exactly like this:

🟢 *BEGINNER PHASE* (Month 1-2)
• Topic 1
• Topic 2
• Topic 3
• 🔨 Project: [project idea]
• 📚 Resource: [free resource to use]

🟡 *INTERMEDIATE PHASE* (Month 3-4)
• Topic 1
• Topic 2
• Topic 3
• 🔨 Project: [project idea]
• 📚 Resource: [free resource to use]

🔴 *ADVANCED PHASE* (Month 5-6)
• Topic 1
• Topic 2
• Topic 3
• 🔨 Project: [project idea]
• 📚 Resource: [free resource to use]

🏆 *CAREER READY*
• Job titles you can apply for
• Skills to highlight on CV
• Portfolio tips

Be specific and practical. Max 400 words.""",
            max_tokens=800
        )
        await safe_reply(update,
            f"🗺 *{topic.title()} Roadmap*\n\n{roadmap}\n\n"
            f"_Get free books on this topic: /books {topic}_"
        )
    except Exception as e:
        logger.error(f"/roadmap error: {e}")
        await update.message.reply_text(f"⚠️ Error: {str(e)[:100]}")


async def books_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).strip()
    if not topic:
        await update.message.reply_text("Usage: `/books python programming`", parse_mode="Markdown")
        return

    await update.message.reply_text(
        f"📖 Searching free *{topic.title()}* books across 4 libraries...",
        parse_mode="Markdown"
    )

    try:
        books = await fetch_all_books(topic)
        if not books:
            await update.message.reply_text(
                f"😔 No books found for *{topic.title()}* right now.\n"
                f"Try a broader search like `/books programming` or `/books python`",
                parse_mode="Markdown"
            )
            return

        await update.message.reply_text(
            f"✅ Found *{len(books)}* books! Sending the best ones now...",
            parse_mode="Markdown"
        )
        await send_books(update, topic, books)

    except Exception as e:
        logger.error(f"/books error: {e}")
        await update.message.reply_text("⚠️ Something went wrong. Please try again.")


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args).strip()
    if not question:
        await update.message.reply_text(
            "Usage: `/ask how does DNS work?`", parse_mode="Markdown"
        )
        return

    await update.message.reply_text("💭 Thinking...", parse_mode="Markdown")

    try:
        answer = ask_groq(
            f"""You are MindVault, an expert tech mentor. Answer this question clearly:

"{question}"

Rules:
- Use simple, beginner-friendly language
- Give a direct answer first, then explain
- Use examples or analogies where helpful
- If it's a code question, include a simple code snippet
- End with a tip or what to explore next
- Max 300 words""",
            max_tokens=600
        )
        await safe_reply(update, f"💡 *{question}*\n\n{answer}")
    except Exception as e:
        logger.error(f"/ask error: {e}")
        await update.message.reply_text(f"⚠️ Error: {str(e)[:100]}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text — guide them to use commands."""
    text = update.message.text.strip()
    if not text:
        return

    # Try to detect what they want and suggest the right command
    suggestion = ask_groq(
        f"""A user sent this message to MindVault, a tech learning bot: "{text}"

Based on their message, suggest the exact command they should use from these options:
/what, /learn, /roadmap, /books, /ask

Reply with ONLY one short sentence like:
"Try: /what python" or "Try: /ask {text}" or "Try: /roadmap javascript"

Pick the most fitting command. Keep it very short.""",
        max_tokens=50
    )

    await update.message.reply_text(
        f"👋 Use a command to get started!\n\n{suggestion}\n\n"
        f"Type /help to see all available commands.",
        parse_mode="Markdown"
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
    app.add_handler(CommandHandler("topics", topics_command))
    app.add_handler(CommandHandler("what", what_command))
    app.add_handler(CommandHandler("learn", learn_command))
    app.add_handler(CommandHandler("roadmap", roadmap_command))
    app.add_handler(CommandHandler("books", books_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("MindVault bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
