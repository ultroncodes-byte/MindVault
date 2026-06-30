import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import contextmanager

import bcrypt
import jwt
import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from groq import Groq
import uvicorn

# ─── LOGGING ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── ENVIRONMENT ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-this-in-production")
if JWT_SECRET_KEY == "change-this-in-production":
    logger.warning("Using default JWT_SECRET_KEY – set a secure one in production")

groq_client = Groq(api_key=GROQ_API_KEY)

# ─── DATABASE ─────────────────────────────────────────────────────────────
DB_PATH = "mindvault.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                name TEXT,
                daily_hours INTEGER DEFAULT 1,
                goal TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS roadmaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                goal TEXT NOT NULL,
                roadmap_text TEXT NOT NULL,
                phases_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                phase TEXT,
                completed BOOLEAN DEFAULT 0,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, topic)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                score INTEGER,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
    logger.info("Database initialized.")

# ─── AUTH ─────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm="HS256")

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    payload = decode_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, email, name, daily_hours, goal FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = ""

class RoadmapRequest(BaseModel):
    goal: str
    daily_hours: Optional[int] = 1

class TopicComplete(BaseModel):
    topic: str
    phase: Optional[str] = None

# ─── GROQ HELPER ──────────────────────────────────────────────────────────
def ask_groq(prompt: str, max_tokens: int = 800) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# ─── BOOK SEARCH ──────────────────────────────────────────────────────────
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
    import asyncio
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

# ─── FRONTEND HTML ──────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MindVault 2.0 – Free AI Tech School</title>
    <style>
        /* ─── RESET & BASE ─────────────────────────────────────────────── */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f8fafc; color: #0f172a; line-height: 1.6; }
        a { color: #2563eb; text-decoration: none; }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

        /* ─── BUTTONS ──────────────────────────────────────────────────── */
        .btn { display: inline-block; padding: 10px 24px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn-primary { background: #2563eb; color: white; }
        .btn-primary:hover { background: #1d4ed8; transform: translateY(-1px); }
        .btn-outline { background: transparent; border: 2px solid #2563eb; color: #2563eb; }
        .btn-outline:hover { background: #2563eb; color: white; }
        .btn-success { background: #16a34a; color: white; }
        .btn-success:hover { background: #15803d; }
        .btn-danger { background: #dc2626; color: white; }
        .btn-danger:hover { background: #b91c1c; }

        /* ─── HEADER ───────────────────────────────────────────────────── */
        header { background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 16px 0; position: sticky; top: 0; z-index: 100; }
        .header-flex { display: flex; justify-content: space-between; align-items: center; }
        .logo { font-size: 1.5rem; font-weight: 800; color: #2563eb; }
        .logo span { color: #0f172a; }
        .nav-buttons { display: flex; gap: 12px; align-items: center; }

        /* ─── HERO ─────────────────────────────────────────────────────── */
        .hero { padding: 80px 0 60px; text-align: center; }
        .hero h1 { font-size: 3.5rem; font-weight: 800; line-height: 1.2; margin-bottom: 16px; }
        .hero h1 .highlight { background: linear-gradient(135deg, #2563eb, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero p { font-size: 1.25rem; color: #475569; max-width: 640px; margin: 0 auto 32px; }
        .hero-actions { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }

        /* ─── FEATURES ────────────────────────────────────────────────── */
        .features { padding: 60px 0; background: white; }
        .features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 32px; margin-top: 40px; }
        .feature-card { padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; transition: 0.2s; }
        .feature-card:hover { border-color: #2563eb; box-shadow: 0 4px 12px rgba(37,99,235,0.1); }
        .feature-card .icon { font-size: 2.5rem; margin-bottom: 12px; }
        .feature-card h3 { font-size: 1.25rem; margin-bottom: 8px; }
        .feature-card p { color: #64748b; }

        /* ─── MODAL ────────────────────────────────────────────────────── */
        .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(15,23,42,0.6); backdrop-filter: blur(4px); justify-content: center; align-items: center; z-index: 200; }
        .modal-overlay.active { display: flex; }
        .modal { background: white; padding: 40px; border-radius: 16px; max-width: 420px; width: 90%; max-height: 90vh; overflow-y: auto; }
        .modal h2 { margin-bottom: 24px; font-size: 1.75rem; }
        .modal .form-group { margin-bottom: 16px; }
        .modal label { display: block; font-weight: 600; margin-bottom: 4px; }
        .modal input { width: 100%; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; }
        .modal input:focus { outline: 2px solid #2563eb; border-color: transparent; }
        .modal .error { color: #dc2626; font-size: 0.875rem; margin-top: 8px; display: none; }
        .modal .switch { margin-top: 16px; text-align: center; font-size: 0.9rem; }
        .modal .switch a { font-weight: 600; cursor: pointer; }

        /* ─── DASHBOARD ────────────────────────────────────────────────── */
        #dashboard { display: none; padding: 40px 0; }
        .dashboard-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; flex-wrap: wrap; gap: 16px; }
        .dashboard-header h2 { font-size: 2rem; }
        .tabs { display: flex; gap: 8px; flex-wrap: wrap; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 24px; }
        .tab-btn { padding: 8px 20px; border-radius: 20px; border: none; background: transparent; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .tab-btn.active { background: #2563eb; color: white; }
        .tab-btn:hover:not(.active) { background: #e2e8f0; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* ─── CARDS & UTILITY ──────────────────────────────────────────── */
        .card { background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
        .card h3 { margin-bottom: 12px; }
        .input-group { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
        .input-group input, .input-group textarea { flex: 1; min-width: 200px; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; }
        .input-group textarea { min-height: 80px; resize: vertical; }
        .result-box { background: #f1f5f9; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
        .book-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e2e8f0; }
        .book-item:last-child { border-bottom: none; }
        .book-item .title { font-weight: 600; }
        .book-item .source { font-size: 0.875rem; color: #64748b; }
        .book-item .btn { padding: 4px 12px; font-size: 0.875rem; }

        .progress-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0; }
        .progress-item .done { color: #16a34a; font-weight: 600; }

        /* ─── RESPONSIVE ────────────────────────────────────────────────── */
        @media (max-width: 640px) {
            .hero h1 { font-size: 2.5rem; }
            .header-flex { flex-direction: column; gap: 12px; }
            .nav-buttons { width: 100%; justify-content: center; }
        }
    </style>
</head>
<body>

    <!-- ─── HEADER ────────────────────────────────────────────────────── -->
    <header>
        <div class="container header-flex">
            <div class="logo">Mind<span>Vault</span></div>
            <div class="nav-buttons" id="navButtons">
                <button class="btn btn-outline" onclick="openModal('login')">Login</button>
                <button class="btn btn-primary" onclick="openModal('register')">Get Started</button>
            </div>
            <div class="nav-buttons" id="navUser" style="display:none;">
                <span id="userEmail" style="font-weight:600;"></span>
                <button class="btn btn-danger" onclick="logout()">Logout</button>
            </div>
        </div>
    </header>

    <!-- ─── HERO / LANDING ────────────────────────────────────────────── -->
    <section class="hero" id="landing">
        <div class="container">
            <h1>Your Free AI-Powered<br><span class="highlight">Tech School</span></h1>
            <p>MindVault gives you personalized roadmaps, free books, quizzes, and expert answers – all powered by Groq LLaMA 3.3-70B.</p>
            <div class="hero-actions">
                <button class="btn btn-primary" onclick="openModal('register')">Start Learning Now</button>
                <button class="btn btn-outline" onclick="openModal('login')">Sign In</button>
            </div>
        </div>
    </section>

    <!-- ─── FEATURES ───────────────────────────────────────────────────── -->
    <section class="features" id="features">
        <div class="container">
            <h2 style="text-align:center; font-size:2rem;">Everything You Need to Succeed</h2>
            <div class="features-grid">
                <div class="feature-card"><div class="icon">🧠</div><h3>Personalized Roadmaps</h3><p>Tell us your goal and get a step‑by‑step learning plan.</p></div>
                <div class="feature-card"><div class="icon">📚</div><h3>Free Books</h3><p>Search and download free tech books from multiple libraries.</p></div>
                <div class="feature-card"><div class="icon">📝</div><h3>AI Quizzes</h3><p>Test your knowledge with instant 5‑question quizzes.</p></div>
                <div class="feature-card"><div class="icon">💡</div><h3>Ask Anything</h3><p>Get clear, beginner‑friendly answers to any tech question.</p></div>
                <div class="feature-card"><div class="icon">📖</div><h3>Structured Lessons</h3><p>Learn any topic with an expert‑crafted lesson.</p></div>
                <div class="feature-card"><div class="icon">✅</div><h3>Track Progress</h3><p>Mark topics complete and watch your skills grow.</p></div>
            </div>
        </div>
    </section>

    <!-- ─── MODALS ────────────────────────────────────────────────────── -->
    <!-- Login Modal -->
    <div class="modal-overlay" id="loginModal">
        <div class="modal">
            <h2>Welcome Back</h2>
            <form id="loginForm" onsubmit="login(event)">
                <div class="form-group">
                    <label>Email</label>
                    <input type="email" id="loginEmail" required placeholder="you@example.com" />
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="loginPassword" required placeholder="••••••••" />
                </div>
                <div class="error" id="loginError">Invalid credentials</div>
                <button type="submit" class="btn btn-primary" style="width:100%;">Sign In</button>
                <div class="switch">Don't have an account? <a onclick="switchModal('register')">Register</a></div>
            </form>
        </div>
    </div>

    <!-- Register Modal -->
    <div class="modal-overlay" id="registerModal">
        <div class="modal">
            <h2>Create Your Account</h2>
            <form id="registerForm" onsubmit="register(event)">
                <div class="form-group">
                    <label>Full Name</label>
                    <input type="text" id="registerName" placeholder="Jane Doe" />
                </div>
                <div class="form-group">
                    <label>Email</label>
                    <input type="email" id="registerEmail" required placeholder="you@example.com" />
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="registerPassword" required placeholder="Min 8 characters" minlength="8" />
                </div>
                <div class="error" id="registerError">Registration failed</div>
                <button type="submit" class="btn btn-primary" style="width:100%;">Create Account</button>
                <div class="switch">Already have an account? <a onclick="switchModal('login')">Sign In</a></div>
            </form>
        </div>
    </div>

    <!-- ─── DASHBOARD ──────────────────────────────────────────────────── -->
    <section id="dashboard">
        <div class="container">
            <div class="dashboard-header">
                <h2>📊 Dashboard</h2>
                <span id="dashUser" style="font-size:1.1rem; color:#475569;"></span>
            </div>

            <!-- Tabs -->
            <div class="tabs">
                <button class="tab-btn active" data-tab="roadmap">🗺 Roadmap</button>
                <button class="tab-btn" data-tab="books">📚 Books</button>
                <button class="tab-btn" data-tab="quiz">📝 Quiz</button>
                <button class="tab-btn" data-tab="learn">📖 Learn</button>
                <button class="tab-btn" data-tab="ask">💬 Ask</button>
                <button class="tab-btn" data-tab="progress">✅ Progress</button>
            </div>

            <!-- Roadmap Tab -->
            <div class="tab-content active" id="tab-roadmap">
                <div class="card">
                    <h3>Generate Your Roadmap</h3>
                    <div class="input-group">
                        <input type="text" id="roadmapGoal" placeholder="e.g. Become a cybersecurity expert" />
                        <button class="btn btn-primary" onclick="generateRoadmap()">Generate</button>
                    </div>
                    <div id="roadmapResult" class="result-box" style="display:none;"></div>
                </div>
                <div class="card">
                    <h3>Your Last Roadmap</h3>
                    <div id="savedRoadmap" class="result-box">No roadmap saved yet.</div>
                </div>
            </div>

            <!-- Books Tab -->
            <div class="tab-content" id="tab-books">
                <div class="card">
                    <h3>Search Free Books</h3>
                    <div class="input-group">
                        <input type="text" id="bookTopic" placeholder="e.g. python programming" />
                        <button class="btn btn-primary" onclick="searchBooks()">Search</button>
                    </div>
                    <div id="bookResults"></div>
                </div>
            </div>

            <!-- Quiz Tab -->
            <div class="tab-content" id="tab-quiz">
                <div class="card">
                    <h3>Generate a Quiz</h3>
                    <div class="input-group">
                        <input type="text" id="quizTopic" placeholder="e.g. React hooks" />
                        <button class="btn btn-primary" onclick="generateQuiz()">Generate Quiz</button>
                    </div>
                    <div id="quizResult" class="result-box" style="display:none;"></div>
                </div>
            </div>

            <!-- Learn Tab -->
            <div class="tab-content" id="tab-learn">
                <div class="card">
                    <h3>Learn a Topic</h3>
                    <div class="input-group">
                        <input type="text" id="learnTopic" placeholder="e.g. Docker" />
                        <button class="btn btn-primary" onclick="learnTopic()">Get Lesson</button>
                    </div>
                    <div id="learnResult" class="result-box" style="display:none;"></div>
                </div>
            </div>

            <!-- Ask Tab -->
            <div class="tab-content" id="tab-ask">
                <div class="card">
                    <h3>Ask a Tech Question</h3>
                    <div class="input-group">
                        <textarea id="askQuestion" placeholder="e.g. How does DNS work?"></textarea>
                        <button class="btn btn-primary" onclick="askQuestion()">Ask</button>
                    </div>
                    <div id="askResult" class="result-box" style="display:none;"></div>
                </div>
            </div>

            <!-- Progress Tab -->
            <div class="tab-content" id="tab-progress">
                <div class="card">
                    <h3>Your Progress</h3>
                    <div id="progressList">Loading...</div>
                </div>
                <div class="card">
                    <h3>Mark Topic Complete</h3>
                    <div class="input-group">
                        <input type="text" id="progressTopic" placeholder="Topic name" />
                        <input type="text" id="progressPhase" placeholder="Phase (optional)" />
                        <button class="btn btn-success" onclick="markComplete()">Mark Complete</button>
                    </div>
                </div>
            </div>

        </div>
    </section>

    <!-- ─── JAVASCRIPT ────────────────────────────────────────────────── -->
    <script>
        // ─── STATE ────────────────────────────────────────────────────
        let token = localStorage.getItem('token');
        let currentUser = null;

        // ─── DOM REFS ─────────────────────────────────────────────────
        const landing = document.getElementById('landing');
        const features = document.getElementById('features');
        const dashboard = document.getElementById('dashboard');
        const navButtons = document.getElementById('navButtons');
        const navUser = document.getElementById('navUser');
        const userEmailSpan = document.getElementById('userEmail');
        const dashUserSpan = document.getElementById('dashUser');

        // ─── INIT ─────────────────────────────────────────────────────
        function init() {
            if (token) {
                fetchUser();
            } else {
                showLanding();
            }
            // Tab switching
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    document.getElementById('tab-' + this.dataset.tab).classList.add('active');
                });
            });
        }

        // ─── UI HELPERS ───────────────────────────────────────────────
        function showLanding() {
            landing.style.display = 'block';
            features.style.display = 'block';
            dashboard.style.display = 'none';
            navButtons.style.display = 'flex';
            navUser.style.display = 'none';
        }

        function showDashboard(user) {
            landing.style.display = 'none';
            features.style.display = 'none';
            dashboard.style.display = 'block';
            navButtons.style.display = 'none';
            navUser.style.display = 'flex';
            userEmailSpan.textContent = user.email;
            dashUserSpan.textContent = '👤 ' + user.email;
            // Load saved roadmap
            loadSavedRoadmap();
            // Load progress
            loadProgress();
        }

        function openModal(type) {
            document.getElementById(type + 'Modal').classList.add('active');
            document.getElementById(type + 'Error').style.display = 'none';
        }

        function closeModal(type) {
            document.getElementById(type + 'Modal').classList.remove('active');
        }

        function switchModal(type) {
            closeModal('login');
            closeModal('register');
            openModal(type);
        }

        // ─── AUTH ──────────────────────────────────────────────────────
        async function register(e) {
            e.preventDefault();
            const name = document.getElementById('registerName').value;
            const email = document.getElementById('registerEmail').value;
            const password = document.getElementById('registerPassword').value;
            const errorEl = document.getElementById('registerError');
            try {
                const res = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password, name })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Registration failed');
                // Auto-login after registration
                const loginRes = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ username: email, password })
                });
                const loginData = await loginRes.json();
                if (!loginRes.ok) throw new Error(loginData.detail || 'Login failed');
                token = loginData.access_token;
                localStorage.setItem('token', token);
                closeModal('register');
                fetchUser();
            } catch (err) {
                errorEl.textContent = err.message;
                errorEl.style.display = 'block';
            }
        }

        async function login(e) {
            e.preventDefault();
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            const errorEl = document.getElementById('loginError');
            try {
                const res = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ username: email, password })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Login failed');
                token = data.access_token;
                localStorage.setItem('token', token);
                closeModal('login');
                fetchUser();
            } catch (err) {
                errorEl.textContent = err.message;
                errorEl.style.display = 'block';
            }
        }

        async function fetchUser() {
            try {
                const res = await fetch('/me', {
                    headers: { 'Authorization': 'Bearer ' + token }
                });
                if (!res.ok) throw new Error('Session expired');
                const user = await res.json();
                currentUser = user;
                showDashboard(user);
            } catch (err) {
                localStorage.removeItem('token');
                token = null;
                showLanding();
            }
        }

        function logout() {
            localStorage.removeItem('token');
            token = null;
            showLanding();
        }

        // ─── API CALLS (protected) ─────────────────────────────────────
        async function apiCall(method, url, body) {
            const options = {
                method,
                headers: {
                    'Authorization': 'Bearer ' + token,
                    'Content-Type': 'application/json'
                }
            };
            if (body) options.body = JSON.stringify(body);
            const res = await fetch(url, options);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Request failed');
            }
            return res.json();
        }

        // ─── ROADMAP ──────────────────────────────────────────────────
        async function generateRoadmap() {
            const goal = document.getElementById('roadmapGoal').value.trim();
            if (!goal) return alert('Please enter a learning goal.');
            const resultDiv = document.getElementById('roadmapResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Generating...';
            try {
                const data = await apiCall('POST', '/roadmap', { goal });
                resultDiv.textContent = data.roadmap;
                // Also update saved roadmap
                document.getElementById('savedRoadmap').textContent = data.roadmap;
            } catch (err) {
                resultDiv.textContent = 'Error: ' + err.message;
            }
        }

        async function loadSavedRoadmap() {
            try {
                const data = await apiCall('GET', '/roadmap');
                document.getElementById('savedRoadmap').textContent = data.roadmap_text;
            } catch (err) {
                // ignore if none
            }
        }

        // ─── BOOKS ──────────────────────────────────────────────────────
        async function searchBooks() {
            const topic = document.getElementById('bookTopic').value.trim();
            if (!topic) return alert('Enter a topic.');
            const container = document.getElementById('bookResults');
            container.innerHTML = 'Searching...';
            try {
                const data = await apiCall('GET', '/books/' + encodeURIComponent(topic));
                if (!data.books || data.books.length === 0) {
                    container.innerHTML = '<p>No books found.</p>';
                    return;
                }
                let html = '';
                data.books.forEach(b => {
                    html += `<div class="book-item">
                                <span><span class="title">${b.title}</span> <span class="source">(${b.source})</span></span>
                                <a href="${b.link}" target="_blank" class="btn btn-outline">View</a>
                            </div>`;
                });
                container.innerHTML = html;
            } catch (err) {
                container.innerHTML = '<p>Error: ' + err.message + '</p>';
            }
        }

        // ─── QUIZ ──────────────────────────────────────────────────────
        async function generateQuiz() {
            const topic = document.getElementById('quizTopic').value.trim();
            if (!topic) return alert('Enter a topic.');
            const resultDiv = document.getElementById('quizResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Generating quiz...';
            try {
                const data = await apiCall('POST', '/quiz?topic=' + encodeURIComponent(topic));
                let html = '';
                if (data.quiz && Array.isArray(data.quiz)) {
                    data.quiz.forEach((q, i) => {
                        html += `<p><strong>${i+1}. ${q.question}</strong></p>
                                 <ul style="list-style:none; margin-bottom:12px;">`;
                        q.options.forEach(opt => {
                            html += `<li style="padding:2px 0;">${opt}</li>`;
                        });
                        html += `<li><em>Answer: ${q.answer}</em></li></ul>`;
                    });
                } else {
                    html = 'Could not parse quiz: ' + JSON.stringify(data);
                }
                resultDiv.innerHTML = html;
            } catch (err) {
                resultDiv.textContent = 'Error: ' + err.message;
            }
        }

        // ─── LEARN ──────────────────────────────────────────────────────
        async function learnTopic() {
            const topic = document.getElementById('learnTopic').value.trim();
            if (!topic) return alert('Enter a topic.');
            const resultDiv = document.getElementById('learnResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Generating lesson...';
            try {
                const data = await apiCall('POST', '/learn?topic=' + encodeURIComponent(topic));
                resultDiv.textContent = data.lesson;
            } catch (err) {
                resultDiv.textContent = 'Error: ' + err.message;
            }
        }

        // ─── ASK ──────────────────────────────────────────────────────
        async function askQuestion() {
            const question = document.getElementById('askQuestion').value.trim();
            if (!question) return alert('Enter a question.');
            const resultDiv = document.getElementById('askResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Thinking...';
            try {
                const data = await apiCall('POST', '/ask?question=' + encodeURIComponent(question));
                resultDiv.textContent = data.answer;
            } catch (err) {
                resultDiv.textContent = 'Error: ' + err.message;
            }
        }

        // ─── PROGRESS ──────────────────────────────────────────────────
        async function loadProgress() {
            const container = document.getElementById('progressList');
            try {
                const data = await apiCall('GET', '/progress');
                if (!data || data.length === 0) {
                    container.innerHTML = '<p>No topics completed yet.</p>';
                    return;
                }
                let html = '';
                data.forEach(p => {
                    const status = p.completed ? '✅ Done' : '⏳ Pending';
                    html += `<div class="progress-item"><span>${p.topic} ${p.phase ? '('+p.phase+')' : ''}</span><span class="${p.completed ? 'done' : ''}">${status}</span></div>`;
                });
                container.innerHTML = html;
            } catch (err) {
                container.innerHTML = '<p>Error loading progress.</p>';
            }
        }

        async function markComplete() {
            const topic = document.getElementById('progressTopic').value.trim();
            const phase = document.getElementById('progressPhase').value.trim();
            if (!topic) return alert('Enter a topic.');
            try {
                await apiCall('POST', '/progress', { topic, phase });
                alert('Marked as complete!');
                loadProgress();
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }

        // ─── CLOSE MODALS ON OVERLAY CLICK ─────────────────────────────
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', function(e) {
                if (e.target === this) this.classList.remove('active');
            });
        });

        // ─── START ─────────────────────────────────────────────────────
        init();
    </script>

</body>
</html>
"""

# ─── FASTAPI APP ──────────────────────────────────────────────────────────
app = FastAPI(title="MindVault 2.0", version="2.0")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTML_TEMPLATE

# ─── API ENDPOINTS ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register")
def register(user: UserRegister):
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (user.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = hash_password(user.password)
        conn.execute(
            "INSERT INTO users (email, hashed_password, name) VALUES (?, ?, ?)",
            (user.email, hashed, user.name or "")
        )
        conn.commit()
        new_user = conn.execute("SELECT id, email, name FROM users WHERE email = ?", (user.email,)).fetchone()
    return {"message": "User created", "user": dict(new_user)}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, email, hashed_password FROM users WHERE email = ?",
            (form_data.username,)
        ).fetchone()
        if not user or not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user["id"])})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.post("/roadmap")
def generate_roadmap(req: RoadmapRequest, current_user: dict = Depends(get_current_user)):
    prompt = f"""Create a phase-based learning roadmap for "{req.goal}" from beginner to job-ready.
Structure with phases (Beginner, Intermediate, Advanced) including topics, projects, and free resources.
Make it detailed and practical. Max 400 words."""
    roadmap_text = ask_groq(prompt, max_tokens=800)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO roadmaps (user_id, goal, roadmap_text) VALUES (?, ?, ?)",
            (current_user["id"], req.goal, roadmap_text)
        )
        conn.commit()
    return {"goal": req.goal, "roadmap": roadmap_text}

@app.get("/roadmap")
def get_roadmap(current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        roadmap = conn.execute(
            "SELECT goal, roadmap_text, created_at FROM roadmaps WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (current_user["id"],)
        ).fetchone()
        if not roadmap:
            raise HTTPException(status_code=404, detail="No roadmap found for this user")
    return dict(roadmap)

@app.post("/progress")
def mark_topic_complete(topic: TopicComplete, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO progress (user_id, topic, phase, completed, completed_at)
               VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, topic) DO UPDATE SET completed=1, completed_at=CURRENT_TIMESTAMP""",
            (current_user["id"], topic.topic, topic.phase)
        )
        conn.commit()
    return {"message": f"Topic '{topic.topic}' marked as complete"}

@app.get("/progress")
def get_progress(current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        progress = conn.execute(
            "SELECT topic, phase, completed, completed_at FROM progress WHERE user_id = ?",
            (current_user["id"],)
        ).fetchall()
    return [dict(row) for row in progress]

@app.get("/books/{topic}")
async def get_books(topic: str, current_user: dict = Depends(get_current_user)):
    books = await fetch_all_books(topic)
    if not books:
        return {"message": f"No books found for '{topic}'", "books": []}
    return {"books": books}

@app.post("/quiz")
def generate_quiz(topic: str, current_user: dict = Depends(get_current_user)):
    prompt = f"""Generate 5 multiple-choice questions about '{topic}' to test understanding.
For each question, provide 4 options (A, B, C, D) and indicate the correct answer.
Format the output as a JSON array with objects: {{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A"}}.
Only return valid JSON."""
    response = ask_groq(prompt, max_tokens=600)
    try:
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            quiz = json.loads(json_match.group())
        else:
            quiz = json.loads(response)
    except:
        quiz = {"error": "Could not parse quiz", "raw": response}
    return {"topic": topic, "quiz": quiz}

@app.post("/learn")
def learn_topic(topic: str, current_user: dict = Depends(get_current_user)):
    prompt = f"""You are MindVault, an expert tech teacher. Teach "{topic}" as a structured lesson.

Structure:
1. 📌 Introduction — What it is and why it matters
2. 🔑 Core Concepts — Key things to understand
3. 💻 Practical Example — Simple code or use case
4. ✅ Key Takeaways — 3-5 bullet points
5. 🚀 Next Steps — What to learn next

Use simple language. Max 400 words."""
    lesson = ask_groq(prompt, max_tokens=800)
    return {"topic": topic, "lesson": lesson}

@app.post("/ask")
def ask_question(question: str, current_user: dict = Depends(get_current_user)):
    prompt = f"""You are MindVault, an expert tech mentor. Answer this question clearly:

"{question}"

Rules:
- Use simple, beginner-friendly language
- Give a direct answer first, then explain
- Use examples or analogies where helpful
- If code, include a simple snippet
- End with a tip or what to explore next
- Max 300 words"""
    answer = ask_groq(prompt, max_tokens=600)
    return {"question": question, "answer": answer}

# ─── RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
