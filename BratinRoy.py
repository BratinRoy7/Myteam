
# discovery_app.py
# Streamlit web app: Discovery AI chatbot by Wellsoft Corporation
# Single-file implementation with guest chat and email/password auth.
# Chat history for signed-in users stored in PostgreSQL and auto-deleted after 2 weeks.
# API key is read from Streamlit Secrets (OPENAI_API_KEY).

import os
import time
import base64
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor

# Optional: if not installed, add openai to your environment
# pip install openai>=1.0.0
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# App configuration and constants
# -----------------------------
APP_NAME = "Discovery"
COMPANY = "Wellsoft Corporation"
CEO_CHAIRMAN = "Bratin Roy"
FOUNDER = "Bratin Roy"
FOUNDED_DATE = "October 12, 2025"
CONTACT_EMAIL = "corporation.wellsoft@gmail.com"
INSTAGRAM = "https://instagram.com/wellsoftcorporation"
WEBSITE = "https://wellsoftcorporation.tiiny.site"
LOGO_URL = "https://i.ibb.co/C5MTvx1V/wellsoft-logo.png"  # Fixed URL

# PostgreSQL connection string
DATABASE_URL = "postgresql://neondb_owner:npg_BzfWOYT0sQM1@ep-cool-wildflower-af5obep7.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require"

# Retention policy
RETENTION_DAYS = 14

# OpenAI model (adjust if needed)
DEFAULT_MODEL = "gpt-4o-mini"

# UI theme
st.set_page_config(
    page_title=f"{APP_NAME} — AI chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Database Utilities
# -----------------------------
def get_db_connection():
    """Create PostgreSQL connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def init_database():
    """Initialize database tables."""
    conn = get_db_connection()
    if conn is None:
        return False
        
    try:
        with conn.cursor() as cur:
            # Create users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            
            # Create chats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    title TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            
            # Create messages table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    attachments TEXT,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_created_at ON chats(created_at)")
            
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False
    finally:
        conn.close()

def purge_old_history(user_id: str, days: int = RETENTION_DAYS):
    """Delete messages and chats older than 'days' for the given user."""
    conn = get_db_connection()
    if conn is None:
        return
        
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with conn.cursor() as cur:
            # Delete old messages first (due to foreign key constraints)
            cur.execute("""
                DELETE FROM messages 
                WHERE chat_id IN (
                    SELECT chat_id FROM chats 
                    WHERE user_id = %s AND created_at < %s
                )
            """, (user_id, cutoff))
            
            # Then delete old chats
            cur.execute("""
                DELETE FROM chats 
                WHERE user_id = %s AND created_at < %s
            """, (user_id, cutoff))
            
        conn.commit()
    except Exception as e:
        st.error(f"Error purging old history: {e}")
    finally:
        conn.close()

def create_user(email: str, password: str) -> Optional[str]:
    """Create a user with a password hash. Returns user_id or None."""
    user_id = str(uuid.uuid4())
    password_hash = base64.urlsafe_b64encode(password.encode()).decode()
    created_at = datetime.utcnow()
    
    conn = get_db_connection()
    if conn is None:
        return None
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (user_id, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
                (user_id, email.lower().strip(), password_hash, created_at)
            )
        conn.commit()
        return user_id
    except psycopg2.IntegrityError:
        return None
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return None
    finally:
        conn.close()

def authenticate_user(email: str, password: str) -> Optional[str]:
    """Verify user credentials. Returns user_id or None."""
    conn = get_db_connection()
    if conn is None:
        return None
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, password_hash FROM users WHERE email = %s",
                (email.lower().strip(),)
            )
            row = cur.fetchone()
            
        if not row:
            return None
            
        user_id, stored_hash = row
        input_hash = base64.urlsafe_b64encode(password.encode()).decode()
        
        if input_hash == stored_hash:
            return user_id
        return None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return None
    finally:
        conn.close()

def create_chat(user_id: str, title: str = "New chat") -> Optional[str]:
    """Create a new chat and return chat_id."""
    chat_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    
    conn = get_db_connection()
    if conn is None:
        return None
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (chat_id, user_id, created_at, title) VALUES (%s, %s, %s, %s)",
                (chat_id, user_id, created_at, title)
            )
        conn.commit()
        return chat_id
    except Exception as e:
        st.error(f"Error creating chat: {e}")
        return None
    finally:
        conn.close()

def save_message(chat_id: str, role: str, content: str, attachments: Optional[str] = None):
    """Save a message to the database."""
    conn = get_db_connection()
    if conn is None:
        return
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (chat_id, role, content, created_at, attachments) VALUES (%s, %s, %s, %s, %s)",
                (chat_id, role, content, datetime.utcnow(), attachments or "")
            )
        conn.commit()
    except Exception as e:
        st.error(f"Error saving message: {e}")
    finally:
        conn.close()

def load_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    """Load all messages for a chat."""
    conn = get_db_connection()
    if conn is None:
        return []
        
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, created_at, attachments FROM messages WHERE chat_id = %s ORDER BY id ASC",
                (chat_id,)
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        st.error(f"Error loading messages: {e}")
        return []
    finally:
        conn.close()

def get_user_chats(user_id: str) -> List[Dict[str, Any]]:
    """Get all chats for a user."""
    conn = get_db_connection()
    if conn is None:
        return []
        
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT chat_id, title, created_at FROM chats WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        st.error(f"Error loading user chats: {e}")
        return []
    finally:
        conn.close()

def get_openai_client() -> Optional[Any]:
    """Create OpenAI client using Streamlit secrets."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to Streamlit secrets.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

def format_attachments(files: List[Dict[str, Any]]) -> str:
    if not files:
        return ""
    names = [f.get("name") for f in files if f.get("name")]
    return ", ".join(names)

def bytes_to_dataurl(file_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(file_bytes).decode()
    return f"data:{mime_type};base64,{b64}"

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    defaults = {
        "route": "loading",   # "loading" -> "home" -> "chat"
        "loading_start": time.time(),
        "guest_mode": True,
        "user_id": None,
        "email": None,
        "chat_id": None,
        "messages": [],       # guest mode in-memory messages
        "attachments": [],    # for current prompt
        "model": DEFAULT_MODEL,
        "tone": "Helpful",
        "temperature": 0.7,
        "max_output_tokens": 800,
        "theme_dark": True,
        "typing": False,
        "preset": "None",
        "client": None,
        "db_initialized": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Initialize database and OpenAI client
if not st.session_state["db_initialized"]:
    if init_database():
        st.session_state["db_initialized"] = True

if st.session_state["client"] is None:
    st.session_state["client"] = get_openai_client()

# -----------------------------
# UI helpers
# -----------------------------
def apply_theme():
    # Streamlit built-in theming is in config; we simulate accents with CSS.
    dark = st.session_state["theme_dark"]
    bg = "#0e1117" if dark else "#ffffff"
    fg = "#e8eaed" if dark else "#111827"
    accent = "#14b8a6"  # teal
    subtle = "#1f2937" if dark else "#f3f4f6"

    st.markdown(f"""
        <style>
        .main {{
            background-color: {bg};
            color: {fg};
        }}
        .discovery-header {{
            display: flex; align-items: center; gap: 12px;
            padding: 8px 0 16px 0; border-bottom: 1px solid {subtle};
        }}
        .discovery-tag {{
            font-size: 12px; color: {fg}; opacity: 0.7;
        }}
        .bubble-user {{
            background: {accent}22; border: 1px solid {accent}55; padding: 12px; border-radius: 12px;
            margin: 8px 0;
        }}
        .bubble-assistant {{
            background: {subtle}; border: 1px solid {accent}22; padding: 12px; border-radius: 12px;
            margin: 8px 0;
        }}
        .typing {{
            display:inline-block; width:6px; height:6px; border-radius:50%; background:{accent};
            animation: blink 1.2s infinite;
        }}
        @keyframes blink {{
            0% {{ opacity: 0.2; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.2; }}
        }}
        .footer {{
            margin-top: 24px; font-size: 12px; opacity: 0.7;
        }}
        .suggestion-button {{
            width: 100%;
            margin: 2px 0;
        }}
        </style>
    """, unsafe_allow_html=True)

def avatar(role: str):
    if role == "user":
        return "🧑‍💻"
    return "🤖"

def title_bar():
    # Header with logo and brand
    cols = st.columns([0.08, 0.92])
    with cols[0]:
        st.image(LOGO_URL, width=48)
    with cols[1]:
        st.markdown(f"""
            <div class="discovery-header">
                <div>
                    <h2 style="margin-bottom:4px">{APP_NAME}</h2>
                    <div class="discovery-tag">Powered by {COMPANY}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def show_typing_indicator():
    st.markdown("""
        <div style="display:flex;align-items:center;gap:6px;">
            <div class="typing"></div><div class="typing"></div><div class="typing"></div>
        </div>
    """, unsafe_allow_html=True)

def render_message(role: str, content: str, attachments: Optional[str] = None):
    bubble_class = "bubble-user" if role == "user" else "bubble-assistant"
    st.markdown(f"""
        <div class="{bubble_class}">
            <div style="font-size:14px;opacity:0.7;margin-bottom:6px;">{avatar(role)} {role.capitalize()}</div>
            <div style="font-size:16px;line-height:1.6;white-space:pre-wrap;">{content}</div>
            {"<div style='margin-top:8px;font-size:12px;opacity:0.7;'>Attachments: "+attachments+"</div>" if attachments else ""}
        </div>
    """, unsafe_allow_html=True)

def prompt_presets() -> Dict[str, str]:
    return {
        "None": "",
        "Brainstorm": "Help me brainstorm creative ideas and organize them into themes.",
        "Summarize": "Summarize the following content concisely with bullet points and action items.",
        "Explain like I'm 5": "Explain the concept simply with analogies and a short example.",
        "Code helper": "You're a senior engineer. Provide clear steps, samples, and caveats.",
    }

def make_system_prompt(tone: str) -> str:
    return f"You are {APP_NAME}, an empathetic, efficient AI assistant. Tone: {tone}. Be concise and precise. Avoid repetition."

# -----------------------------
# Loading screen
# -----------------------------
def screen_loading():
    apply_theme()
    st.markdown("<div style='height:18vh'></div>", unsafe_allow_html=True)
    st.image(LOGO_URL, width=128)
    st.markdown("""
        <h3 style="margin-top:12px; letter-spacing:0.5px;text-align:center;">
        Welcome To Discovery powered by Wellsoft Corporation
        </h3>
    """, unsafe_allow_html=True)
    st.caption("Preparing your experience...")
    
    # Initialize components
    if st.session_state["client"] is None:
        st.session_state["client"] = get_openai_client()
    
    time.sleep(1.5)
    st.session_state["route"] = "home"
    st.rerun()

# -----------------------------
# Home screen: mode selection and auth
# -----------------------------
def screen_home():
    apply_theme()
    title_bar()
    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Chat without sign up")
        st.write("Start instantly in Guest mode. No chat history will be saved.")
        if st.button("Continue as Guest", use_container_width=True, type="primary"):
            st.session_state["guest_mode"] = True
            st.session_state["email"] = None
            st.session_state["user_id"] = None
            st.session_state["chat_id"] = None
            st.session_state["messages"] = []
            st.session_state["attachments"] = []
            st.session_state["route"] = "chat"
            st.rerun()

    with right:
        st.subheader("Sign up or sign in with email")
        with st.expander("Create account", expanded=True):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            if st.button("Sign up", use_container_width=True):
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    user_id = create_user(email, password)
                    if user_id:
                        st.success("Account created successfully! You can now sign in.")
                    else:
                        st.error("Email already exists or invalid input.")

        with st.expander("Sign in", expanded=True):
            email_in = st.text_input("Email", key="signin_email")
            password_in = st.text_input("Password", type="password", key="signin_password")
            if st.button("Sign in", use_container_width=True):
                if not email_in or not password_in:
                    st.error("Please enter both email and password.")
                else:
                    user_id = authenticate_user(email_in, password_in)
                    if user_id:
                        # Purge old chats
                        purge_old_history(user_id, RETENTION_DAYS)
                        # Create a fresh chat for this session
                        chat_id = create_chat(user_id, title="Session chat")
                        if chat_id:
                            st.session_state["guest_mode"] = False
                            st.session_state["email"] = email_in.lower().strip()
                            st.session_state["user_id"] = user_id
                            st.session_state["chat_id"] = chat_id
                            st.session_state["messages"] = []
                            st.session_state["attachments"] = []
                            st.session_state["route"] = "chat"
                            st.success("Signed in successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to create chat session.")
                    else:
                        st.error("Invalid email or password.")

    st.markdown("---")
    st.info("💡 Note: In Guest mode, your chat exists only in this session and is not saved to any database.")

# -----------------------------
# Settings sidebar with About and unique features
# -----------------------------
def sidebar_settings():
    presets = prompt_presets()

    st.sidebar.header("Settings")
    st.sidebar.toggle("Dark theme", key="theme_dark", value=st.session_state["theme_dark"])

    st.sidebar.selectbox("Response tone", ["Helpful", "Formal", "Friendly", "Direct"], key="tone")
    st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, st.session_state["temperature"], 0.05, key="temperature")
    st.sidebar.slider("Max output tokens", 256, 2048, st.session_state["max_output_tokens"], 64, key="max_output_tokens")
    st.sidebar.selectbox("Prompt preset", list(presets.keys()), key="preset")

    # Unique features
    st.sidebar.subheader("Unique features")
    st.sidebar.checkbox("Inline citations mode (format responses with simple numbered references)", key="feat_citations", value=False)
    st.sidebar.checkbox("Concise answers mode (extra brevity for long outputs)", key="feat_concise", value=False)
    st.sidebar.checkbox("Code-friendly mode (prefer code blocks when relevant)", key="feat_code", value=False)

    st.sidebar.button("Clear current conversation", on_click=clear_current_conversation)

    # Export conversation
    if st.sidebar.button("Export conversation as Markdown"):
        export_markdown()

    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.write("Discovery AI-powered chatbot designed to assist you with anything you need. Built with cutting-edge technology.")
    st.sidebar.write(f"Wellsoft Corporation CEO & Chairman: {CEO_CHAIRMAN}")
    st.sidebar.write(f"Founder: {FOUNDER}")
    st.sidebar.write(f"Founded: {FOUNDED_DATE}")
    st.sidebar.write(f"Contact Us: [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})")
    st.sidebar.write(f"Follow Us on Instagram: [{INSTAGRAM}]({INSTAGRAM})")
    st.sidebar.write(f"Visit Website: [{WEBSITE}]({WEBSITE})")

    st.sidebar.markdown("---")
    st.sidebar.header("License")
    st.sidebar.caption("MIT License — see footer for full text.")

def clear_current_conversation():
    if st.session_state["guest_mode"]:
        st.session_state["messages"] = []
        st.session_state["attachments"] = []
    else:
        # Create a new chat record and switch to it
        chat_id = create_chat(st.session_state["user_id"], title="New chat")
        if chat_id:
            st.session_state["chat_id"] = chat_id
            st.session_state["attachments"] = []
            st.session_state["messages"] = []
        else:
            st.error("Failed to create new chat.")
            return
    st.success("Conversation cleared.")
    st.rerun()

def export_markdown():
    # Combine session messages into markdown; include stored DB messages for signed users
    md_lines = [f"# {APP_NAME} conversation export", f"_Exported: {datetime.utcnow().isoformat()}_"]
    if st.session_state["guest_mode"]:
        for m in st.session_state["messages"]:
            md_lines.append(f"\n**{m['role'].capitalize()}:**\n\n{m['content']}")
    else:
        messages = load_chat_messages(st.session_state["chat_id"])
        for m in messages:
            md_lines.append(f"\n**{m['role'].capitalize()}:**\n\n{m['content']}")
    md_str = "\n".join(md_lines)
    b = md_str.encode("utf-8")
    st.sidebar.download_button("Download .md", data=b, file_name="discovery_export.md", mime="text/markdown")

# -----------------------------
# Chat screen
# -----------------------------
def screen_chat():
    apply_theme()
    title_bar()
    
    # User info display
    if st.session_state["guest_mode"]:
        st.info("🔸 You are in Guest mode - chat history will not be saved")
    else:
        st.success(f"🔹 Signed in as: {st.session_state['email']}")
    
    sidebar_settings()
    st.markdown("---")

    # File upload (only for signed-in users)
    if not st.session_state["guest_mode"]:
        st.write("Attach files to include their content in your prompt. Supported: text, markdown, CSV, images (base64).")
        uploads = st.file_uploader("Attach files", accept_multiple_files=True, key="file_uploader")
        if uploads:
            st.session_state["attachments"] = []
            for f in uploads:
                # Keep small files; base64-encode for context (simple demo). For large files, consider embeddings or chunking.
                content_bytes = f.read()
                mime = f.type or "application/octet-stream"
                dataurl = bytes_to_dataurl(content_bytes, mime)
                st.session_state["attachments"].append({"name": f.name, "mime": mime, "dataurl": dataurl})
            st.success(f"📎 Attached: {', '.join([f.name for f in uploads])}")

    # Show conversation
    container = st.container()
    with container:
        if st.session_state["guest_mode"]:
            for m in st.session_state["messages"]:
                render_message(m["role"], m["content"])
        else:
            db_messages = load_chat_messages(st.session_state["chat_id"])
            for m in db_messages:
                render_message(m["role"], m["content"], attachments=m.get("attachments") or "")

    # Typing indicator
    if st.session_state.get("typing", False):
        show_typing_indicator()

    # Prompt input area
    st.markdown("---")
    cols = st.columns([0.8, 0.2])
    with cols[0]:
        prompt = st.text_area("Message", placeholder="Ask anything…", height=100, key="prompt_input")
    with cols[1]:
        send = st.button("Send", use_container_width=True, type="primary")
        if st.button("Clear", use_container_width=True):
            clear_current_conversation()

    # Quick suggestion chips
    st.write("💡 Quick suggestions:")
    sug_cols = st.columns(4)
    suggestions = [
        "Summarize this article…",
        "Draft a professional email…",
        "Explain this concept simply…",
        "Help me write code with examples…",
    ]
    for i, s in enumerate(suggestions):
        if sug_cols[i].button(s, use_container_width=True, key=f"sugg_{i}"):
            st.session_state["preset"] = "None"
            handle_send(s)
            st.rerun()

    if send and prompt:
        handle_send(prompt)

    # Footer + License
    st.markdown("---")
    st.markdown("### License")
    st.markdown("""
    MIT License

    Copyright (c) 2025 Wellsoft Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """)

    st.markdown(f"""
    <div class="footer">
        © 2025 {COMPANY}. {APP_NAME} helps you get things done with care and clarity.
    </div>
    """, unsafe_allow_html=True)

def handle_send(user_text: str):
    if not user_text or not user_text.strip():
        st.warning("Please enter a message.")
        return

    # Prepare messages
    sys_prompt = make_system_prompt(st.session_state["tone"])
    messages_payload = [{"role": "system", "content": sys_prompt}]

    # Apply preset if any
    preset_text = prompt_presets().get(st.session_state["preset"], "")
    if preset_text:
        user_text = f"{preset_text}\n\n{user_text}"

    # Attachments (signed-in only): include an attachment summary
    attachment_note = ""
    if not st.session_state["guest_mode"] and st.session_state["attachments"]:
        names = format_attachments(st.session_state["attachments"])
        attachment_note = f"\n\n[Attached files: {names}]"
        # Simple embedding of data URLs for demonstration (be mindful of token limits in production)
        for a in st.session_state["attachments"]:
            # Add a brief descriptor line; do not dump massive data
            messages_payload.append({
                "role": "system",
                "content": f"Attachment meta: name={a['name']}, mime={a['mime']}. If relevant, ask to analyze content."
            })

    messages_payload.append({"role": "user", "content": user_text + attachment_note})

    # Client
    client = st.session_state["client"]
    if client is None:
        st.error("API client not configured. Please add OPENAI_API_KEY in Streamlit Secrets.")
        return

    # Display user's message immediately
    attachments_display = format_attachments(st.session_state["attachments"]) if not st.session_state["guest_mode"] else None
    render_message("user", user_text.strip(), attachments=attachments_display)

    # Save user message (signed-in users only)
    if not st.session_state["guest_mode"]:
        save_message(st.session_state["chat_id"], "user", user_text.strip(), attachments=format_attachments(st.session_state["attachments"]))

    # Typing indicator
    st.session_state["typing"] = True
    st.rerun()  # Force rerun to show typing indicator
    
    st.toast("Discovery is thinking…", icon="🤖")

    # Request completion
    try:
        temperature = st.session_state["temperature"]
        max_tokens = st.session_state["max_output_tokens"]
        model = st.session_state["model"]

        # Build assistant style toggles
        concise_note = "Keep answers concise." if st.session_state.get("feat_concise") else ""
        code_note = "Prefer code blocks when examples help." if st.session_state.get("feat_code") else ""
        citations_note = "If you reference external info, add simple inline references [1], [2]." if st.session_state.get("feat_citations") else ""

        messages_payload[0]["content"] = messages_payload[0]["content"] + f" {concise_note} {code_note} {citations_note}"

        # Make completion request
        resp = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        assistant_text = resp.choices[0].message.content.strip()

        # Render assistant response
        render_message("assistant", assistant_text)

        # Save assistant message (signed-in users only)
        if not st.session_state["guest_mode"]:
            save_message(st.session_state["chat_id"], "assistant", assistant_text)

    except Exception as e:
        st.error(f"Something went wrong while generating a response: {e}")
    finally:
        st.session_state["typing"] = False
        # Clear prompt input
        st.session_state["prompt_input"] = ""

# -----------------------------
# Router
# -----------------------------
def router():
    route = st.session_state["route"]
    if route == "loading":
        screen_loading()
    elif route == "home":
        screen_home()
    else:
        screen_chat()

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    router()
