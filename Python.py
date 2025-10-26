import streamlit as st
import sqlite3
import requests
import json
import threading
import time
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
import random
import hashlib
import google.generativeai as genai
import os
from PIL import Image
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBXmfTDCqWuUGDM4kCs2RT86I4KBf1ghB4"
genai.configure(api_key=GEMINI_API_KEY)

class DiscoveryAIBrain:
    """Core AI logic: knowledge base, preprocessing, similarity, learning, and web search."""

    def __init__(self):
        self.knowledge_base = self.initialize_knowledge_base()
        self.conversation_history = []
        self.learning_memory = []
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.setup_gemini_model()

    def setup_gemini_model(self):
        """Initialize Gemini model"""
        try:
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Error initializing Gemini model: {e}")
            self.gemini_model = None

    def initialize_knowledge_base(self):
        """Initialize the knowledge base with company information."""
        company = {
            'name': 'WellSoft Corporation',
            'foundation_year': '2020',
            'ceo': 'Bratin Roy',
            'industry': 'AI Solutions & Software Development',
            'mission': 'Democratizing AI technology for businesses of all sizes',
            'products': [
                'Discovery AI Platform',
                'Enterprise Knowledge Management',
                'Custom AI Solutions',
                'AI-powered Chatbots'
            ],
            'values': [
                'Innovation',
                'Customer Success',
                'Transparency',
                'Continuous Learning'
            ],
            'contact': 'contact@wellsoft.com',
            'achievements': [
                'Developing cutting-edge AI solutions that transform businesses'
            ]
        }

        ceo_info = {
            'name': 'Bratin Roy',
            'foundation_year': '2020',
            'achievements': [
                'Pioneering AI solutions since 2020',
                'Leading innovation in enterprise AI',
                'Building transformative technology solutions'
            ],
            'values': ['Innovation', 'Excellence', 'Leadership', 'Vision']
        }

        return {
            "company_intro": f"""**WellSoft Corporation - Pioneering AI Solutions**

🏢 **Company Overview:**
• **Founded:** {company['foundation_year']} by Bratin Roy
• **CEO & Founder:** {company['ceo']}
• **Industry:** {company['industry']}
• **Mission:** {company['mission']}

🚀 **Products & Services:** {', '.join(company['products'])}

💡 **Core Values:** {', '.join(company['values'])}

📧 **Contact:** {company['contact']}

Under {company['ceo']}'s visionary leadership, we're {company['achievements'][0]}!""",

            "ceo_intro": f"""**{ceo_info['name']} - Visionary CEO and Founder of WellSoft Corporation**

Under his leadership:
• Founded in {ceo_info['foundation_year']}
• {ceo_info['achievements'][1]}
• {ceo_info['achievements'][2]}
• Built a company based on {', '.join(ceo_info['values'][:2])}

His vision drives our innovation in AI knowledge solutions!""",

            "greetings": {
                "hello": "Hello! I'm Discovery AI from WellSoft Corporation. How can I assist you today?",
                "hi": "Hi there! I'm here to help you learn about WellSoft Corporation and our AI solutions.",
                "hey": "Hey! Welcome to Discovery AI. What would you like to know about WellSoft?",
                "good morning": "Good morning! Ready to explore AI solutions with WellSoft Corporation?",
                "good afternoon": "Good afternoon! How can I help you discover our AI offerings?",
                "good evening": "Good evening! I'm here to assist with any questions about WellSoft."
            },

            "capabilities": [
                "I can tell you about WellSoft Corporation and our products",
                "I can discuss our CEO Bratin Roy and his vision",
                "I can help with AI-related questions",
                "I can provide information about our services and values",
                "I can learn from our conversations to improve",
                "I can process uploaded files and extract information"
            ],

            "services": {
                "discovery ai": "Our flagship Discovery AI platform helps businesses leverage artificial intelligence for knowledge management and customer engagement.",
                "enterprise solutions": "We provide custom AI solutions tailored to enterprise needs, including automation, analytics, and intelligent systems.",
                "consulting": "Our AI consulting services help businesses implement and optimize AI technologies for maximum impact.",
                "support": "We offer comprehensive support and maintenance for all our AI solutions."
            }
        }

    def preprocess_text(self, text):
        """Preprocess text for similarity comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words if word.isalnum()]
        return ' '.join(words)

    def get_similarity_score(self, query, context):
        """Calculate similarity between query and context."""
        try:
            query_processed = self.preprocess_text(query)
            context_processed = self.preprocess_text(context)
            
            tfidf_matrix = self.vectorizer.fit_transform([query_processed, context_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity[0][0]
        except:
            return 0.0

    def find_best_match(self, query):
        """Find the best matching response from knowledge base."""
        best_score = 0
        best_response = None
        
        query_lower = query.lower()

        # Check greetings
        for greeting, response in self.knowledge_base["greetings"].items():
            if greeting in query_lower:
                return response

        # Check company info
        company_intro_score = self.get_similarity_score(query, self.knowledge_base["company_intro"])
        if company_intro_score > best_score:
            best_score = company_intro_score
            best_response = self.knowledge_base["company_intro"]

        # Check CEO info
        ceo_score = self.get_similarity_score(query, self.knowledge_base["ceo_intro"])
        if ceo_score > best_score:
            best_score = ceo_score
            best_response = self.knowledge_base["ceo_intro"]

        # Check services
        for service, description in self.knowledge_base["services"].items():
            service_score = self.get_similarity_score(query, service + " " + description)
            if service_score > best_score:
                best_score = service_score
                best_response = f"**{service.title()}**: {description}"

        # Check capabilities
        for capability in self.knowledge_base["capabilities"]:
            capability_score = self.get_similarity_score(query, capability)
            if capability_score > best_score:
                best_score = capability_score
                best_response = f"I can help with that! {capability}"

        return best_response if best_score > 0.3 else None

    def generate_gemini_response(self, prompt, uploaded_file=None):
        """Generate response using Gemini API with optional file processing."""
        try:
            if not self.gemini_model:
                return "Gemini API is currently unavailable. Please try again later."

            if uploaded_file:
                # Process file with Gemini
                file_content = self.process_uploaded_file(uploaded_file)
                if file_content:
                    enhanced_prompt = f"""
                    User query: {prompt}
                    
                    Uploaded file content: {file_content}
                    
                    Please provide a helpful response considering both the user's query and the uploaded file content.
                    """
                    response = self.gemini_model.generate_content(enhanced_prompt)
                else:
                    response = self.gemini_model.generate_content(prompt)
            else:
                response = self.gemini_model.generate_content(prompt)

            return response.text
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."

    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file and extract text content."""
        try:
            # For text files
            if uploaded_file.type == "text/plain":
                return uploaded_file.getvalue().decode("utf-8")
            
            # For PDF files (you would need PyPDF2 or similar)
            elif uploaded_file.type == "application/pdf":
                return "PDF file detected. Please ensure you have the necessary libraries installed for PDF processing."
            
            # For images
            elif uploaded_file.type.startswith('image/'):
                image = Image.open(uploaded_file)
                return f"Image file: {uploaded_file.name} (Size: {image.size})"
            
            else:
                return f"File type: {uploaded_file.type}. Content processing may be limited."
                
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def get_response(self, user_input, uploaded_file=None):
        """Main method to get AI response."""
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "file_uploaded": uploaded_file.name if uploaded_file else None
        })

        # First try knowledge base
        knowledge_response = self.find_best_match(user_input)
        
        if knowledge_response:
            return knowledge_response
        
        # If no good match in knowledge base, use Gemini
        if uploaded_file:
            gemini_response = self.generate_gemini_response(
                f"Based on the uploaded file and this question: {user_input}", 
                uploaded_file
            )
        else:
            gemini_response = self.generate_gemini_response(user_input)
        
        return gemini_response

    def learn_from_interaction(self, user_input, response, feedback=None):
        """Learn from user interactions to improve future responses."""
        learning_entry = {
            "user_input": user_input,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        self.learning_memory.append(learning_entry)
        
        # Keep only last 100 learning entries
        if len(self.learning_memory) > 100:
            self.learning_memory.pop(0)

class DatabaseManager:
    """Enhanced database manager with user analytics."""

    def __init__(self, db_path="discovery_ai.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT,
                file_uploaded TEXT,
                session_id TEXT
            )
        ''')
        
        # User analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                action_type TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_conversation(self, user_input, ai_response, file_uploaded=None, session_id=None):
        """Save conversation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (timestamp, user_input, ai_response, file_uploaded, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), user_input, ai_response, file_uploaded, session_id))
        
        conn.commit()
        conn.close()

    def get_conversation_stats(self):
        """Get conversation statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT session_id) FROM conversations')
        unique_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE file_uploaded IS NOT NULL')
        files_processed = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "unique_sessions": unique_sessions,
            "files_processed": files_processed
        }

# Initialize global brain and DB manager using Streamlit's cache
@st.cache_resource
def get_ai_brain_and_db_manager():
    """Initializes the AI brain and DB manager once."""
    ai_brain = DiscoveryAIBrain()
    db_manager = DatabaseManager()
    return ai_brain, db_manager

ai_brain, db_manager = get_ai_brain_and_db_manager()

def setup_streamlit_app():
    """Sets up the Streamlit page layout and initial state."""
    st.set_page_config(
        page_title="Discovery AI - WellSoft Corporation",
        page_icon="🚀",
        layout="centered",
        initial_sidebar_state="expanded"
    )

def display_welcome_interface():
    """Display attractive welcome interface."""
    st.markdown(
        """
        <style>
        .welcome-header {
            font-size: 3.5em;
            font-weight: bold;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .cursive-message {
            font-family: 'Brush Script MT', cursive;
            font-size: 2em;
            text-align: center;
            color: #FFD700;
            margin-bottom: 2em;
        }
        .company-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2em;
            border-radius: 15px;
            color: white;
            margin: 2em 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="welcome-header">Discovery AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="cursive-message">Your Intelligent Assistant from WellSoft Corporation</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="company-info">
            <h3>🤖 Powered by Gemini AI & Advanced NLP</h3>
            <p>Experience the future of AI-powered conversations with file processing capabilities</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def handle_user_input(prompt, uploaded_file=None, session_id=None):
    """Processes user input, generates AI response, and updates state/DB."""
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "file": uploaded_file})
        
        # Generate AI response
        with st.spinner("🤔 Thinking..."):
            ai_response = ai_brain.get_response(prompt, uploaded_file)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Save to database
        db_manager.save_conversation(
            prompt, 
            ai_response, 
            uploaded_file.name if uploaded_file else None,
            session_id
        )

def display_chat_history():
    """Displays all messages in the session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                if message.get("file"):
                    st.info(f"📎 File: {message['file'].name}")
                st.markdown(f"👤 {message['content']}")
            else:
                st.success(f"🤖 {message['content']}")

def display_company_highlights():
    """Display WellSoft Corporation highlights in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 WellSoft Corporation")
    
    st.sidebar.markdown("""
    **Our Mission:**
    Democratizing AI technology for businesses of all sizes
    
    **Core Products:**
    • Discovery AI Platform
    • Enterprise Solutions
    • AI Consulting
    • Custom Development
    
    **Contact:**
    📧 contact@wellsoft.com
    """)

def display_stats():
    """Displays enhanced live statistics."""
    conv_count = len(ai_brain.conversation_history)
    db_stats = db_manager.get_conversation_stats()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Live Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Current Session", len(st.session_state.messages)//2)
        st.metric("Total Conversations", db_stats["total_conversations"])
    
    with col2:
        st.metric("Unique Sessions", db_stats["unique_sessions"])
        st.metric("Files Processed", db_stats["files_processed"])

def display_quick_questions():
    """Display quick question buttons."""
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏢 About Company"):
            handle_user_input("Tell me about WellSoft Corporation")
        if st.button("👨‍💼 About CEO"):
            handle_user_input("Tell me about Bratin Roy")
    
    with col2:
        if st.button("🚀 Services"):
            handle_user_input("What services do you offer?")
        if st.button("💼 Products"):
            handle_user_input("What are your main products?")
    
    with col3:
        if st.button("🎯 Mission"):
            handle_user_input("What is your mission?")
        if st.button("📞 Contact"):
            handle_user_input("How can I contact WellSoft?")

def handle_file_upload():
    """Handle file upload and return file object."""
    uploaded_file = st.sidebar.file_uploader(
        "📎 Upload a file for analysis",
        type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
        help="Upload text files, PDFs, or images for analysis"
    )
    return uploaded_file

# Main Streamlit Execution
def main_streamlit():
    """Main function to run the Streamlit application."""
    setup_streamlit_app()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    # Display interface
    display_welcome_interface()
    
    # Sidebar
    with st.sidebar:
        display_company_highlights()
        display_stats()
        
        # File upload
        uploaded_file = handle_file_upload()
        if uploaded_file:
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Quick questions
    display_quick_questions()
    
    st.markdown("---")
    st.markdown("### 💬 Chat with Discovery AI")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    prompt = st.chat_input("Type your message here...")
    
    if prompt:
        handle_user_input(prompt, uploaded_file, st.session_state.session_id)
        st.rerun()

# Preserved Telegram Bot Class (for reference/separate deployment)
class TelegramBot:
    """Handles Telegram bot functionality (NOT run in Streamlit)."""
    def __init__(self, token, ai_brain, db_manager):
        self.token = token
        self.ai_brain = ai_brain
        self.db_manager = db_manager
        self.base_url = f"https://api.telegram.org/bot{token}/"

if __name__ == "__main__":
    main_streamlit()
