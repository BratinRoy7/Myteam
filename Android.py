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
        self.knowledge_base = self._initialize_knowledge_base()
        self.conversation_history = []
        self.vectorizer = TfidfVectorizer()
        self.stemmer = PorterStemmer()
        self.tfidf_matrix = None
        self._train_similarity_model()
        
    def _initialize_knowledge_base(self):
        """Initialize the company knowledge base."""
        company = {
            'name': 'WellSoft Corporation',
            'foundation_year': 2023,
            'ceo': 'Bratin Roy',
            'industry': 'AI Solutions & Software Development',
            'mission': 'To democratize AI technology and make it accessible for businesses of all sizes',
            'products': ['Discovery AI', 'Enterprise AI Solutions', 'Custom AI Development', 'AI Consulting'],
            'values': ['Innovation', 'Customer-Centricity', 'Transparency', 'Excellence', 'Collaboration'],
            'contact': 'contact@wellsoft.com',
            'achievements': ['pioneering cutting-edge AI solutions that transform businesses']
        }
        
        ceo_info = {
            'name': 'Bratin Roy',
            'foundation_year': 2023,
            'achievements': [
                'Founded WellSoft Corporation with a vision to democratize AI',
                'Pioneered enterprise AI solutions for SMBs',
                'Built a team of 50+ AI experts'
            ],
            'values': ['Innovation', 'Leadership', 'Technical Excellence', 'Visionary Thinking'],
            'background': 'Seasoned AI entrepreneur and technology visionary'
        }
        
        products = {
            'Discovery AI': 'An intelligent conversational AI that learns and adapts to user needs',
            'Enterprise AI Solutions': 'Custom AI implementations for large organizations',
            'Custom AI Development': 'Tailored AI solutions for specific business requirements',
            'AI Consulting': 'Strategic AI advisory and implementation services'
        }
        
        return {
            'company': company,
            'ceo': ceo_info,
            'products': products,
            'faq': self._initialize_faq()
        }
    
    def _initialize_faq(self):
        """Initialize frequently asked questions."""
        return {
            "What does WellSoft do?": "We provide AI-powered solutions including conversational AI, enterprise AI implementations, and custom AI development services.",
            "Who is the founder?": "WellSoft was founded by Bratin Roy, a visionary AI entrepreneur.",
            "What is Discovery AI?": "Discovery AI is our flagship conversational AI platform that learns from interactions and provides intelligent responses.",
            "How can I contact WellSoft?": "You can reach us at contact@wellsoft.com for any inquiries.",
            "What industries do you serve?": "We serve businesses across various industries including healthcare, finance, retail, and technology."
        }
    
    def _train_similarity_model(self):
        """Train the TF-IDF similarity model."""
        documents = list(self.knowledge_base['faq'].keys()) + list(self.knowledge_base['faq'].values())
        if documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def preprocess_text(self, text):
        """Preprocess text for similarity comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words if word not in ['?', '!', '.', ',']]
        return ' '.join(words)
    
    def find_most_similar(self, query, threshold=0.3):
        """Find the most similar question in knowledge base."""
        if self.tfidf_matrix is None:
            return None
            
        query_processed = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query_processed])
        
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        max_similarity = np.max(similarities)
        
        if max_similarity > threshold:
            max_index = np.argmax(similarities)
            documents = list(self.knowledge_base['faq'].keys()) + list(self.knowledge_base['faq'].values())
            return documents[max_index % len(self.knowledge_base['faq'])]
        return None
    
    def generate_response(self, user_input, context=None):
        """Generate AI response using knowledge base and Gemini AI."""
        # First, try to find similar question in knowledge base
        similar_question = self.find_most_similar(user_input)
        
        if similar_question:
            if similar_question in self.knowledge_base['faq']:
                return self.knowledge_base['faq'][similar_question]
            else:
                # Find the question that matches this answer
                for question, answer in self.knowledge_base['faq'].items():
                    if answer == similar_question:
                        return answer
        
        # If no good match found, use Gemini AI
        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
            You are Discovery AI, an assistant for WellSoft Corporation. 
            Company Info: {self.knowledge_base['company']}
            
            User Question: {user_input}
            
            Provide a helpful, professional response. If you don't know something, 
            be honest and suggest contacting the company.
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Please try again later. Error: {str(e)}"
    
    def learn_from_interaction(self, user_input, ai_response):
        """Learn from new interactions (basic implementation)."""
        self.conversation_history.append({
            'user': user_input,
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 conversations to manage memory
        if len(self.conversation_history) > 100:
            self.conversation_history.pop(0)

class DatabaseManager:
    """Enhanced database manager with user analytics."""
    
    def __init__(self, db_path="discovery_ai.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_input TEXT,
                ai_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                similarity_score REAL
            )
        ''')
        
        # User analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                interaction_count INTEGER,
                first_interaction DATETIME,
                last_interaction DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, session_id, user_input, ai_response, similarity_score=None):
        """Store conversation in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (session_id, user_input, ai_response, similarity_score)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_input, ai_response, similarity_score))
        
        # Update user analytics
        cursor.execute('''
            INSERT OR REPLACE INTO user_analytics 
            (session_id, interaction_count, first_interaction, last_interaction)
            VALUES (?, 
                    COALESCE((SELECT interaction_count + 1 FROM user_analytics WHERE session_id = ?), 1),
                    COALESCE((SELECT first_interaction FROM user_analytics WHERE session_id = ?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP)
        ''', (session_id, session_id, session_id))
        
        conn.commit()
        conn.close()
    
    def get_conversation_stats(self):
        """Get conversation statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT session_id) FROM conversations')
        unique_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM user_analytics')
        total_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_conversations': total_conversations,
            'unique_users': unique_users,
            'total_sessions': total_sessions
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
    
    # Company info card
    st.markdown(
        """
        <div class="company-info">
            <h3>🏢 WellSoft Corporation</h3>
            <p><strong>Founder & CEO:</strong> Bratin Roy</p>
            <p><strong>Mission:</strong> Democratizing AI technology for businesses of all sizes</p>
            <p><strong>Specialization:</strong> AI Solutions, Enterprise Software, Custom Development</p>
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
        with st.spinner("Discovery AI is thinking..."):
            ai_response = ai_brain.generate_response(prompt)
            
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Store in database
        if session_id:
            db_manager.store_conversation(session_id, prompt, ai_response)
        
        # Learn from interaction
        ai_brain.learn_from_interaction(prompt, ai_response)

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
    **Founded:** 2023  
    **CEO:** Bratin Roy  
    **Industry:** AI Solutions  
    
    **Products:**
    - Discovery AI
    - Enterprise Solutions
    - Custom Development
    - AI Consulting
    """)

def display_stats():
    """Displays enhanced live statistics."""
    conv_count = len(ai_brain.conversation_history)
    db_stats = db_manager.get_conversation_stats()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Live Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Session Chats", conv_count)
    with col2:
        st.metric("Total Users", db_stats['unique_users'])
    
    st.sidebar.metric("Total Conversations", db_stats['total_conversations'])

def display_quick_questions():
    """Display quick question buttons."""
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What is WellSoft?"):
            handle_user_input("What is WellSoft Corporation?")
    
    with col2:
        if st.button("Who is the founder?"):
            handle_user_input("Who founded WellSoft?")
    
    with col3:
        if st.button("Contact info"):
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
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]
    
    # Display welcome interface
    display_welcome_interface()
    
    # Sidebar
    with st.sidebar:
        st.title("Discovery AI")
        st.markdown("---")
        display_company_highlights()
        display_stats()
        
        uploaded_file = handle_file_upload()
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
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
    
    def send_message(self, chat_id, text):
        """Send message via Telegram bot."""
        url = self.base_url + "sendMessage"
        data = {"chat_id": chat_id, "text": text}
        requests.post(url, data=data)
    
    def handle_update(self, update):
        """Handle incoming Telegram update."""
        try:
            message = update.get("message", {})
            chat_id = message.get("chat", {}).get("id")
            text = message.get("text", "")
            
            if text:
                # Generate AI response
                response = self.ai_brain.generate_response(text)
                
                # Send response
                self.send_message(chat_id, response)
                
                # Store in database
                session_id = f"telegram_{chat_id}"
                self.db_manager.store_conversation(session_id, text, response)
                
                # Learn from interaction
                self.ai_brain.learn_from_interaction(text, response)
                
        except Exception as e:
            print(f"Error handling Telegram update: {e}")

if __name__ == "__main__":
    main_streamlit()
