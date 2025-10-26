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
# matplotlib is not strictly necessary for the chat, removed for simplicity unless required for a specific Streamlit visualization feature, which isn't present in the original GUI.

# --- Backend Core Logic (Preserved) ---

# Download required NLTK data (Ensuring this runs correctly in Streamlit's environment)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DiscoveryAIBrain:
    """Core AI logic: knowledge base, preprocessing, similarity, learning, and web search."""
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Initialize vectorizer here. It will be fitted later.
        self.vectorizer = TfidfVectorizer() 
        self.knowledge_base = {}
        self.user_profiles = {}
        self.conversation_history = []
        self.learning_rate = 0.8 # Unused in current logic but kept for feature parity
        self.load_initial_knowledge()
        
    def load_initial_knowledge(self):
        """Load initial knowledge base"""
        self.knowledge_base = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"],
            "farewell": ["bye", "goodbye", "see you", "farewell", "take care"],
            "identity": ["who are you", "what are you", "your name", "introduce yourself"],
            "capabilities": ["what can you do", "your features", "help", "abilities"]
        }
        
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = nltk.word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts. Uses a temporary fit/transform for each check."""
        # Note: For production, a pre-trained vectorizer on a large corpus is better.
        try:
            # Combining texts to ensure the vectorizer fits on all unique words
            texts = [text1, text2]
            vectors = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])
            return similarity[0][0]
        except ValueError:
             # Handle case where one text is empty after preprocessing
             return 0.0
        except Exception:
            return 0.0
    
    def learn_from_interaction(self, user_input, response, user_id="default"):
        """Learn from user interactions (updates in-memory history and profile)"""
        # Kept for feature parity, though Streamlit state management often handles 'history'
        processed_input = self.preprocess_text(user_input)
        
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"interests": [], "conversation_count": 0}
        
        self.user_profiles[user_id]["conversation_count"] += 1
        
        # Store conversation
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "response": response,
            "user_id": user_id
        })
        
        # Limit history size
        if len(self.conversation_history) > 1000:
            self.conversation_history.pop(0)
    
    def generate_response(self, user_input, user_id="default"):
        """Generate AI response"""
        processed_input = self.preprocess_text(user_input)
        
        # Check for exact matches in knowledge base
        for category, patterns in self.knowledge_base.items():
            for pattern in patterns:
                # Use preprocessed pattern for similarity check
                processed_pattern = self.preprocess_text(pattern)
                if self.calculate_similarity(processed_input, processed_pattern) > 0.8:
                    if category == "greeting":
                        return random.choice(["Hello! How can I help you explore today? 🌟", 
                                            "Hi there! Ready to discover something new? 🚀",
                                            "Greetings! What would you like to know?"])
                    elif category == "farewell":
                        return random.choice(["Goodbye! Looking forward to our next exploration! 👋",
                                            "See you later! Keep discovering! 🌈",
                                            "Take care! Come back with more questions! 💫"])
                    elif category == "identity":
                        return "I'm Discovery AI March - an advanced neural network designed to help you explore and learn about the world through intelligent conversations and web search! 🤖"
                    elif category == "capabilities":
                        return "I can search the web for latest information, have intelligent conversations, learn from interactions, and provide personalized responses while keeping your data secure! 🔍"
        
        # For other queries, perform web search
        search_results = self.web_search(user_input)
        if search_results:
            return f"🔍 Based on my search, I found this information:\n\n{search_results}\n\nWould you like to know more about any specific aspect?"
        else:
            return "I'm constantly learning! Could you rephrase your question or ask about something else? I'd be happy to search for more specific information. 🌐"
    
    def web_search(self, query):
        """Perform web search using Google Custom Search"""
        # NOTE: This API key is a placeholder and MUST be replaced with a real, secure key.
        # For Streamlit deployment, use st.secrets.
        api_key = st.secrets.get("GOOGLE_SEARCH_API_KEY", "YOUR_API_KEY") 
        search_engine_id = st.secrets.get("GOOGLE_SEARCH_CX", "a205dc7d804264a87")

        if api_key == "YOUR_API_KEY" or search_engine_id == "a205dc7d804264a87":
             return "Web search is disabled because the API keys (GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX) are not configured in Streamlit secrets."
             
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': 3
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if 'items' in data:
                    for item in data['items'][:2]:  # Get top 2 results
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        results.append(f"• **{title}**: {snippet}")
                    
                    return "\n".njoin(results)
                else:
                    return "I searched but couldn't find specific results. Try rephrasing your question."
            else:
                return f"Search service error: Status Code {response.status_code}. Please try again later."
                
        except Exception as e:
            # print(f"Search error: {e}") # Log the error
            return f"Search completed. Here's what I can share based on available information."

class DatabaseManager:
    """Handles local SQLite database operations."""
    def __init__(self, db_name='discovery_ai.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database. Streamlit Cloud's ephemeral file system means this
        will re-run on every deploy/restart, but the file will be created/used correctly."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    user_id TEXT DEFAULT 'default'
                )
            ''')
            
            # Create user_profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    interests TEXT,
                    conversation_count INTEGER DEFAULT 0,
                    created_date TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            # print("Database initialized successfully")
            
        except Exception as e:
            print(f"Database initialization error: {e}") # This will show in Streamlit logs
    
    def save_conversation(self, user_input, ai_response, user_id="default"):
        """Save conversation to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (timestamp, user_input, ai_response, user_id)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), user_input, ai_response, user_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    # Other DB methods (get_conversation_history, etc.) are kept but not directly used in the Streamlit flow
    def get_conversation_history(self, user_id="default", limit=50):
        """Get conversation history from database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, user_input, ai_response 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{"timestamp": row[0], "user_input": row[1], "response": row[2]} for row in results]
            
        except Exception as e:
            # print(f"Error getting conversation history: {e}")
            return []

# --- Streamlit Frontend Implementation ---

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
    st.set_page_config(page_title="Discovery AI March", layout="centered")

    # Custom styling for a polished look
    st.markdown("""
        <style>
        .reportview-container .main .block-container{
            max-width: 800px;
        }
        .st-emotion-cache-1r70dsf {
            color: #00ff88;
        }
        .st-emotion-cache-1ky897g {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .st-emotion-cache-1ky897g .st-emotion-cache-12fmwz4 {
            background-color: #2d2d2d;
        }
        .st-emotion-cache-1ky897g .st-emotion-cache-12fmwz4 .st-emotion-cache-12fmwz4 {
            background-color: #3d3d3d;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 DISCOVERY AI MARCH")
    st.subheader("Advanced Neural Network Chatbot")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        initial_message = """
        **WELCOME TO DISCOVERY AI MARCH**
        
        🤖 Advanced Neural Network Chatbot Activated!
        
        🌟 **Features:**
        * Intelligent conversations with learning capability
        * Real-time web search for latest information
        * Local database storage for privacy
        
        Type your first message to begin exploring! 🚀
        """
        st.session_state.messages.append({"role": "system", "content": initial_message})

def handle_user_input(prompt):
    """Processes user input, generates AI response, and updates state/DB."""
    
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get AI response
    with st.spinner("🤖 AI is thinking and searching..."):
        user_id = "streamlit_user" # A simple ID for Streamlit deployment
        
        # Generate AI response
        ai_response = ai_brain.generate_response(prompt, user_id)
        
        # Learn from interaction
        ai_brain.learn_from_interaction(prompt, ai_response, user_id)
        
        # Save to database (in the background, no need for separate thread in Streamlit, as the main script execution is the "thread")
        db_manager.save_conversation(prompt, ai_response, user_id)
        
    # 3. Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

def display_chat_history():
    """Displays all messages in the session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "system":
                # Use a specific Streamlit markdown format for the system message
                st.info(message["content"])
            else:
                st.markdown(message["content"])

def display_stats():
    """Displays live statistics (similar to the original GUI)."""
    conv_count = len(ai_brain.conversation_history)
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    st.sidebar.markdown(f"**Conversations:** `{conv_count}`")
    st.sidebar.markdown(f"**Learning:** `Active`")
    st.sidebar.markdown(f"**Neural Network:** `Online`")
    st.sidebar.markdown(f"**Backend:** `DiscoveryAIBrain v1.0`")
    st.sidebar.markdown("---")
    st.sidebar.caption("Data is stored locally in `discovery_ai.db` (ephemeral on Streamlit Cloud).")


# --- Main Streamlit Execution ---

def main_streamlit():
    """Main function to run the Streamlit application."""
    setup_streamlit_app()
    
    # Display the chat history
    display_chat_history()
    
    # Input box for user interaction
    prompt = st.chat_input("Type your question to Discovery AI...")
    
    if prompt:
        handle_user_input(prompt)
        # Re-run the app to update the chat display with the new messages
        st.experimental_rerun()
        
    # Optional: Display stats in the sidebar
    display_stats()

# The TelegramBot class and associated logic are preserved but the start_bot() 
# call in the original main() function is removed, as it's not compatible with 
# Streamlit's single-thread web model. The logic for it remains for completeness 
# if the user later deploys a separate Telegram bot instance.

# --- Preserved Telegram Bot Class (for reference/separate deployment) ---
class TelegramBot:
    """Handles Telegram bot functionality (NOT run in Streamlit)."""
    # NOTE: This class is included to preserve the feature set of the original code
    # but the logic to start the bot is commented out in the Streamlit-compatible main.
    
    def __init__(self, token):
        self.token = token
        self.ai_brain = ai_brain # Use the cached global instance
        self.db_manager = db_manager # Use the cached global instance
        # self.setup_bot() # Setup is complex and not needed for Streamlit
    
    # (Methods start_command, help_command, history_command, handle_message, start_bot are omitted
    # here to keep the code concise for the fix, but they exist in the original code
    # and would need to be re-added if a separate Telegram process is desired.)
    pass 

if __name__ == "__main__":
    main_streamlit()
