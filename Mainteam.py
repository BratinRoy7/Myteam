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
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DiscoveryAIBrain:
    """Core AI logic: knowledge base, preprocessing, similarity, learning, and web search."""
    
    def __init__(self):
        self.conversation_history = []
        self.learned_patterns = {}
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knowledge_base = self.initialize_knowledge_base()
        self.user_feedback = {}
        
    def initialize_knowledge_base(self):
        """Initialize with comprehensive company and AI knowledge."""
        return {
            "company": {
                "name": "WellSoft Corporation",
                "ceo": "Bratin Roy",
                "foundation_year": 2025,
                "industry": "Artificial Intelligence & Information Technology",
                "mission": "To democratize AI knowledge and provide encyclopedia-like AI information services",
                "products": [
                    "Discovery AI", "AI Encyclopedia", "KnowledgeBot", 
                    "Smart Research Assistant", "AI Learning Platform"
                ],
                "values": [
                    "Innovation", "Accuracy", "Accessibility", 
                    "User-Centric", "Continuous Learning"
                ],
                "contact": "contact@wellsoftcorporation.com",
                "achievements": [
                    "Pioneering AI knowledge dissemination",
                    "Revolutionizing how people access AI information",
                    "Building the world's most comprehensive AI encyclopedia"
                ]
            },
            "ai_knowledge": {
                "machine_learning": "Machine learning is a subset of AI that enables systems to learn and improve from experience without explicit programming.",
                "deep_learning": "Deep learning uses neural networks with multiple layers to analyze various factors of data.",
                "natural_language_processing": "NLP enables computers to understand, interpret, and generate human language.",
                "computer_vision": "Computer vision enables machines to interpret and understand visual information from the world.",
                "neural_networks": "Neural networks are computing systems inspired by biological neural networks in human brains.",
                "ai_ethics": "AI ethics involves ensuring AI systems are developed and used responsibly and fairly.",
                "robotics": "Robotics combines AI with mechanical engineering to create intelligent machines.",
                "expert_systems": "Expert systems emulate the decision-making ability of human experts."
            },
            "services": {
                "ai_encyclopedia": "Comprehensive AI knowledge base with real-time updates",
                "research_assistance": "AI-powered research and information gathering",
                "learning_platform": "Interactive AI education and training",
                "custom_solutions": "Tailored AI solutions for businesses and individuals"
            }
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing with stemming and cleaning."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    def calculate_similarity(self, query, knowledge_items):
        """Calculate similarity between query and knowledge items."""
        if not knowledge_items:
            return []
            
        all_texts = [query] + list(knowledge_items.values())
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            return similarities[0]
        except Exception:
            return [0] * len(knowledge_items)
    
    def search_knowledge_base(self, query):
        """Search through the knowledge base for relevant information."""
        processed_query = self.preprocess_text(query)
        results = []
        
        # Search in AI knowledge
        ai_similarities = self.calculate_similarity(
            processed_query, 
            self.knowledge_base["ai_knowledge"]
        )
        
        for (topic, info), similarity in zip(
            self.knowledge_base["ai_knowledge"].items(), ai_similarities
        ):
            if similarity > 0.1:
                results.append({
                    "type": "ai_knowledge",
                    "topic": topic,
                    "information": info,
                    "similarity": similarity
                })
        
        # Search in company information
        company_info = {
            "company_name": self.knowledge_base["company"]["name"],
            "ceo": self.knowledge_base["company"]["ceo"],
            "mission": self.knowledge_base["company"]["mission"],
            "products": " ".join(self.knowledge_base["company"]["products"]),
            "values": " ".join(self.knowledge_base["company"]["values"])
        }
        
        company_similarities = self.calculate_similarity(
            processed_query, company_info
        )
        
        if company_similarities and max(company_similarities) > 0.1:
            results.append({
                "type": "company_info",
                "information": self.get_company_introduction(),
                "similarity": max(company_similarities)
            })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:3]
    
    def get_company_introduction(self):
        """Generate comprehensive company introduction."""
        company = self.knowledge_base["company"]
        # Fixed f-string syntax and used company['ceo'] rather than hard-coded name
        return f"""
**WellSoft Corporation** - Pioneering AI Solutions

🏢 **Company Overview:**
- **Founded:** {company['foundation_year']}
- **CEO & Founder:** {company['ceo']}
- **Industry:** {company['industry']}
- **Mission:** {company['mission']}

🚀 **Products & Services:**
{', '.join(company['products'])}

💡 **Core Values:**
{', '.join(company['values'])}

📧 **Contact:** {company['contact']}

Under {company['ceo']}'s visionary leadership, we're {company['achievements'][0]}!"""
    
    def get_ceo_info(self):
        """Generate detailed CEO information."""
        ceo_info = {
            "name": "Bratin Roy",
            "foundation_year": 2025,
            "achievements": [
                "Founded WellSoft Corporation in 2025",
                "Pioneered AI encyclopedia and knowledge dissemination",
                "Built a comprehensive AI information platform"
            ],
            "values": ["Innovation", "User-Centric Design", "Knowledge Democratization"]
        }
        
        return f"""
**{ceo_info['name']}** - Visionary CEO and Founder of WellSoft Corporation

Under his leadership:
• Founded in {ceo_info['foundation_year']}
• {ceo_info['achievements'][1]}
• {ceo_info['achievements'][2]}
• Built a company based on {', '.join(ceo_info['values'][:2])}

His vision drives our innovation in AI knowledge solutions!"""
    
    def generate_response(self, user_input, user_id="default"):
        """Generate AI response based on user input."""
        # Add to conversation history
        self.conversation_history.append({
            "user": user_id,
            "input": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Search for relevant information
        search_results = self.search_knowledge_base(user_input)
        
        if search_results:
            # Use the most relevant result
            best_match = search_results[0]
            
            if best_match["type"] == "ai_knowledge":
                response = f"**{best_match['topic'].replace('_', ' ').title()}:**\n\n{best_match['information']}\n\n"
                response += "*This information is part of our comprehensive AI encyclopedia.*"
            
            elif best_match["type"] == "company_info":
                response = best_match["information"]
            
            else:
                response = self.generate_default_response(user_input)
        
        else:
            response = self.generate_default_response(user_input)
        
        # Add learning from this interaction
        self.learn_from_interaction(user_input, response, user_id)
        
        return response
    
    def generate_default_response(self, user_input):
        """Generate a default response when no specific knowledge is found."""
        default_responses = [
            f"I'm Discovery AI, powered by WellSoft Corporation. I specialize in providing comprehensive AI knowledge. Could you rephrase your question about AI concepts?",
            f"As an AI encyclopedia from WellSoft Corporation, I focus on AI-related information. How can I help you learn about artificial intelligence?",
            f"I'd love to help you explore AI concepts! Could you specify which area of artificial intelligence you're interested in?"
        ]
        
        # Check if it's about the company
        company_keywords = ['wellsoft', 'corporation', 'bratin', 'roy', 'company', 'founder']
        if any(keyword in user_input.lower() for keyword in company_keywords):
            return self.get_company_introduction()
        
        return random.choice(default_responses)
    
    def learn_from_interaction(self, user_input, response, user_id):
        """Learn from user interactions to improve future responses."""
        key_phrases = self.extract_key_phrases(user_input)
        for phrase in key_phrases:
            if phrase not in self.learned_patterns:
                self.learned_patterns[phrase] = []
            self.learned_patterns[phrase].append({
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "user": user_id
            })
    
    def extract_key_phrases(self, text):
        """Extract key phrases from text for learning."""
        words = text.lower().split()
        key_phrases = []
        
        # Add individual important words
        important_words = ['ai', 'artificial', 'intelligence', 'machine', 'learning', 
                          'deep', 'neural', 'network', 'algorithm', 'data']
        
        for word in words:
            if word in important_words and len(word) > 2:
                key_phrases.append(word)
        
        # Add 2-word phrases
        if len(words) >= 2:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:
                    key_phrases.append(phrase)
        
        return key_phrases[:5]  # Limit to top 5 phrases

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
                user_id TEXT,
                user_input TEXT,
                ai_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        # User analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                interaction_count INTEGER DEFAULT 0,
                first_interaction DATETIME,
                last_interaction DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, user_id, user_input, ai_response, session_id):
        """Store conversation in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_id, user_input, ai_response, session_id)
            VALUES (?, ?, ?, ?)
        ''', (user_id, user_input, ai_response, session_id))
        
        # Update user analytics
        self.update_user_analytics(user_id, session_id)
        
        conn.commit()
        conn.close()
    
    def update_user_analytics(self, user_id, session_id):
        """Update user interaction analytics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute(
            'SELECT * FROM user_analytics WHERE user_id = ? AND session_id = ?',
            (user_id, session_id)
        )
        
        if cursor.fetchone():
            # Update existing
            cursor.execute('''
                UPDATE user_analytics 
                SET interaction_count = interaction_count + 1,
                    last_interaction = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
            ''', (user_id, session_id))
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO user_analytics (user_id, session_id, interaction_count, first_interaction, last_interaction)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (user_id, session_id))
        
        conn.commit()
        conn.close()
    
    def get_conversation_stats(self):
        """Get conversation statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM conversations')
        unique_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE date(timestamp) = date("now")')
        today_conversations = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "unique_users": unique_users,
            "today_conversations": today_conversations
        }

# Load Google Gemini / API key securely
# Order of precedence: Streamlit secrets -> environment variable -> (optional) hard-coded fallback (NOT RECOMMENDED)
GOOGLE_API_KEY = None
if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif os.environ.get("GOOGLE_API_KEY"):
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
else:
    # WARNING: Hard-coding API keys is insecure. Uncomment the following line only for quick local testing.
    # GOOGLE_API_KEY = "AIzaSyBXmfTDCqWuUGDM4kCs2RT86I4KBf1ghB4"
    pass

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
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"

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
    
    st.markdown('<div class="welcome-header">🚀 Discovery AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cursive-message">Welcome to Discovery AI powered by WellSoft Corporation</div>', 
        unsafe_allow_html=True
    )
    
    # Company info card
    st.markdown(
        """
        <div class="company-info">
            <h3>🏢 WellSoft Corporation</h3>
            <p><strong>Founded in 2025 by Bratin Roy</strong></p>
            <p>Pioneering AI encyclopedia and knowledge dissemination platform</p>
            <p>Democratizing AI information for everyone</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def handle_user_input(prompt):
    """Processes user input, generates AI response, and updates state/DB."""
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate AI response
        with st.spinner("🤖 Discovery AI is thinking..."):
            ai_response = ai_brain.generate_response(
                prompt, 
                st.session_state.user_id
            )
        
        # Add AI response to chat history using the correct role ("assistant")
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Store in database
        db_manager.store_conversation(
            st.session_state.user_id,
            prompt,
            ai_response,
            st.session_state.session_id
        )

def display_chat_history():
    """Displays all messages in the session state."""
    for message in st.session_state.messages:
        # Map any legacy 'system' role to 'assistant'
        role = "assistant" if message.get("role") in ("system", "assistant") else "user"
        with st.chat_message(role):
            if role == "assistant":
                st.markdown(f"🤖 {message['content']}")
            else:
                st.markdown(f"👤 {message['content']}")

def display_company_highlights():
    """Display WellSoft Corporation highlights in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 WellSoft Corporation")
    
    company = ai_brain.knowledge_base["company"]
    
    st.sidebar.markdown(f"""
    **Founded:** {company['foundation_year']}  
    **CEO:** {company['ceo']}  
    **Industry:** {company['industry']}  
    
    **Mission:**  
    {company['mission']}
    
    **Core Values:**  
    {', '.join(company['values'])}
    
    **Contact:**  
    {company['contact']}
    """)

def display_stats():
    """Displays enhanced live statistics."""
    db_stats = db_manager.get_conversation_stats()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Live Statistics")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Current Session", len(st.session_state.messages)//2)
        st.metric("Total Knowledge", len(ai_brain.knowledge_base["ai_knowledge"]))
    
    with col2:
        st.metric("Learned Patterns", len(ai_brain.learned_patterns))
        st.metric("DB Conversations", db_stats["total_conversations"])

def display_quick_questions():
    """Display quick question buttons."""
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What is AI?"):
            handle_user_input("What is artificial intelligence?")
    
    with col2:
        if st.button("Company Info"):
            handle_user_input("Tell me about WellSoft Corporation")
    
    with col3:
        if st.button("Machine Learning"):
            handle_user_input("Explain machine learning")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("CEO Info"):
            handle_user_input("Who is Bratin Roy?")
    
    with col5:
        if st.button("AI Ethics"):
            handle_user_input("What are AI ethics?")
    
    with col6:
        if st.button("Services"):
            handle_user_input("What services do you offer?")

# Main Streamlit Execution
def main_streamlit():
    """Main function to run the Streamlit application."""
    setup_streamlit_app()
    
    # Display welcome interface
    display_welcome_interface()
    
    # Sidebar content
    with st.sidebar:
        st.title("🚀 Discovery AI")
        st.markdown("---")
        display_company_highlights()
        display_stats()
    
    # Quick questions section
    display_quick_questions()
    
    st.markdown("---")
    st.subheader("💬 Chat with Discovery AI")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about AI..."):
        handle_user_input(prompt)
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
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload)
            return response.json()
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
    
    def handle_update(self, update):
        """Handle incoming Telegram updates."""
        try:
            message = update.get("message", {})
            chat_id = message.get("chat", {}).get("id")
            text = message.get("text", "")
            
            if text:
                # Generate AI response
                ai_response = self.ai_brain.generate_response(text, f"telegram_{chat_id}")
                
                # Send response
                self.send_message(chat_id, ai_response)
                
                # Store in database
                self.db_manager.store_conversation(
                    f"telegram_{chat_id}",
                    text,
                    ai_response,
                    f"telegram_session_{chat_id}"
                )
                
        except Exception as e:
            print(f"Error handling update: {e}")

if __name__ == "__main__":
    main_streamlit()
                
                
