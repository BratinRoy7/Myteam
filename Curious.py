import streamlit as st
import sqlite3
import os
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
import random
import hashlib
import time
from google import genai
from google.genai import types

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        st.warning(f"Could not download NLTK data: {e}. Some features may be limited.")

class DiscoveryAIBrain:
    """Core AI logic: knowledge base, preprocessing, similarity, learning, and Gemini integration."""
    
    def __init__(self):
        self.conversation_history = []
        self.learned_patterns = {}
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorizer_fitted = False
        self.knowledge_base = self.initialize_knowledge_base()
        self.user_feedback = {}
        
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
                self.gemini_available = True
            except Exception as e:
                st.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_available = False
                self.gemini_client = None
        else:
            self.gemini_available = False
            self.gemini_client = None
            
        self._fit_vectorizer()
        
    def _fit_vectorizer(self):
        """Fit TF-IDF vectorizer on the knowledge base once during initialization."""
        all_knowledge_texts = []
        try:
            for category, content in self.knowledge_base.items():
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, str):
                            all_knowledge_texts.append(value)
                        elif isinstance(value, list):
                            all_knowledge_texts.extend(value)
                elif isinstance(content, list):
                    all_knowledge_texts.extend(content)
            
            if len(all_knowledge_texts) >= 2:
                self.vectorizer.fit(all_knowledge_texts)
                self.vectorizer_fitted = True
            else:
                st.warning("Insufficient corpus data for vectorizer. Using fallback responses.")
        except Exception as e:
            st.warning(f"Vectorizer fitting failed: {e}. Using fallback responses.")
    
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
                "machine_learning": "Machine learning is a subset of AI that enables systems to learn and improve from experience without explicit programming. It uses algorithms to identify patterns in data and make predictions or decisions.",
                "deep_learning": "Deep learning uses neural networks with multiple layers to analyze various factors of data. It's particularly effective for image recognition, natural language processing, and complex pattern recognition.",
                "natural_language_processing": "NLP enables computers to understand, interpret, and generate human language. It powers applications like chatbots, translation services, and sentiment analysis.",
                "computer_vision": "Computer vision enables machines to interpret and understand visual information from the world. It's used in facial recognition, autonomous vehicles, and medical imaging.",
                "neural_networks": "Neural networks are computing systems inspired by biological neural networks in human brains. They consist of interconnected nodes that process information in layers.",
                "ai_ethics": "AI ethics involves ensuring AI systems are developed and used responsibly and fairly. It addresses concerns like bias, privacy, transparency, and accountability.",
                "robotics": "Robotics combines AI with mechanical engineering to create intelligent machines capable of performing tasks autonomously or semi-autonomously.",
                "expert_systems": "Expert systems emulate the decision-making ability of human experts. They use knowledge bases and inference engines to solve complex problems.",
                "reinforcement_learning": "Reinforcement learning is a type of machine learning where agents learn to make decisions by receiving rewards or penalties for their actions.",
                "generative_ai": "Generative AI creates new content including text, images, music, and code. Models like GPT and DALL-E are examples of generative AI."
            },
            "services": {
                "ai_encyclopedia": "Comprehensive AI knowledge base with real-time updates and accurate information",
                "research_assistance": "AI-powered research and information gathering for students and professionals",
                "learning_platform": "Interactive AI education and training with hands-on examples",
                "custom_solutions": "Tailored AI solutions for businesses and individuals"
            }
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing with stemming and cleaning."""
        try:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = nltk.word_tokenize(text)
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            return ' '.join(stemmed_tokens)
        except Exception as e:
            return text.lower()
    
    def calculate_similarity(self, query, knowledge_items):
        """Calculate similarity between query and knowledge items."""
        if not knowledge_items or not self.vectorizer_fitted:
            return []
            
        try:
            knowledge_texts = list(knowledge_items.values())
            query_vector = self.vectorizer.transform([query])
            knowledge_vectors = self.vectorizer.transform(knowledge_texts)
            similarities = cosine_similarity(query_vector, knowledge_vectors)
            return similarities[0]
        except Exception as e:
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
        
        if len(company_similarities) > 0 and max(company_similarities) > 0.1:
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
        return f"""**WellSoft Corporation** - Pioneering AI Solutions

ðŸ¢ **Company Overview:**
- **Founded:** {company['foundation_year']} by Bratin Roy
- **CEO & Founder:** {company['ceo']}
- **Industry:** {company['industry']}
- **Mission:** {company['mission']}

ðŸš€ **Products & Services:**
{', '.join(company['products'])}

ðŸ’¡ **Core Values:**
{', '.join(company['values'])}

ðŸ“§ **Contact:** {company['contact']}

Under {company['ceo']}'s visionary leadership, we're {company['achievements'][0]}!"""
    
    def get_ceo_info(self):
        """Generate detailed CEO information."""
        return """**Bratin Roy** - Visionary CEO and Founder of WellSoft Corporation

Under his leadership:
â€¢ Founded in 2025
â€¢ Pioneered AI encyclopedia and knowledge dissemination
â€¢ Built a comprehensive AI information platform
â€¢ Built a company based on Innovation and User-Centric Design

His vision drives our innovation in AI knowledge solutions!"""
    
    def query_gemini_api(self, user_input):
        """Query Gemini API for enhanced responses."""
        if not self.gemini_available or self.gemini_client is None:
            return None
            
        try:
            prompt = f"""You are Discovery AI, an AI assistant from WellSoft Corporation specializing in AI knowledge and information.
Provide a helpful, accurate response to the following user query. If the query is about WellSoft Corporation,
its CEO Bratin Roy, or AI topics, incorporate relevant information naturally.

User Query: {user_input}

Please provide a comprehensive yet concise response:"""
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text if response.text else None
        except Exception as e:
            st.warning(f"Gemini API temporarily unavailable: {str(e)[:100]}")
            return None
    
    def generate_response(self, user_input, user_id="default"):
        """Generate AI response based on user input with Gemini enhancement."""
        self.conversation_history.append({
            "user": user_id,
            "input": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Search knowledge base
        search_results = self.search_knowledge_base(user_input)
        
        # Query Gemini API
        gemini_response = self.query_gemini_api(user_input)
        
        # Combine results intelligently
        if search_results and gemini_response:
            best_match = search_results[0]
            
            if best_match["type"] == "ai_knowledge":
                response = f"**{best_match['topic'].replace('_', ' ').title()}:**\n\n{best_match['information']}\n\n"
                response += f"**Additional Insight:**\n{gemini_response}\n\n"
                response += "*Enhanced with Google Gemini AI*"
            elif best_match["type"] == "company_info":
                response = best_match["information"] + f"\n\n**Additional Information:**\n{gemini_response}"
            else:
                response = gemini_response
        elif gemini_response:
            response = f"{gemini_response}\n\n*Powered by Google Gemini AI*"
        elif search_results:
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
        
        self.learn_from_interaction(user_input, response, user_id)
        return response
    
    def generate_default_response(self, user_input):
        """Generate a default response when no specific knowledge is found."""
        default_responses = [
            "I'm Discovery AI, powered by WellSoft Corporation and enhanced with Google Gemini AI. I specialize in providing comprehensive AI knowledge. Could you rephrase your question about AI concepts?",
            "As an AI encyclopedia from WellSoft Corporation with Gemini AI integration, I focus on AI-related information. How can I help you learn about artificial intelligence?",
            "I'd love to help you explore AI concepts! Could you specify which area of artificial intelligence you're interested in?"
        ]
        
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
        
        important_words = ['ai', 'artificial', 'intelligence', 'machine', 'learning', 
                          'deep', 'neural', 'network', 'algorithm', 'data']
        
        for word in words:
            if word in important_words and len(word) > 2:
                key_phrases.append(word)
        
        if len(words) >= 2:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:
                    key_phrases.append(phrase)
        
        return key_phrases[:5]

class DatabaseManager:
    """Enhanced database manager with user analytics."""
    
    def __init__(self, db_path="discovery_ai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                # Feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT,
                        conversation_id INTEGER,
                        feedback_type TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")
    
    def store_conversation(self, user_id, user_input, ai_response, session_id):
        """Store conversation in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO conversations (user_id, user_input, ai_response, session_id)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, user_input, ai_response, session_id))
                
                conversation_id = cursor.lastrowid
                
                self.update_user_analytics(user_id, session_id, cursor)
                
                conn.commit()
                return conversation_id
        except Exception as e:
            st.error(f"Error storing conversation: {e}")
            return None
    
    def update_user_analytics(self, user_id, session_id, cursor):
        """Update user interaction analytics."""
        try:
            cursor.execute(
                'SELECT * FROM user_analytics WHERE user_id = ? AND session_id = ?',
                (user_id, session_id)
            )
            
            if cursor.fetchone():
                cursor.execute('''
                    UPDATE user_analytics 
                    SET interaction_count = interaction_count + 1,
                        last_interaction = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
            else:
                cursor.execute('''
                    INSERT INTO user_analytics (user_id, session_id, interaction_count, first_interaction, last_interaction)
                    VALUES (?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (user_id, session_id))
        except Exception as e:
            st.error(f"Error updating analytics: {e}")
    
    def store_feedback(self, user_id, session_id, conversation_id, feedback_type):
        """Store user feedback."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO feedback (user_id, session_id, conversation_id, feedback_type)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, session_id, conversation_id, feedback_type))
                
                conn.commit()
        except Exception as e:
            st.error(f"Error storing feedback: {e}")
    
    def get_conversation_stats(self):
        """Get conversation statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM conversations')
                total_conversations = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT user_id) FROM conversations')
                unique_users = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM conversations WHERE date(timestamp) = date("now")')
                today_conversations = cursor.fetchone()[0]
                
                return {
                    "total_conversations": total_conversations,
                    "unique_users": unique_users,
                    "today_conversations": today_conversations
                }
        except Exception as e:
            return {
                "total_conversations": 0,
                "unique_users": 0,
                "today_conversations": 0
            }

# Initialize global brain and DB manager using Streamlit's cache
@st.cache_resource
def get_ai_brain_and_db_manager(_api_key_hash=None):
    """Initializes the AI brain and DB manager once. Cache invalidates when API key changes."""
    ai_brain = DiscoveryAIBrain()
    db_manager = DatabaseManager()
    return ai_brain, db_manager

def get_api_key_hash():
    """Generate hash of API key for cache invalidation."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return hashlib.md5(api_key.encode()).hexdigest() if api_key else "no_key"

def setup_streamlit_app():
    """Sets up the Streamlit page layout and initial state."""
    st.set_page_config(
        page_title="Discovery AI - WellSoft Corporation",
        page_icon="ðŸš€",
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
    
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

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
        .gemini-badge {
            background: linear-gradient(135deg, #4285F4, #34A853, #FBBC05, #EA4335);
            padding: 0.5em 1em;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            text-align: center;
            margin: 1em 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="welcome-header">ðŸš€ Discovery AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cursive-message">Welcome to Discovery AI powered by WellSoft Corporation</div>', 
        unsafe_allow_html=True
    )
    
    # Gemini badge
    st.markdown(
        '<div class="gemini-badge">ðŸ¤– Enhanced with Google Gemini AI</div>',
        unsafe_allow_html=True
    )
    
    # Company info card
    st.markdown(
        """
        <div class="company-info">
            <h3>ðŸ¢ WellSoft Corporation</h3>
            <p><strong>Founded in 2025 by Bratin Roy</strong></p>
            <p>Pioneering AI encyclopedia and knowledge dissemination platform</p>
            <p>Democratizing AI information for everyone</p>
            <p><strong>Now powered by Google Gemini AI technology</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_chat_history(ai_brain, db_manager):
    """Displays all messages in the session state with feedback options."""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message['content'])
                
                # Add feedback buttons for each assistant message
                col1, col2, col3 = st.columns([1, 1, 8])
                feedback_key = f"feedback_{idx}"
                
                with col1:
                    if st.button("ðŸ‘", key=f"thumbs_up_{idx}"):
                        if feedback_key not in st.session_state.feedback_given:
                            conversation_id = message.get("conversation_id")
                            if conversation_id:
                                db_manager.store_feedback(
                                    st.session_state.user_id,
                                    st.session_state.session_id,
                                    conversation_id,
                                    "positive"
                                )
                                st.session_state.feedback_given.add(feedback_key)
                                st.success("Thanks for your feedback!")
                
                with col2:
                    if st.button("ðŸ‘Ž", key=f"thumbs_down_{idx}"):
                        if feedback_key not in st.session_state.feedback_given:
                            conversation_id = message.get("conversation_id")
                            if conversation_id:
                                db_manager.store_feedback(
                                    st.session_state.user_id,
                                    st.session_state.session_id,
                                    conversation_id,
                                    "negative"
                                )
                                st.session_state.feedback_given.add(feedback_key)
                                st.info("Thanks! We'll improve our responses.")
            else:
                st.markdown(message['content'])

def display_sidebar(ai_brain, db_manager):
    """Display comprehensive sidebar with company info and stats."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("ðŸ¢ WellSoft Corporation")
        
        company = ai_brain.knowledge_base["company"]
        
        st.markdown(f"""
        **Founded:** {company['foundation_year']}  
        **CEO:** {company['ceo']}  
        **Industry:** {company['industry']}  
        
        **Mission:**  
        {company['mission']}
        
        **Core Values:**  
        {', '.join(company['values'][:3])}
        """)
        
        st.markdown("---")
        st.subheader("ðŸ“Š Session Statistics")
        
        stats = db_manager.get_conversation_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chats", stats["total_conversations"])
        with col2:
            st.metric("Unique Users", stats["unique_users"])
        
        st.metric("Today's Chats", stats["today_conversations"])
        
        st.markdown("---")
        st.subheader("ðŸ’¡ AI Topics")
        
        topics = list(ai_brain.knowledge_base["ai_knowledge"].keys())
        st.markdown("Explore these AI concepts:")
        for topic in topics[:5]:
            st.markdown(f"â€¢ {topic.replace('_', ' ').title()}")
        
        st.markdown("---")
        st.markdown("### âš™ï¸ Session Info")
        st.markdown(f"**User ID:** {st.session_state.user_id}")
        st.markdown(f"**Session:** {st.session_state.session_id}")
        
        if st.button("ðŸ”„ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.rerun()

def main():
    """Main application function."""
    setup_streamlit_app()
    
    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("âš ï¸ GEMINI_API_KEY not found in environment variables. Please add it to use AI features.")
        st.info("The app will work with limited functionality using the knowledge base only.")
    
    # Initialize AI brain and database with API key hash for cache invalidation
    api_key_hash = get_api_key_hash()
    ai_brain, db_manager = get_ai_brain_and_db_manager(_api_key_hash=api_key_hash)
    
    # Display welcome interface
    if not st.session_state.messages:
        display_welcome_interface()
    
    # Display sidebar
    display_sidebar(ai_brain, db_manager)
    
    # Display chat history
    display_chat_history(ai_brain, db_manager)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about AI or WellSoft Corporation..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("ðŸ¤– Discovery AI is thinking..."):
                ai_response = ai_brain.generate_response(
                    prompt, 
                    st.session_state.user_id
                )
            
            st.markdown(ai_response)
        
        # Store in database
        conversation_id = db_manager.store_conversation(
            st.session_state.user_id,
            prompt,
            ai_response,
            st.session_state.session_id
        )
        
        # Add AI response to chat history with conversation_id for feedback
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_response,
            "conversation_id": conversation_id
        })
        
        st.rerun()

if __name__ == "__main__":
    main()
