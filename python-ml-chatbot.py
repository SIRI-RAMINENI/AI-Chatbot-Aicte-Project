import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import time
import random
import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Python AI Chatbot with ML")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")
        
        # Set minimum size
        self.root.minsize(400, 400)
        
        # Configure main window grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Initialize responses
        self.load_responses()
        
        # Initialize ML model
        self.init_ml_model()
        
        # Create widgets
        self.create_header_frame()
        self.create_chat_frame()
        self.create_input_frame()
        
        # Add initial bot message
        self.add_bot_message("Hello! I'm your AI assistant with machine learning capabilities. How can I help you today?")
        
        # Initialize typing indicator state
        self.typing_indicator_showing = False
        
        # For storing conversation history
        self.conversation_history = []
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_responses(self):
        """Load or initialize chatbot responses"""
        self.responses_file = "chatbot_responses.json"
        
        # Default responses if file doesn't exist
        self.default_responses = {
            "greetings": [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Greetings! How may I be of service?"
            ],
            "farewells": [
                "Goodbye! Have a great day!",
                "Farewell! Come back if you have more questions.",
                "Bye! It was nice chatting with you."
            ],
            "unknown": [
                "I'm not sure I understand. Can you rephrase that?",
                "I don't have an answer for that yet. Is there something else I can help with?",
                "That's an interesting question. Let me think about that."
            ],
            "thanks": [
                "You're welcome!",
                "Anytime! Happy to help.",
                "My pleasure!"
            ],
            "capabilities": [
                "I can answer questions, provide information, and have conversations on various topics. I also learn from our interactions!",
                "I'm designed to be helpful, harmless, and honest in my responses. I use machine learning to improve over time.",
                "I can assist with information, answer questions, or just chat. The more we talk, the more I learn!"
            ],
            "learning": [
                "I'm constantly learning from our conversations to provide better responses.",
                "My machine learning model helps me understand your questions better over time.",
                "Each conversation helps me improve my responses through my ML model."
            ],
            "weather": [
                "I don't have access to real-time weather data, but I can discuss weather patterns in general.",
                "Weather forecasting involves analyzing atmospheric conditions using various measurements and models.",
                "If you're interested in the current weather, you might want to check a weather service or app."
            ],
            "jokes": [
                "Why don't scientists trust atoms? Because they make up everything!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!"
            ],
            "tech": [
                "Technology is advancing rapidly, with AI being one of the most exciting fields.",
                "Machine learning allows computers to learn patterns from data without explicit programming.",
                "Python is a versatile programming language widely used in data science and AI development."
            ]
        }
        
        # Try to load responses from file, or create file if it doesn't exist
        try:
            if os.path.exists(self.responses_file):
                with open(self.responses_file, 'r') as f:
                    self.responses = json.load(f)
            else:
                self.responses = self.default_responses
                with open(self.responses_file, 'w') as f:
                    json.dump(self.default_responses, f, indent=4)
        except Exception as e:
            print(f"Error loading responses: {e}")
            self.responses = self.default_responses
    
    def init_ml_model(self):
        """Initialize or load the machine learning model"""
        self.model_file = "chatbot_ml_model.pkl"
        self.training_data_file = "chatbot_training_data.json"
        
        # Default training data
        self.default_training_data = {
            "data": [
                {"message": "hello", "category": "greetings"},
                {"message": "hi", "category": "greetings"},
                {"message": "hey there", "category": "greetings"},
                {"message": "good morning", "category": "greetings"},
                {"message": "goodbye", "category": "farewells"},
                {"message": "bye", "category": "farewells"},
                {"message": "see you later", "category": "farewells"},
                {"message": "thanks", "category": "thanks"},
                {"message": "thank you", "category": "thanks"},
                {"message": "appreciate it", "category": "thanks"},
                {"message": "what can you do", "category": "capabilities"},
                {"message": "what are your capabilities", "category": "capabilities"},
                {"message": "help me", "category": "capabilities"},
                {"message": "how do you learn", "category": "learning"},
                {"message": "tell me about machine learning", "category": "learning"},
                {"message": "how does AI work", "category": "learning"},
                {"message": "what's the weather", "category": "weather"},
                {"message": "will it rain today", "category": "weather"},
                {"message": "tell me a joke", "category": "jokes"},
                {"message": "know any jokes", "category": "jokes"},
                {"message": "make me laugh", "category": "jokes"},
                {"message": "tell me about technology", "category": "tech"},
                {"message": "how does the internet work", "category": "tech"},
                {"message": "what is artificial intelligence", "category": "tech"}
            ],
            "categories": ["greetings", "farewells", "thanks", "capabilities", "learning", "weather", "jokes", "tech", "unknown"]
        }
        
        # Load or create training data
        try:
            if os.path.exists(self.training_data_file):
                with open(self.training_data_file, 'r') as f:
                    self.training_data = json.load(f)
            else:
                self.training_data = self.default_training_data
                with open(self.training_data_file, 'w') as f:
                    json.dump(self.default_training_data, f, indent=4)
        except Exception as e:
            print(f"Error loading training data: {e}")
            self.training_data = self.default_training_data
        
        # Setup the ML pipeline
        self.ml_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Try to load existing model or train a new one
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.ml_pipeline = pickle.load(f)
                print("ML model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.train_model()
        else:
            self.train_model()
    
    def train_model(self):
        """Train the machine learning model with the current training data"""
        if not self.training_data["data"]:
            print("No training data available")
            return
        
        X = [item["message"] for item in self.training_data["data"]]
        y = [item["category"] for item in self.training_data["data"]]
        
        try:
            self.ml_pipeline.fit(X, y)
            print("Model trained successfully")
            
            # Save the model
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.ml_pipeline, f)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error training model: {e}")
    
    def predict_category(self, message):
        """Use the ML model to predict the category of a message"""
        try:
            # If we have very few training examples, the prediction might fail
            # so we'll use a try-except block
            
            # First try exact matching with training data
            for item in self.training_data["data"]:
                if message.lower() == item["message"].lower():
                    return item["category"]
            
            # Next try with the ML model
            predicted_category = self.ml_pipeline.predict([message])[0]
            print(f"Predicted category: {predicted_category}")
            
            # Calculate confidence
            probabilities = self.ml_pipeline.predict_proba([message])[0]
            max_prob = max(probabilities)
            print(f"Confidence: {max_prob:.2f}")
            
            # If confidence is too low, return "unknown"
            if max_prob < 0.3:
                return "unknown"
                
            return predicted_category
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown"
    
    def update_training_data(self, message, category):
        """Add new training data and retrain the model"""
        # Add the new data point
        self.training_data["data"].append({"message": message.lower(), "category": category})
        
        # Save the updated training data
        with open(self.training_data_file, 'w') as f:
            json.dump(self.training_data, f, indent=4)
        
        # Retrain the model (this could be done less frequently in a real application)
        self.train_model()
    
    def create_header_frame(self):
        """Create the header frame with title and buttons"""
        header_frame = tk.Frame(self.root, bg="#2563eb", pady=10)
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Title
        title_label = tk.Label(header_frame, text="AI Chatbot with ML", font=("Arial", 16, "bold"), 
                            bg="#2563eb", fg="white")
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_button = tk.Button(header_frame, text="Clear Chat", 
                              command=self.clear_chat,
                              bg="#1d4ed8", fg="white",
                              activebackground="#1e40af", activeforeground="white",
                              relief=tk.FLAT, padx=10)
        clear_button.pack(side=tk.RIGHT, padx=10)
        
        # Train button
        train_button = tk.Button(header_frame, text="Show Stats", 
                              command=self.show_model_stats,
                              bg="#1d4ed8", fg="white",
                              activebackground="#1e40af", activeforeground="white",
                              relief=tk.FLAT, padx=10)
        train_button.pack(side=tk.RIGHT, padx=10)

    def create_chat_frame(self):
        """Create the chat history display area"""
        # Chat frame
        self.chat_frame = tk.Frame(self.root, bg="#f3f4f6")
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD,
                                                  bg="#f3f4f6", relief=tk.FLAT,
                                                  font=("Arial", 10))
        self.chat_display.grid(row=0, column=0, sticky="nsew")
        self.chat_display.configure(state=tk.DISABLED)
        
        # Configure tags for different message types
        self.chat_display.tag_configure("user", background="#e1f0ff", justify="right")
        self.chat_display.tag_configure("user_header", background="#e1f0ff", foreground="#2563eb", font=("Arial", 9, "bold"))
        self.chat_display.tag_configure("bot", background="#f0f0f0", justify="left")
        self.chat_display.tag_configure("bot_header", background="#f0f0f0", foreground="#4b5563", font=("Arial", 9, "bold"))
        self.chat_display.tag_configure("typing", background="#f0f0f0", foreground="#6b7280", font=("Arial", 10, "italic"))
        self.chat_display.tag_configure("debug", foreground="#6b7280", font=("Arial", 8))

    def create_input_frame(self):
        """Create the input area for user messages"""
        input_frame = tk.Frame(self.root, bg="#ffffff", pady=10)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Message entry
        self.message_entry = tk.Entry(input_frame, font=("Arial", 11), relief=tk.GROOVE, bd=1)
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.message_entry.bind("<Return>", self.send_message)
        self.message_entry.focus_set()
        
        # Send button
        send_button = tk.Button(input_frame, text="Send", width=8,
                             command=self.send_message_button,
                             bg="#2563eb", fg="white",
                             activebackground="#1e40af", activeforeground="white",
                             relief=tk.FLAT)
        send_button.grid(row=0, column=1)

    def send_message_button(self):
        """Handler for Send button click"""
        self.send_message(None)

    def send_message(self, event=None):
        """Process and display user message and get response"""
        message = self.message_entry.get().strip()
        if not message:
            return
        
        # Clear entry field
        self.message_entry.delete(0, tk.END)
        
        # Display user message
        self.add_user_message(message)
        
        # Show typing indicator
        self.show_typing_indicator()
        
        # Process message in a thread to avoid freezing GUI
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        """Process message and generate response"""
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "message": message})
        
        # Simulate processing time (would be real processing time with a real AI model)
        time.sleep(random.uniform(0.8, 1.5))
        
        # Generate response based on message content
        category = self.predict_category(message)
        response = self.get_response(message, category)
        
        # Add debug info if the message is complex
        if len(message.split()) > 3 and category != "unknown":
            # 25% chance to show learning info
            if random.random() < 0.25:
                self.add_debug_message(f"I classified your message as '{category}' and I'm learning from this interaction.")
                # Actually update the training data with this example
                self.update_training_data(message, category)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "bot", "message": response})
        
        # Hide typing indicator and display response
        self.hide_typing_indicator()
        self.add_bot_message(response)

    def get_response(self, message, category):
        """Generate a response based on the predicted category"""
        # If we have responses for this category, choose one randomly
        if category in self.responses:
            return random.choice(self.responses[category])
        
        # For unknown messages, try to find similar messages in training data
        if category == "unknown":
            # Use TF-IDF to vectorize the message and find similar ones
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            
            # Get all training messages
            train_messages = [item["message"] for item in self.training_data["data"]]
            
            # If we have training data
            if train_messages:
                try:
                    # Transform training messages and the current message
                    tfidf_matrix = vectorizer.fit_transform(train_messages + [message])
                    
                    # Calculate similarity between the current message and all training messages
                    message_vector = tfidf_matrix[-1]
                    train_vectors = tfidf_matrix[:-1]
                    
                    # Calculate cosine similarity
                    similarities = cosine_similarity(message_vector, train_vectors)[0]
                    
                    # Find the most similar message
                    most_similar_idx = similarities.argmax()
                    similarity_score = similarities[most_similar_idx]
                    
                    # If similarity is high enough, use the category of the most similar message
                    if similarity_score > 0.5:
                        similar_category = self.training_data["data"][most_similar_idx]["category"]
                        self.add_debug_message(f"Found similar message with {similarity_score:.2f} similarity score in category '{similar_category}'")
                        
                        # Add this message to training data
                        self.update_training_data(message, similar_category)
                        
                        # Return a response for this category
                        if similar_category in self.responses:
                            return random.choice(self.responses[similar_category])
                except Exception as e:
                    print(f"Error in similarity calculation: {e}")
            
            # If all else fails, return an unknown response
            return random.choice(self.responses["unknown"])
        
        # Default response if category is not found
        return random.choice(self.responses["unknown"])

    def add_user_message(self, message):
        """Add a user message to the chat display"""
        self.chat_display.configure(state=tk.NORMAL)
        
        # Add some space if there are already messages
        if self.chat_display.get("1.0", tk.END) != "\n":
            self.chat_display.insert(tk.END, "\n\n")
        
        # Add user tag
        self.chat_display.insert(tk.END, "You: ", "user_header")
        self.chat_display.insert(tk.END, f"\n{message}\n", "user")
        
        # Scroll to the end
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def add_bot_message(self, message):
        """Add a bot message to the chat display"""
        self.chat_display.configure(state=tk.NORMAL)
        
        # Add some space if there are already messages
        if self.chat_display.get("1.0", tk.END) != "\n":
            self.chat_display.insert(tk.END, "\n\n")
        
        # Add bot tag and message
        self.chat_display.insert(tk.END, "AI Assistant: ", "bot_header")
        self.chat_display.insert(tk.END, f"\n{message}\n", "bot")
        
        # Scroll to the end
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
    
    def add_debug_message(self, message):
        """Add a debug message to the chat display"""
        self.chat_display.configure(state=tk.NORMAL)
        
        # Add debug message
        self.chat_display.insert(tk.END, f"\n[{message}]\n", "debug")
        
        # Scroll to the end
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def show_typing_indicator(self):
        """Show the typing indicator"""
        if not self.typing_indicator_showing:
            self.typing_indicator_showing = True
            self.chat_display.configure(state=tk.NORMAL)
            
            # Add some space if there are already messages
            if self.chat_display.get("1.0", tk.END) != "\n":
                self.chat_display.insert(tk.END, "\n\n")
            
            # Add the indicator
            self.chat_display.insert(tk.END, "AI Assistant is thinking...", "typing")
            
            # Scroll to the end
            self.chat_display.see(tk.END)
            self.chat_display.configure(state=tk.DISABLED)

    def hide_typing_indicator(self):
        """Hide the typing indicator"""
        if self.typing_indicator_showing:
            self.chat_display.configure(state=tk.NORMAL)
            
            # Find and delete the typing indicator
            content = self.chat_display.get("1.0", tk.END)
            if "AI Assistant is thinking..." in content:
                start_index = content.find("AI Assistant is thinking...")
                if start_index != -1:
                    # Calculate line and character position
                    lines = content[:start_index].count("\n")
                    line_start = content.rfind("\n", 0, start_index) + 1
                    char = start_index - line_start
                    
                    # Delete the indicator
                    start_pos = f"{lines + 1}.{char}"
                    end_pos = f"{lines + 1}.{char + len('AI Assistant is thinking...')}"
                    self.chat_display.delete(start_pos, end_pos)
            
            self.chat_display.configure(state=tk.DISABLED)
            self.typing_indicator_showing = False

    def clear_chat(self):
        """Clear the chat history"""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.add_bot_message("Hello! I'm your AI assistant with machine learning capabilities. How can I help you today?")
        
        # Clear conversation history
        self.conversation_history = []

    def show_model_stats(self):
        """Show statistics about the ML model"""
        # Count categories in training data
        category_counts = {}
        for item in self.training_data["data"]:
            category = item["category"]
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        # Create a message with statistics
        stats_message = "Machine Learning Model Statistics:\n\n"
        stats_message += f"Training examples: {len(self.training_data['data'])}\n"
        stats_message += "Category distribution:\n"
        
        for category, count in category_counts.items():
            percentage = (count / len(self.training_data["data"])) * 100
            stats_message += f"- {category}: {count} examples ({percentage:.1f}%)\n"
        
        # Show in a messagebox
        messagebox.showinfo("Model Statistics", stats_message)

    def on_closing(self):
        """Handle window close event"""
        # Save any unsaved data before closing
        with open(self.training_data_file, 'w') as f:
            json.dump(self.training_data, f, indent=4)
        
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

# To run the application
def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
