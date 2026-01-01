import streamlit as st
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# NLTK SAFE DOWNLOADS
# -------------------------
def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_resources()

# -------------------------
# INTENTS DATA
# -------------------------
intents = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning"],
        "responses": ["Hello! Ask me any programming question.", "Hi! How can I help with coding?"]
    },

    "python_basics": {
        "patterns": ["what is python", "python basics", "why use python", "features of python"],
        "responses": ["Python is a high-level, interpreted language known for simplicity and readability."]
    },

    "java_basics": {
        "patterns": ["what is java", "java basics", "features of java"],
        "responses": ["Java is an object-oriented, platform-independent programming language."]
    },

    "oops": {
        "patterns": ["what is oops", "oops concepts", "object oriented programming"],
        "responses": ["OOPS includes Encapsulation, Inheritance, Polymorphism, and Abstraction."]
    },

    "data_structures": {
        "patterns": ["what are data structures", "types of data structures", "stack queue linked list"],
        "responses": ["Common data structures include Array, Stack, Queue, Linked List, Tree, and Graph."]
    },

    "errors": {
        "patterns": ["error in code", "syntax error", "runtime error", "exception"],
        "responses": ["Errors occur due to syntax mistakes, logical issues, or runtime problems."]
    },

    "thanks": {
        "patterns": ["thanks", "thank you"],
        "responses": ["You're welcome!", "Happy coding!"]
    },

    "goodbye": {
        "patterns": ["bye", "exit", "quit"],
        "responses": ["Goodbye! Keep practicing coding."]
    }
}
language_info = {
    "python": {
        "for_loop": (
            "Python for loop syntax:\n\n"
            "for variable in iterable:\n"
            "    statements\n\n"
            "Example:\n"
            "for i in range(5):\n"
            "    print(i)"
        ),
        "while_loop": (
            "Python while loop syntax:\n\n"
            "while condition:\n"
            "    statements"
        ),
        "if_statement": (
            "Python if statement syntax:\n\n"
            "if condition:\n"
            "    statements\nelif another_condition:\n    statements\nelse:\n    statements"
        ),
        "features": "Python is interpreted, dynamically typed, and easy to learn.",
        "uses": "Python is used in web development, data science, AI, and automation."
    },

    "java": {
        "for_loop": (
            "Java for loop syntax:\n\n"
            "for(initialization; condition; increment) {\n"
            "    statements;\n"
            "}"
        ),
        "while_loop": (
            "Java while loop syntax:\n\n"
            "while(condition) {\n"
            "    statements;\n"
            "}"
        ),
        "if_statement": (
            "Java if statement syntax:\n\n"
            "if(condition) {\n"
            "    statements;\n"
            "} else if (another_condition) {\n"
            "    statements;\n"
            "} else {\n"
            "    statements;\n"
            "}"
        ),
        "features": "Java is object-oriented, platform-independent, and strongly typed.",
        "uses": "Java is used in enterprise applications, Android apps, and backend systems."
    },

    "c": {
        "for_loop": (
            "C for loop syntax:\n\n"
            "for(initialization; condition; increment) {\n"
            "    statements;\n"
            "}"
        ),
        "while_loop": (
            "C while loop syntax:\n\n"
            "while(condition) {\n"
            "    statements;\n"
            "}"
        ),
        "if_statement": (
            "C if statement syntax:\n\n"
            "if(condition) {\n"
            "    statements;\n"
            "} else if (another_condition) {\n"
            "    statements;\n"
            "} else {\n"
            "    statements;\n"
            "}"
        ),
        "features": "C is a procedural, fast, low-level programming language.",
        "uses": "C is used in system programming, embedded systems, and OS development."
    }
}



# -------------------------
# NLP PREPARATION
# -------------------------
lemmatizer = WordNetLemmatizer()

X, y = [], []

for intent, data in intents.items():
    for pattern in data["patterns"]:
        X.append(pattern)
        y.append(intent)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

X = [preprocess(sentence) for sentence in X]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vectors = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectors, y)

# -------------------------
# FALLBACK KEYWORD LOGIC
# -------------------------
def keyword_response(text):
    text = text.lower()

    if "for loop" in text:
        return "A for loop iterates over a sequence.\nExample:\nfor i in range(5): print(i)"

    if "while loop" in text:
        return "A while loop runs as long as a condition is true."

    if "function" in text:
        return "A function is a reusable block of code that performs a task."

    if "list" in text and "python" in text:
        return "A Python list is mutable.\nExample:\nmy_list = [1, 2, 3]"

    if "dictionary" in text:
        return "A dictionary stores key-value pairs.\nExample:\nd = {'a': 1}"

    if "class" in text:
        return "A class is a blueprint for creating objects."

    return None

def comparison_response(text):
    text = text.lower()
    
    # Difference questions
    if "distinguish" in text or "difference" in text or "compare" in text:
        if "python" in text and "java" in text:
            return (
                "Difference between Python and Java:\n\n"
                "Python:\n"
                "- Interpreted\n"
                "- Dynamically typed\n"
                "- Shorter syntax\n\n"
                "Java:\n"
                "- Compiled + Interpreted\n"
                "- Statically typed\n"
                "- Verbose syntax"
            )
        if "java" in text and "c" in text:
            return (
                "Difference between Java and C:\n\n"
                "Java:\n"
                "- Object-oriented\n"
                "- Platform independent\n\n"
                "C:\n"
                "- Procedural\n"
                "- Platform dependent"
            )
    
    # Ease of learning / better language questions
    if "easier" in text or "easy" in text or "better" in text:
        if "python" in text and "java" in text:
            return (
                "Which is easier?\n\n"
                "Python is generally considered easier to learn than Java because:\n"
                "- Simple and readable syntax\n"
                "- Less boilerplate code\n"
                "- Dynamically typed\n\n"
                "Java is more verbose and requires understanding of OOP concepts early."
            )
        if "java" in text and "c" in text:
            return (
                "Ease of learning:\n\n"
                "Java is generally easier than C because it handles memory management automatically and has simpler syntax.\n"
                "C requires manual memory management and deeper understanding of low-level concepts."
            )
    
    return None

# -------------------------
# CHATBOT RESPONSE
# -------------------------
def chatbot_response(user_input):
    text = user_input.lower()

    # -------- 1. COMPARISON QUESTIONS --------
    comp = comparison_response(text)
    if comp:
        return comp

    # -------- 2. LANGUAGE + SYNTAX QUESTIONS --------
    for lang, info in language_info.items():
        if lang in text:
            if "for loop" in text:
                return info.get("for_loop")
            if "while loop" in text:
                return info.get("while_loop")
            if "if" in text or "else" in text or "elif" in text or "conditional" in text:
                return info.get("if_statement")
            if "features" in text:
                return info.get("features")
            if "uses" in text or "applications" in text:
                return info.get("uses")

    # -------- 3. RULE-BASED INTENTS --------
    for intent, data in intents.items():
        for pattern in data["patterns"]:
            if pattern in text:
                return random.choice(data["responses"])

    # -------- 4. ML FALLBACK --------
    processed = preprocess(user_input)
    vector = vectorizer.transform([processed])

    intent = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    if confidence < 0.2:
        return "I'm not sure. Please ask a specific programming question."

    return random.choice(intents[intent]["responses"])




# -------------------------
# STREAMLIT UI
# -------------------------
# -------------------------
# STREAMLIT UI (WhatsApp-style chat)
# -------------------------
import streamlit as st
import random

st.set_page_config(page_title="Programming Chatbot", page_icon="💻")
st.title("💻 Programming Chatbot")
st.write("Chat with the bot directly, ask simple questions about java, python and C!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Container for chat messages
chat_placeholder = st.empty()

# Function to render chat bubbles
def render_chat():
    with chat_placeholder.container():
        for chat in st.session_state.chat_history:
            if chat["sender"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#DCF8C6;
                        padding:10px 15px;
                        border-radius:20px;
                        width:fit-content;
                        max-width:70%;
                        margin-left:auto;
                        margin-bottom:5px;
                    ">
                        🧑 You: {chat['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#F1F0F0;
                        padding:10px 15px;
                        border-radius:20px;
                        width:fit-content;
                        max-width:70%;
                        margin-right:auto;
                        margin-bottom:5px;
                    ">
                        🤖 Bot: {chat['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Input box at bottom
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="Type a message and press Enter")
    submit_button = st.form_submit_button(label="")

if submit_button and user_input:
    bot_reply = chatbot_response(user_input)

    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": bot_reply})

# Render chat
render_chat()
