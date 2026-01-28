import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a helpful assistant. Please respond to the question asked "),
        ("user" , "Question:{question}")
    ]
)

## streamlit framework - Page configuration
st.set_page_config(page_title="LLAMA3 AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5em;
            color: #1f77b4;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #666;
            margin-bottom: 30px;
        }
        .response-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .input-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🤖 LLAMA3 AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Langchain & Ollama</div>', unsafe_allow_html=True)

# Sidebar for model info
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info("Model: **LLAMA3**")
    st.markdown("---")
    st.subheader("About")
    st.write("This is an AI assistant powered by LLAMA3 model running locally via Ollama.")

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("❓ Ask Your Question")
input_text = st.text_area(
    "Enter your question here:",
    placeholder="What would you like to know?",
    height=100,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Submit button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("🚀 Get Answer", use_container_width=True)

## ollama LLAMA3 model
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

# Display response
if input_text and submit_button:
    with st.spinner("🔄 Thinking..."):
        response = chain.invoke({"question": input_text})
    st.markdown('<div class="response-container">', unsafe_allow_html=True)
    st.subheader("💡 Answer")
    st.write(response)
    st.markdown('</div>', unsafe_allow_html=True)