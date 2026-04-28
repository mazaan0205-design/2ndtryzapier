import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from PyPDF2 import PdfReader

# 1. Start by looking for local .env file (for your local PC safety)
load_dotenv()

st.set_page_config(page_title="W3S Builder", layout="wide")

# --- LEFT SIDEBAR (The "Builder" Controls) ---
with st.sidebar:
    st.title("🛠️ W3S Builder")
    
    st.subheader("1. Instructions")
    # This is where you write the 'Brain' rules on the left
    instructions = st.text_area(
        "Define Agent Rules:", 
        value="You are a helpful assistant.", 
        height=300
    )
    
    st.divider()
    
    st.subheader("2. Knowledge Base")
    # This is where you upload files for the bot to read
    uploaded_files = st.file_uploader(
        "Upload reference files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    knowledge_text = ""
    if uploaded_files:
        for file in uploaded_files:
            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    knowledge_text += page.extract_text()
            else:
                knowledge_text += file.read().decode("utf-8")
        st.success(f"Synced {len(uploaded_files)} files.")

# --- MAIN SCREEN (The Chat) ---
st.title("🤖 W3S Personal Workspace")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle Chat
if user_input := st.chat_input("Test your configuration..."):
    st.chat_message("user").markdown(user_input)
    
    # We pull the API Key from the system environment (No secrets typed here!)
    # When you deploy, you just add GROQ_API_KEY to the Streamlit dashboard
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error("⚠️ No API Key found! Please add GROQ_API_KEY to your environment or secrets.")
        st.stop()

    # Create the 'Master Prompt'
    full_system_prompt = f"{instructions}\n\nKNOWLEDGE CONTEXT:\n{knowledge_text}"

    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
            
            # Combine Instructions + Memory + Current Question
            payload = [SystemMessage(content=full_system_prompt)] + st.session_state.messages + [HumanMessage(content=user_input)]
            
            response = llm.invoke(payload)
            st.markdown(response.content)
            
            # Update History
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(AIMessage(content=response.content))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")