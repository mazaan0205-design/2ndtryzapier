import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

# Document Processing
from PyPDF2 import PdfReader

load_dotenv()

st.set_page_config(page_title="W3S Builder - RAG Agent", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- FUNCTIONS ---
def process_files(uploaded_files):
    """Processes files and updates the Chroma Vector Store."""
    text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:
            text += file.read().decode("utf-8")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    
    # Create Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(
        texts=splits, 
        embedding=embeddings, 
        collection_name="w3s_collection"
    )
    return vector_store

# --- LEFT SIDEBAR ---
with st.sidebar:
    st.title("🛠️ W3S Builder")
    
    # Token Usage Metrics
    st.subheader("📊 Usage Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Tokens", st.session_state.total_tokens)
    if "last_token_usage" in st.session_state:
        col2.metric("Last Query", st.session_state.last_token_usage)

    st.divider()
    
    st.subheader("1. Instructions")
    instructions = st.text_area(
        "Define Agent Rules:", 
        value="You are a helpful assistant. Use the provided tools to search the knowledge base before answering.", 
        height=200
    )
    
    st.divider()
    
    st.subheader("2. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload reference files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Sync Knowledge Base"):
        if uploaded_files:
            with st.spinner("Indexing documents..."):
                st.session_state.vector_store = process_files(uploaded_files)
                st.success("Vector Database Updated!")
        else:
            st.warning("Please upload files first.")

# --- MAIN SCREEN ---
st.title("🤖 W3S RAG Workspace")

# Display history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle Chat
if user_input := st.chat_input("Ask about your documents..."):
    st.chat_message("user").markdown(user_input)
    
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ No API Key found!")
        st.stop()

    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
            
            # Setup Tools
            tools = []
            if st.session_state.vector_store:
                retriever = st.session_state.vector_store.as_retriever()
                tool = create_retriever_tool(
                    retriever, 
                    "knowledge_base_search",
                    "Search for information about the uploaded documents."
                )
                tools.append(tool)

            # Initialize Agent
            prompt = hub.pull("hwchase17/openai-functions-agent")
            # Injecting system instructions
            prompt.messages[0].content = instructions
            
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

            # Invoke Agent
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.messages
            })
            
            answer = response["output"]
            st.markdown(answer)
            
            # --- TOKEN TRACKING ---
            # Groq returns metadata in the response object
            # Note: Usage extraction can vary by langchain-groq version
            usage = getattr(llm, "last_metadata", {}).get("usage", {})
            current_usage = usage.get("total_tokens", 0)
            
            st.session_state.last_token_usage = current_usage
            st.session_state.total_tokens += current_usage
            
            # Update History
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(AIMessage(content=answer))
            
            # Force refresh to update sidebar metrics
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")