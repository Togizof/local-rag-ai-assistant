import os
import logging
import streamlit as st
from src.pipeline import RAGPipeline

# Page setup
st.set_page_config(
    page_title="Local RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pipeline resource to load models once
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

os.makedirs("data/docs", exist_ok=True)

st.title("Local RAG AI Assistant 🤖")
st.markdown(
    "Ask questions about your documents offline. "
    "All data stays on your computer."
)

# Initialize models
try:
    with st.spinner("Loading models... Please wait."):
        pipeline = get_pipeline()
    st.sidebar.success("Models are ready!")
except Exception as e:
    st.sidebar.error("Error loading models!")
    st.error(f"Cannot start the system. Detail: {e}")
    st.stop()

# Sidebar
st.sidebar.header("📂 Manage Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload files (.txt, .md)", 
    type=["txt", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("📚 Index Documents"):
        total_chunks = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data/docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                chunks_added = pipeline.ingest_document(file_path)
                total_chunks += chunks_added
        
        st.sidebar.success(f"Added {total_chunks} chunks.")
        st.rerun()

chunk_count = pipeline.db.count_chunks()
st.sidebar.markdown(f"**Database:** {chunk_count} chunks indexed.")

if st.sidebar.button("🗑️ Reset Database", type="primary"):
    pipeline.db.clear_database()
    for f in os.listdir("data/docs"):
        if f.endswith((".txt", ".md")):
            try:
                os.remove(os.path.join("data/docs", f))
            except OSError as e:
                logging.warning(f"Error deleting file {f}: {e}")
    st.sidebar.warning("Database cleared.")
    st.rerun()

st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Number of sources (Top-K)", min_value=1, max_value=5, value=3)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**[{idx+1}] {src['document_name']}** (Score: {src['score']:.4f})")
                    st.caption(f"\"{src['content']}\"")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if chunk_count == 0:
        with st.chat_message("assistant"):
            warning_msg = "No documents found. Please upload some files in the sidebar first."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        # Generate answer
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            stream_gen, retrieved_chunks = pipeline.query_stream(prompt, top_k=top_k, temperature=temperature)
            
            full_response = ""
            for chunk in stream_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            sources_to_save = []
            if retrieved_chunks:
                with st.expander("📚 Sources"):
                    for idx, chunk in enumerate(retrieved_chunks):
                        score = chunk.get("score", 0.0)
                        sources_to_save.append({
                            "document_name": chunk["document_name"],
                            "content": chunk["content"],
                            "score": score
                        })
                        st.markdown(f"**[{idx+1}] {chunk['document_name']}** (Score: {score:.4f})")
                        st.caption(f"\"{chunk['content']}\"")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_to_save
            })
