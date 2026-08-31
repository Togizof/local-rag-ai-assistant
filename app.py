import os
import logging
import streamlit as st
from src.pipeline import RAGPipeline

# Web Page Settings
st.set_page_config(
    page_title="My Library Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Gemini Inspired Theme)
st.markdown("""
<style>
    /* Styling for greeting header */
    .gemini-greeting {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #4f46e5, #3b82f6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        margin-top: 20px;
    }
    
    .gemini-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 500;
        color: #888888;
        margin-bottom: 40px;
    }

    /* Style the suggestion buttons to look like clickable cards */
    div.stButton > button {
        border: 1px solid rgba(49, 51, 63, 0.15) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        height: 120px !important;
        width: 100% !important;
        text-align: left !important;
        background-color: transparent !important;
        transition: all 0.25s ease !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    }
    
    div.stButton > button:hover {
        border-color: #3b82f6 !important;
        background-color: rgba(59, 130, 246, 0.04) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05) !important;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Cache pipeline resource to load models once
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

os.makedirs("data/docs", exist_ok=True)

# Sidebar: Book Notes Management
st.sidebar.header("📖 Book Notes Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload reading notes (.txt, .md)", 
    type=["txt", "md"], 
    accept_multiple_files=True
)

# Initialize models
try:
    with st.spinner("Loading library AI models... Please wait."):
        pipeline = get_pipeline()
    st.sidebar.success("Library models are active!")
except Exception as e:
    st.sidebar.error("Error loading library models!")
    st.error(f"Cannot start the library assistant. Detail: {e}")
    st.stop()

if uploaded_files:
    if st.sidebar.button("📚 Index / Save Book Notes"):
        total_chunks = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data/docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Indexing book note: {uploaded_file.name}..."):
                chunks_added = pipeline.ingest_document(file_path)
                total_chunks += chunks_added
        
        st.sidebar.success(f"Successfully added {total_chunks} chunks.")
        st.rerun()

doc_names = pipeline.db.get_all_document_names()
chunk_count = pipeline.db.count_chunks()

st.sidebar.markdown(f"**Library Status:** {chunk_count} passages indexed.")

# List currently loaded books
if doc_names:
    st.sidebar.markdown("### 📚 Current Books:")
    for name in doc_names:
        st.sidebar.markdown(f"- 📄 `{name}`")

if st.sidebar.button("🗑️ Clear Library DB", type="primary"):
    pipeline.db.clear_database()
    for f in os.listdir("data/docs"):
        if f.endswith((".txt", ".md")):
            try:
                os.remove(os.path.join("data/docs", f))
            except OSError as e:
                logging.warning(f"Error deleting file {f}: {e}")
    st.sidebar.warning("Library database cleared.")
    st.rerun()

# Sidebar: Reader Settings
st.sidebar.header("⚙️ Reader Settings")
top_k = st.sidebar.slider("Number of book passages (Top-K)", min_value=1, max_value=5, value=2)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# Tab Layout
tab_chat, tab_inspector = st.tabs(["💬 Reader Chat", "📊 Library Inspector"])

with tab_chat:
    # Session history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome landing screen (Gemini style) if conversation history is empty
    if not st.session_state.messages:
        st.markdown('<div class="gemini-greeting">Merhaba, Okur.</div>', unsafe_allow_html=True)
        st.markdown('<div class="gemini-subtitle">Kitaplığınız hakkında ne öğrenmek istersiniz?</div>', unsafe_allow_html=True)
        
        # Suggestion cards
        col1, col2, col3 = st.columns(3)
        clicked_prompt = None
        
        with col1:
            if st.button("📚 Kitaplığımı Özetle\n\nKütüphanemde hangi kitaplar ve notlar bulunuyor?"):
                clicked_prompt = "Neler biliyorsun?"
        with col2:
            if st.button("📖 Clean Code Kuralları\n\nKitapta fonksiyonlar hakkında ne öneriliyor?"):
                clicked_prompt = "Clean Code kitabındaki fonksiyon kuralları nedir?"
        with col3:
            if st.button("🧠 Zihin Felsefesi\n\nBilinç ve zihin felsefesi hakkında ne yazmışım?"):
                clicked_prompt = "Zihin felsefesi veya bilinç hakkında ne yazmışım?"
                
        # If card is clicked, submit it instantly
        if clicked_prompt:
            st.session_state.messages.append({"role": "user", "content": clicked_prompt})
            with st.chat_message("assistant"):
                stream_gen, retrieved_chunks = pipeline.query_stream(clicked_prompt, top_k=top_k, temperature=temperature)
                full_response = st.write_stream(stream_gen)
                
                sources_to_save = []
                if retrieved_chunks:
                    for chunk in retrieved_chunks:
                        sources_to_save.append({
                            "document_name": chunk["document_name"],
                            "content": chunk["content"],
                            "score": chunk.get("score", 0.0)
                        })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources_to_save
                })
            st.rerun()

    else:
        # Show past messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📖 Book References"):
                        for idx, src in enumerate(message["sources"]):
                            st.markdown(f"**[{idx+1}] {src['document_name']}** (Score: {src['score']:.4f})")
                            st.caption(f"\"{src['content']}\"")

    # Chat input
    if prompt := st.chat_input("Ask a question about your book notes..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if chunk_count == 0:
            with st.chat_message("assistant"):
                warning_msg = "No reading notes found. Please upload some book summaries or notes in the sidebar first."
                st.warning(warning_msg)
                st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        else:
            # Generate answer
            with st.chat_message("assistant"):
                stream_gen, retrieved_chunks = pipeline.query_stream(prompt, top_k=top_k, temperature=temperature)
                
                # Stream content naturally on the screen
                full_response = st.write_stream(stream_gen)
                
                sources_to_save = []
                if retrieved_chunks:
                    with st.expander("📖 Book References"):
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

with tab_inspector:
    st.subheader("Library Status & Information")
    st.write(f"Total indexed chunks: **{chunk_count}**")
    
    if doc_names:
        st.write("Below is the list of files currently loaded in your database. Click to inspect their details.")
        for name in doc_names:
            with st.expander(f"📄 {name}"):
                st.write(f"This file is processed and saved in the local SQLite database.")
    else:
        st.info("No files currently loaded in the database. Use the sidebar to upload files.")
