import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from src.embedder import get_or_create_vector_store
from src.llm_handler import retrieve_and_rerank, get_llm_chain, generate_rag_response_lcel

st.set_page_config(page_title="Shopify AI Support", page_icon="🛍️", layout="centered")
st.title("🛍️ Shopify Returns & Exchanges AI")
st.markdown("Powered by **Llama-3.1 8B**, **BGE-Reranker**, and **ChromaDB**")

# Cache the vector store and LLM chain so they don't reload on every user interaction
@st.cache_resource
def init_systems():
    # Path is relative to the root folder where app.py is located
    vs = get_or_create_vector_store(persist_dir="vector_store_bge")
    chain = get_llm_chain()
    return vs, chain

vectorstore, rag_chain = init_systems()

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Display previous chat messages
for msg in st.session_state.chat_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input block
if user_query := st.chat_input("Ask about Shopify returns..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and re-ranking..."):
            try:
                matched_docs = retrieve_and_rerank(user_query, vectorstore)
                response = generate_rag_response_lcel(
                    query=user_query,
                    sorted_docs=matched_docs,
                    chat_history=st.session_state.chat_history,
                    rag_chain=rag_chain
                )
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
