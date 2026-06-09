"""
chatbotUI.py
------------
Streamlit front-end for the PDF RAG chatbot.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime

import streamlit as st
import importlib, sys
if "rag_backend" in sys.modules:
    del sys.modules["rag_backend"]
# ──────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="centered")

# ──────────────────────────────────────────────────────────────
# Backend import
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading database …")
def load_backend():
    """
    Import the RAG backend exactly once per process.
    Caching with cache_resource prevents Streamlit from re-importing
    (and re-initialising ChromaDB) on every rerun.
    """
    import rag_backend as _backend
    return _backend


try:
    backend = load_backend()
    db_ready = True
except Exception:
    backend = None
    db_ready = False
    _err = traceback.format_exc()

# ──────────────────────────────────────────────────────────────
# Session state bootstrap
# ──────────────────────────────────────────────────────────────
import uuid

# Fetch all available sessions for the sidebar
all_sessions = backend.get_all_sessions() if db_ready else []

# Initialize the active session_id
if "session_id" not in st.session_state:
    if all_sessions:
        st.session_state.session_id = all_sessions[0]["session_id"]
    else:
        st.session_state.session_id = str(uuid.uuid4())

# Load messages for the active session
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = (
        backend.load_chat_history_from_db(st.session_state.session_id) if db_ready else []
    )

def switch_session(new_session_id):
    st.session_state.session_id = new_session_id
    if db_ready:
        st.session_state.messages = backend.load_chat_history_from_db(new_session_id)
    else:
        st.session_state.messages = []

def create_new_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    # ── New Chat Button ──
    st.button("➕ New Chat", use_container_width=True, on_click=create_new_session, type="primary")
    st.divider()

    # ── DB status ──
    if db_ready:
        st.caption(f"✅ {backend.collection.count()} chunks in DB")
    else:
        st.error("❌ Backend failed to load")
        st.code(_err)

    # ── Chat History / Sessions List ──
    st.subheader("💬 Recent Chats")
    if all_sessions:
        for s in all_sessions:
            # Highlight the currently active session
            is_active = s["session_id"] == st.session_state.session_id
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(s["title"], key=f"btn_{s['session_id']}", use_container_width=True, type=btn_type):
                if not is_active:
                    switch_session(s["session_id"])
                    st.rerun()
    else:
        st.caption("No chat history yet.")

    st.divider()

    has_messages = bool(st.session_state.messages)

    # ── Export ──
    export_payload = json.dumps(
        {
            "exported_at": datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "messages": st.session_state.messages,
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        label="📥 Export current chat (JSON)",
        data=export_payload,
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        disabled=not has_messages,
        use_container_width=True,
    )

    # ── Clear ──
    if has_messages:
        if st.button("🗑️ Delete this chat", type="secondary", use_container_width=True):
            if db_ready:
                try:
                    backend.clear_chat_history_in_db(st.session_state.session_id)
                except Exception as e:
                    st.warning(f"DB error: {e}")
            st.session_state.messages = []
            
            # Switch to another session if available
            remaining = [s for s in all_sessions if s["session_id"] != st.session_state.session_id]
            if remaining:
                switch_session(remaining[0]["session_id"])
            else:
                create_new_session()
            st.rerun()

# ──────────────────────────────────────────────────────────────
# Main area — title + chat history
# ──────────────────────────────────────────────────────────────

st.title("📄 Ask Your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ──────────────────────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about your documents …"):
    if not db_ready:
        st.error("Backend is unavailable. Check the sidebar for details.")
        st.stop()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate + stream answer
    with st.chat_message("assistant"):
        with st.spinner("Searching …"):
            try:
                answer = backend.ask_rag(question)
            except Exception as e:
                answer = f"❌ Error: {e}"
        st.markdown(answer)

    # Persist to session + DB
    st.session_state.messages.append({"role": "assistant", "content": answer})
    try:
        backend.save_chat_to_db(st.session_state.session_id, question, answer)
    except Exception as e:
        st.warning(f"⚠️ Could not save to ChromaDB: {e}")

    # Force a rerun so the sidebar updates instantly with the new session history
    st.rerun()