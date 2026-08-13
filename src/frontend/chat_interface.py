# src/frontend/chat_interface.py - RAG chat page.
# Rendered by app.py via st.navigation; do not call st.set_page_config here.

import os
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("CHAT_API_URL", "http://localhost:5003")
PROJ_ROOT = Path(__file__).resolve().parents[2]
SITECONTENT_ROOT = Path(os.getenv("SITECONTENT_ROOT") or (PROJ_ROOT / "sitecontent"))
SCRAPER_PAGE = "scraper_interface.py"

st.title("💬 Chat with Scraped Content")


@st.cache_data(ttl=10)
def get_backend_info() -> dict:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=10).json()
    except requests.RequestException:
        return {}


def get_scraped_folders() -> list[str]:
    """Folder names under sitecontent/ that actually contain markdown."""
    SITECONTENT_ROOT.mkdir(parents=True, exist_ok=True)
    return sorted(
        item.name
        for item in SITECONTENT_ROOT.iterdir()
        if item.is_dir() and any(item.rglob("*.md"))
    )


def api_error(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


folders = get_scraped_folders()

# --- Guard: nothing to chat about yet ---
if not folders:
    st.info("There is no scraped content yet. Crawl a site first, then come back here.")
    if st.button("🕷️ Go to the scraper", type="primary"):
        st.switch_page(SCRAPER_PAGE)
    st.stop()

# Carry a folder over from a fresh scrape, exactly once, so it arrives preselected
# without overriding a choice the user makes later.
pending = st.session_state.get("pending_folder")
if pending and pending in folders:
    st.session_state.selected_folder = pending
    st.session_state.pending_folder = None

info = get_backend_info()
model_choices = info.get("llm_model_choices") or [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📁 Content")
    st.caption(f"Backend: `{BACKEND_URL}`")
    if info:
        st.caption(f"Embeddings: `{info.get('embed_model', '?')}`")

    selected_folder = st.selectbox(
        "Scraped content folder", options=[""] + folders, key="selected_folder"
    )

    if selected_folder:
        folder_path = SITECONTENT_ROOT / selected_folder
        ingested = False

        try:
            resp = requests.get(
                f"{BACKEND_URL}/check_session", params={"folder": str(folder_path)}, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            ingested = bool(data.get("exists"))
            if ingested:
                st.success("Embeddings ready.")
            else:
                st.info("Not ingested yet — click **Ingest** below.")
        except requests.RequestException as e:
            st.error(f"Could not reach the chat API: {e}")

        st.divider()
        st.header("⚙️ Ingest")
        force = st.checkbox("Re-ingest from scratch", value=False)
        if st.button("Ingest with embeddings", use_container_width=True):
            try:
                with st.spinner("Embedding content (the first run downloads the model)..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/ingest",
                        json={"folder": str(folder_path), "force": force},
                        timeout=1800,
                    )
                if resp.status_code >= 400:
                    st.error(f"Ingest failed: {api_error(resp)}")
                else:
                    data = resp.json()
                    st.session_state.session_id = data["session_id"]
                    st.success(f"{data['message']} ({data.get('chunks', '?')} chunks)")
                    st.rerun()
            except requests.RequestException as e:
                st.error(f"Ingest failed: {e}")

        st.divider()
        st.header("🤖 Model")
        model_name = st.selectbox("Groq model", model_choices, index=0)
        api_key = st.text_input(
            "Groq API key (optional)",
            type="password",
            help="Leave blank to use the server's GROQ_API_KEY.",
        )

        if st.button("Initialize chat", type="primary", use_container_width=True):
            try:
                with st.spinner("Connecting..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/chat/init",
                        json={
                            "session_id": st.session_state.session_id,
                            "model_type": "groq",
                            "model_name": model_name,
                            "api_key": api_key or None,
                        },
                        timeout=300,
                    )
                if resp.status_code >= 400:
                    st.error(f"Chat init failed: {api_error(resp)}")
                else:
                    data = resp.json()
                    st.session_state.chat_initialized = True
                    st.success(f"Ready — {data['model']} over {data['chunks']} chunks.")
            except requests.RequestException as e:
                st.error(f"Chat init failed: {e}")

    if st.session_state.chat_initialized:
        st.divider()
        st.subheader("📊 Session")
        st.caption(f"ID: `{st.session_state.session_id}`")
        st.caption(f"Messages: {len(st.session_state.chat_history)}")
        if st.button("Reset chat history"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    if st.button("🕷️ Scrape another site"):
        st.switch_page(SCRAPER_PAGE)

# =========================
# Main
# =========================
if not st.session_state.get("selected_folder"):
    st.info("Pick a scraped folder in the sidebar to begin.")
elif not st.session_state.chat_initialized:
    st.info("Ingest the folder, then click **Initialize chat** in the sidebar.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

    if prompt := st.chat_input("Ask something about the scraped content..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt.strip()},
                    timeout=300,
                )
            if resp.status_code >= 400:
                st.error(f"Chat failed: {api_error(resp)}")
            else:
                response_text = resp.json().get("response", "")
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.chat_history.append(
                    {"user": prompt.strip(), "assistant": response_text}
                )
        except requests.RequestException as e:
            st.error(f"Chat failed: {e}")
