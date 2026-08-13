# src/frontend/scraper_interface.py - Crawler page.
# Rendered by app.py via st.navigation; do not call st.set_page_config here.

import os
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("SCRAPER_API_URL", "http://localhost:5005")
CHAT_PAGE = "chat_interface.py"

st.title("🕷️ Website Scraper")

# --- sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(
        "Recursively crawls a website and saves each page as Markdown for the "
        "RAG pipeline."
    )
    st.caption(f"Backend: `{BACKEND_URL}`")

    if st.session_state.session_id:
        st.success(f"Active session: `{st.session_state.session_id}`")
        if st.button("Clear session"):
            st.session_state.session_id = None
            st.session_state.scrape_last = None
            st.session_state.pending_folder = None
            st.session_state.chat_initialized = False
            st.rerun()

st.markdown("### 🕸️ Crawl settings")
with st.form("scrape_form", clear_on_submit=False):
    url = st.text_input("Website URL to scrape", placeholder="https://example.com")

    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox(
            "Crawl strategy", ["bfs", "dfs"], index=0,
            help="BFS visits all pages at one level before going deeper. DFS goes deep first.",
        )
        max_depth = st.number_input(
            "Max recursion depth", min_value=0, value=2, step=1,
            help="0 means only the starting page.",
        )
        same_domain_only = st.checkbox("Restrict to same domain", value=True)

    with col2:
        max_pages_per_url = st.number_input(
            "Max pages per URL", min_value=1, value=5, step=1, help="Cap for paginated sites."
        )
        max_total_nodes = st.number_input(
            "Global node cap", min_value=1, value=50, step=1, help="Max unique URLs to visit."
        )
        max_total_pages = st.number_input(
            "Global page cap", min_value=1, value=100, step=1,
            help="Safety limit for total pages saved.",
        )

    submitted = st.form_submit_button("🚀 Start scraping", use_container_width=True)

if submitted:
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        payload = {
            "url": url.strip(),
            "strategy": strategy,
            "max_depth": int(max_depth),
            "max_pages_per_url": int(max_pages_per_url),
            "max_total_nodes": int(max_total_nodes),
            "max_total_pages": int(max_total_pages),
            "same_domain_only": bool(same_domain_only),
        }
        try:
            with st.status("🕷️ Crawling...", expanded=True) as status:
                st.write("Starting the crawler — this can take a few minutes.")
                resp = requests.post(f"{BACKEND_URL}/scrape", json=payload, timeout=900)
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                    status.update(label="Scrape failed", state="error")
                    st.error(detail)
                else:
                    data = resp.json()
                    status.update(label="Scrape complete", state="complete", expanded=False)
                    st.session_state.session_id = data.get("session_id")
                    st.session_state.scrape_last = data
                    # Hand the folder to the chat page, but do not navigate yet -
                    # the user decides whether they want to chat.
                    st.session_state.pending_folder = Path(data.get("folder", "")).name
                    st.session_state.chat_initialized = False
                    st.session_state.chat_history = []
                    st.balloons()
        except requests.RequestException as e:
            st.error(f"Could not reach the scraper API at {BACKEND_URL}: {e}")

# --- results panel ---
if st.session_state.scrape_last:
    d = st.session_state.scrape_last
    st.divider()
    st.subheader("📊 Latest result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pages saved", d.get("pages_saved", "?"))
    c2.metric("Files found", d.get("files_found", "?"))
    c3.metric("Status", "Success")

    with st.expander("Technical details"):
        st.write(f"**Session ID:** `{d.get('session_id', '')}`")
        st.write(f"**Saved folder:** `{d.get('folder', '')}`")
        st.write(f"**Collection:** `{d.get('collection_name', '')}`")

    st.divider()
    st.markdown("#### Next step")
    st.write("The content is scraped. Turn it into a searchable knowledge base whenever you want.")
    if st.button("💬 Chat with this content", type="primary", use_container_width=True):
        st.switch_page(CHAT_PAGE)
else:
    st.info("No scrape has been run yet. Enter a URL above to begin.")
