
###------------------------###

# src/frontend/scraper_app.py

import streamlit as st
import requests

BACKEND_URL = "http://localhost:5005"  # FastAPI

st.set_page_config(page_title="Web Scraper", page_icon="🕷️", layout="centered")
st.title("🕷️ Website Scraper")

# --- session state ---
if "scrape_last" not in st.session_state:
    st.session_state.scrape_last = None   # last scrape response
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.markdown("Enter a URL and crawl settings, then hit **Scrape**.")

with st.form("scrape_form", clear_on_submit=False):
    url = st.text_input("Website URL to scrape", placeholder="https://example.com")
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("Crawl Strategy", ["bfs", "dfs"], index=0, key="scrape_strategy_only")
        max_depth = st.number_input("Max Recursion Depth", min_value=0, value=2, step=1, key="scrape_depth_only")
        same_domain_only = st.checkbox("Restrict to same domain", value=True, key="scrape_scope_only")
    with col2:
        max_pages_per_url = st.number_input("Max pages per URL", min_value=1, value=5, step=1, key="scrape_ppu_only")
        max_total_nodes = st.number_input("Global node cap", min_value=1, value=100, step=1, key="scrape_nodes_only")
        max_total_pages = st.number_input("Global page cap", min_value=1, value=200, step=1, key="scrape_pages_only")

    submitted = st.form_submit_button("Scrape", use_container_width=True)

if submitted:
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        try:
            payload = {
                "url": url.strip(),
                "strategy": strategy,
                "max_depth": int(max_depth),
                "max_pages_per_url": int(max_pages_per_url),
                "max_total_nodes": int(max_total_nodes),
                "max_total_pages": int(max_total_pages),
                "same_domain_only": bool(same_domain_only),
            }
            with st.spinner("Scraping… this may take a moment"):
                resp = requests.post(f"{BACKEND_URL}/scrape", json=payload, timeout=300)
                resp.raise_for_status()
                data = resp.json()
            st.session_state.session_id = data.get("session_id")
            st.session_state.scrape_last = data
            st.success("Scrape complete ✅")
        except Exception as e:
            st.error(f"Scrape failed: {e}")

# --- results panel ---
st.divider()
st.subheader("Latest Result")
if st.session_state.scrape_last:
    d = st.session_state.scrape_last
    st.markdown(f"**Session ID:** `{d.get('session_id','')}`")
    st.markdown(f"**Pages Saved:** {d.get('pages_saved','?')}")
    st.markdown(f"**Saved Folder:** `{d.get('folder','')}`")
    st.info("Use this Session ID for subsequent steps in your other UIs (ingest/chat).")
else:
    st.write("No scrape has been run yet.")

# (Optional) quick copy button for Session ID
if st.session_state.session_id:
    st.code(st.session_state.session_id, language="text")
