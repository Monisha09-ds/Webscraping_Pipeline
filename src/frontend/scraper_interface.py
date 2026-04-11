
###------------------------###

# src/frontend/scraper_app.py

import streamlit as st
import requests

BACKEND_URL = "http://localhost:5005"  # FastAPI

st.set_page_config(page_title="Web Scraper", page_icon="🕷️", layout="centered")
st.title("🕷️ Website Scraper")

# --- session state ---
# --- sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("This scraper recursively crawls websites and saves content as Markdown for the RAG pipeline.")
    if st.session_state.session_id:
        st.success(f"Active Session: `{st.session_state.session_id}`")
        if st.button("Clear Session"):
            st.session_state.session_id = None
            st.session_state.scrape_last = None
            st.rerun()

st.markdown("### 🕸️ Crawl Settings")
with st.form("scrape_form", clear_on_submit=False):
    url = st.text_input("Website URL to scrape", placeholder="https://example.com")
    
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("Crawl Strategy", ["bfs", "dfs"], index=0, help="BFS visits all pages at one level before moving deeper. DFS goes deep first.")
        max_depth = st.number_input("Max Recursion Depth", min_value=0, value=2, step=1, help="0 means only the starting page.")
        same_domain_only = st.checkbox("Restrict to same domain", value=True)
    
    with col2:
        max_pages_per_url = st.number_input("Max pages per URL", min_value=1, value=5, step=1, help="Cap for pagination sites.")
        max_total_nodes = st.number_input("Global node cap", min_value=1, value=50, step=1, help="Max unique URLs to visit.")
        max_total_pages = st.number_input("Global page cap", min_value=1, value=100, step=1, help="Safety limit for total pages saved.")

    submitted = st.form_submit_button("🚀 Start Scraping", use_container_width=True)

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
            with st.status("🕷️ Scraping and processing content...", expanded=True) as status:
                st.write("Initializing crawler...")
                resp = requests.post(f"{BACKEND_URL}/scrape", json=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                st.write("Processing markdown output...")
                status.update(label="Scrape Complete!", state="complete", expanded=False)
            
            st.session_state.session_id = data.get("session_id")
            st.session_state.scrape_last = data
            st.balloons()
        except Exception as e:
            st.error(f"Scrape failed: {e}")

# --- results panel ---
if st.session_state.scrape_last:
    st.divider()
    st.subheader("📊 Latest Result Summary")
    d = st.session_state.scrape_last
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages Saved", d.get('pages_saved','?'))
    c2.metric("Format", "Markdown")
    c3.metric("Status", "Success")
    
    with st.expander("Technical Details"):
        st.write(f"**Session ID:** `{d.get('session_id','')}`")
        st.write(f"**Saved Folder:** `{d.get('folder','')}`")
    
    st.success("You can now switch to the **Chat Interface** to query this content!")
    
    # Quick copy for session id
    if st.session_state.session_id:
        st.text("Copy Session ID for Chatbot:")
        st.code(st.session_state.session_id, language="text")
else:
    st.info("No scrape has been run yet. Enter a URL above to begin.")
