


###-------------###
# Streamlit frontend with chat interface (cleaned)
import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://localhost:5001"

st.title("Web Scraper and Chat Pipeline")

# --- Session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"user": "...", "assistant": "..."}

# =========================
# Sidebar: Scrape / Ingest / Chat Init
# =========================
with st.sidebar:
    st.header("Scraper Settings")
    url = st.text_input("Enter Website URL to Scrape", key="scrape_url")
    strategy = st.selectbox("Crawl Strategy", ["bfs", "dfs"], index=0, key="scrape_strategy")
    max_depth = st.number_input("Max Recursion Depth", min_value=0, value=2, key="scrape_max_depth")
    max_pages_per_url = st.number_input("Max Pages per URL", min_value=1, value=5, key="scrape_max_pages_per_url")
    max_total_nodes = st.number_input("Max Total Nodes", min_value=1, value=100, key="scrape_max_total_nodes")
    max_total_pages = st.number_input("Max Total Pages", min_value=1, value=200, key="scrape_max_total_pages")
    same_domain_only = st.checkbox("Restrict to Same Domain", value=True, key="scrape_same_domain_only")

    if st.button("Scrape", key="btn_scrape"):
        if url:
            try:
                payload = {
                    "url": url,
                    "strategy": strategy,
                    "max_depth": int(max_depth),
                    "max_pages_per_url": int(max_pages_per_url),
                    "max_total_nodes": int(max_total_nodes),
                    "max_total_pages": int(max_total_pages),
                    "same_domain_only": bool(same_domain_only),
                }
                resp = requests.post(f"{BACKEND_URL}/scrape", json=payload)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.success(
                    f"Scraped successfully! "
                    f"Session ID: {st.session_state.session_id}, "
                    f"Pages Saved: {data['pages_saved']}, "
                    f"Folder: {data['folder']}"
                )
                # Reset chat state on new scrape
                st.session_state.chat_initialized = False
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Scrape failed: {e}")
        else:
            st.warning("Please enter a URL")

    st.header("Ingest Settings")
    if st.session_state.session_id:
        st.write(f"Current Session ID: `{st.session_state.session_id}`")
        if st.button("Ingest with Embeddings", key="btn_ingest"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ingest", json={"session_id": st.session_state.session_id}
                )
                resp.raise_for_status()
                st.success("Ingested successfully with embeddings!")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
    else:
        st.info("Please scrape first to enable ingest.")

    st.header("Chat Initialization")
    if st.session_state.session_id:
        # ✅ unique keys; no duplicates elsewhere
        init_model_type = st.selectbox(
            "Select Model Type", ["local", "api"], key="init_model_type"
        )
        init_model_name = None
        if init_model_type == "api":
            init_model_name = st.text_input(
                "Gemini model", value="gemini-1.5-flash", key="init_model_name"
            )

        if st.button("Initialize Chat", key="btn_chat_init"):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "model_type": init_model_type,
                    "model_name": init_model_name,
                }
                resp = requests.post(f"{BACKEND_URL}/chat/init", json=payload)
                resp.raise_for_status()
                st.session_state.chat_initialized = True
                st.success("Chat initialized successfully!")
            except Exception as e:
                st.error(f"Chat init failed: {e}")
    else:
        st.info("Please scrape and ingest data first to enable chat initialization.")

# =========================
# Main: Chat Interface
# =========================
st.header("Chat Interface")
if not st.session_state.session_id:
    st.warning("Create a session by scraping a site first.")
elif not st.session_state.chat_initialized:
    st.info("Initialize chat in the sidebar to start chatting.")
else:
    # Display conversation history (simple)
    if st.session_state.chat_history:
        st.write("### Conversation")
        for i, turn in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**You {i}:** {turn['user']}")
            st.markdown(f"**Assistant {i}:** {turn['assistant']}")

    # Input & send
    user_msg = st.text_input("Enter your message", key="chat_msg_input")
    if st.button("Send", key="btn_chat_send"):
        if not user_msg.strip():
            st.warning("Type a message first.")
        else:
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": user_msg.strip(),
                }
                resp = requests.post(f"{BACKEND_URL}/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Update local history so UI stays in sync even if you refresh
                st.session_state.chat_history.append(
                    {"user": user_msg.strip(), "assistant": data.get("response", "")}
                )
                # Clear the input
                st.session_state.chat_msg_input = ""
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Chat failed: {e}")
