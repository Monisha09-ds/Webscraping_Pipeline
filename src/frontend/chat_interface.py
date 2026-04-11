# # Streamlit frontend with chat interface (cleaned)

# import streamlit as st
# import requests
# import uuid
# import os
# from pathlib import Path

# # FastAPI backend URL
# BACKEND_URL = "http://localhost:5003"

# st.title("Web Scraper and Chat Pipeline")

# # --- Session state ---
# if "session_id" not in st.session_state:
#     st.session_state.session_id = None
# if "chat_initialized" not in st.session_state:
#     st.session_state.chat_initialized = False
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []  # list of {"user": "...", "assistant": "..."}

# # --- Helper function to get available scraped folders ---
# def get_scraped_folders(sitecontent_root="/home/aiteam/Desktop/website_content_scrapper/webscraper_pipeline/sitecontent"):
#     sitecontent_path = Path(sitecontent_root)
#     folders = []
#     for item in sitecontent_path.iterdir():
#         if item.is_dir() and any(item.rglob("*.md")):
#             folders.append(item)
#     return [f.name for f in folders]

# # =========================
# # Sidebar: Ingest / Chat Init
# # =========================
# with st.sidebar:
#     st.header("Content Selection")
#     scraped_folders = get_scraped_folders()
#     selected_folder = st.selectbox("Select Scraped Content Folder", options=[""] + scraped_folders, key="selected_folder")

#     if selected_folder:
#         folder_path = Path("/home/aiteam/Desktop/website_content_scrapper/webscraper_pipeline/sitecontent") / selected_folder
#         st.write(f"Selected Folder: {folder_path}")

#         # Check if embeddings exist and set session_id
#         try:
#             resp = requests.get(f"{BACKEND_URL}/check_session", params={"folder": str(folder_path)})
#             resp.raise_for_status()
#             session_data = resp.json()
#             if session_data.get("exists"):
#                 st.session_state.session_id = session_data.get("session_id")
#                 st.info(f"Embeddings already exist. Session ID: `{st.session_state.session_id}`")
#             else:
#                 st.session_state.session_id = str(uuid.uuid4())
#                 st.write(f"New Session ID: `{st.session_state.session_id}` (Embeddings will be created)")
#         except Exception as e:
#             st.error(f"Failed to check session: {e}")

#         st.header("Ingest Settings")
#         if st.session_state.session_id:
#             if st.button("Ingest with Embeddings", key="btn_ingest"):
#                 try:
#                     payload = {"session_id": st.session_state.session_id, "folder": str(folder_path)}
#                     resp = requests.post(
#                         f"{BACKEND_URL}/ingest", json=payload
#                     )
#                     resp.raise_for_status()
#                     data = resp.json()
#                     st.session_state.session_id = data.get("session_id", st.session_state.session_id)  # Sync session_id
#                     if data.get("message") == "Embeddings already exist":
#                         st.info(data["message"])
#                     else:
#                         st.success(data["message"])
#                 except Exception as e:
#                     st.error(f"Ingest failed: {e}")

#         st.header("Chat Initialization")
#         if st.session_state.session_id:
#             init_model_type = st.selectbox(
#                 "Select Model Type", ["local", "api"], key="init_model_type"
#             )
#             init_model_name = None
#             if init_model_type == "api":
#                 init_model_name = st.text_input(
#                     "Gemini model", value="gemini-1.5-flash", key="init_model_name"
#                 )

#             if st.button("Initialize Chat", key="btn_chat_init"):
#                 try:
#                     payload = {
#                         "session_id": st.session_state.session_id,
#                         "model_type": init_model_type,
#                         "model_name": init_model_name,
#                     }
#                     resp = requests.post(f"{BACKEND_URL}/chat/init", json=payload)
#                     resp.raise_for_status()
#                     st.session_state.chat_initialized = True
#                     st.success("Chat initialized successfully!")
#                 except Exception as e:
#                     st.error(f"Chat init failed: {e}")
#         else:
#             st.info("Please select a folder to enable chat initialization.")

# # =========================
# # Main: Chat Interface
# # =========================
# st.header("Chat Interface")
# if not st.session_state.session_id:
#     st.warning("Please select a folder to start chatting.")
# elif not st.session_state.chat_initialized:
#     st.info("Initialize chat in the sidebar to start chatting.")
# else:
#     if st.session_state.chat_history:
#         st.write("### Conversation")
#         for i, turn in enumerate(st.session_state.chat_history, 1):
#             st.markdown(f"**You {i}:** {turn['user']}")
#             st.markdown(f"**Assistant {i}:** {turn['assistant']}")

#     user_msg = st.text_input("Enter your message", key="chat_msg_input")
#     if st.button("Send", key="btn_chat_send"):
#         if not user_msg.strip():
#             st.warning("Type a message first.")
#         else:
#             try:
#                 payload = {
#                     "session_id": st.session_state.session_id,
#                     "message": user_msg.strip(),
#                 }
#                 resp = requests.post(f"{BACKEND_URL}/chat", json=payload)
#                 resp.raise_for_status()
#                 data = resp.json()
#                 st.session_state.chat_history.append(
#                     {"user": user_msg.strip(), "assistant": data.get("response", "")}
#                 )
#                 st.session_state.chat_msg_input = ""
#                 st.experimental_rerun()
#             except Exception as e:
#                 st.error(f"Chat failed: {e}")



####----------------------####

import streamlit as st
import requests
import uuid
from pathlib import Path

# FastAPI backend URL
BACKEND_URL = "http://localhost:5003"

st.title("Web Scraper and Chat Pipeline")

# --- Session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"user": "...", "assistant": "..."}

# --- Helper function to get available scraped folders ---
def get_scraped_folders():
    # Detect project root relative to this file (src/frontend/chat_interface.py)
    PROJ_ROOT = Path(__file__).resolve().parents[2]
    sitecontent_path = PROJ_ROOT / "sitecontent"
    
    if not sitecontent_path.exists():
        sitecontent_path.mkdir(parents=True, exist_ok=True)
        
    folders = []
    for item in sitecontent_path.iterdir():
        if item.is_dir() and any(item.rglob("*.md")):
            folders.append(item)
    return [f.name for f in folders]

# Helper function to generate deterministic session ID based on folder path
def session_id_for_folder(folder: Path) -> str:
    """Generate a stable session_id based on the folder path."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(folder.resolve())))

# =========================
# Sidebar: Ingest / Chat Init
# =========================
with st.sidebar:
    st.header("Content Selection")
    scraped_folders = get_scraped_folders()
    selected_folder = st.selectbox("Select Scraped Content Folder", options=[""] + scraped_folders, key="selected_folder")

    if selected_folder:
        PROJ_ROOT = Path(__file__).resolve().parents[2]
        folder_path = PROJ_ROOT / "sitecontent" / selected_folder
        st.write(f"Selected Folder: {folder_path}")

        # Check if embeddings exist and set session_id
        try:
            resp = requests.get(f"{BACKEND_URL}/check_session", params={"folder": str(folder_path)})
            resp.raise_for_status()
            session_data = resp.json()
            if session_data.get("exists"):
                st.session_state.session_id = session_data.get("session_id")
                st.info(f"Embeddings already exist. Session ID: `{st.session_state.session_id}`")
            else:
                st.session_state.session_id = session_id_for_folder(folder_path)  # Use deterministic session ID
                st.write(f"New Session ID: `{st.session_state.session_id}` (Embeddings will be created)")
        except Exception as e:
            st.error(f"Failed to check session: {e}")

        st.header("Ingest Settings")
        if st.session_state.session_id:
            if st.button("Ingest with Embeddings", key="btn_ingest"):
                try:
                    payload = {"session_id": st.session_state.session_id, "folder": str(folder_path)}
                    resp = requests.post(
                        f"{BACKEND_URL}/ingest", json=payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.session_id = data.get("session_id", st.session_state.session_id)  # Sync session_id
                    if data.get("message") == "Embeddings already exist":
                        st.info(data["message"])
                    else:
                        st.success(data["message"])
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

        st.header("Chat Initialization")
        if st.session_state.session_id:
            init_model_type = st.selectbox(
                "Select Model Type", ["local", "api"], key="init_model_type"
            )
            init_model_name = None
            init_api_key = None
            if init_model_type == "api":
                init_model_name = st.text_input(
                    "Gemini model", value="gemini-1.5-flash", key="init_model_name"
                )
                init_api_key = st.text_input(
                    "Optional: Gemini API Key", type="password", help="Leave blank to use server-side key", key="init_api_key"
                )

            if st.button("Initialize Chat", key="btn_chat_init"):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "model_type": init_model_type,
                        "model_name": init_model_name,
                        "api_key": init_api_key if init_api_key else None
                    }
                    resp = requests.post(f"{BACKEND_URL}/chat/init", json=payload)
                    resp.raise_for_status()
                    st.session_state.chat_initialized = True
                    st.success("Chat initialized successfully!")
                except Exception as e:
                    st.error(f"Chat init failed: {e}")
        else:
            st.info("Please select a folder to enable chat initialization.")

# =========================
# Main: Chat Interface
# =========================
# st.header("Chat Interface")
# if not st.session_state.session_id:
#     st.warning("Please select a folder to start chatting.")
# elif not st.session_state.chat_initialized:
#     st.info("Initialize chat in the sidebar to start chatting.")
# else:
#     if st.session_state.chat_history:
#         st.write("### Conversation")
#         for i, turn in enumerate(st.session_state.chat_history, 1):
#             st.markdown(f"**You {i}:** {turn['user']}")
#             st.markdown(f"**Assistant {i}:** {turn['assistant']}")

#     user_msg = st.text_input("Enter your message", key="chat_msg_input")
#     if st.button("Send", key="btn_chat_send"):
#         if not user_msg.strip():
#             st.warning("Type a message first.")
#         else:
#             try:
#                 payload = {
#                     "session_id": st.session_state.session_id,
#                     "message": user_msg.strip(),
#                 }
#                 resp = requests.post(f"{BACKEND_URL}/chat", json=payload)
#                 resp.raise_for_status()
#                 data = resp.json()
#                 st.session_state.chat_history.append(
#                     {"user": user_msg.strip(), "assistant": data.get("response", "")}
#                 )
#                 st.session_state.chat_msg_input = ""
#                 st.experimental_rerun()
#             except Exception as e:
#                 st.error(f"Chat failed: {e}")




# Main: Chat Interface
st.header("Chat Interface")

if not st.session_state.session_id:
    st.warning("Create a session by scraping a site first.")
elif not st.session_state.chat_initialized:
    st.info("Initialize chat in the sidebar to start chatting.")
else:
    # --- Display Sidebar Stats ---
    with st.sidebar:
        st.divider()
        st.subheader("📊 Session Stats")
        st.write(f"**Session ID:** `{st.session_state.session_id}`")
        st.write(f"**Messages:** {len(st.session_state.chat_history)}")
        if st.button("Reset Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # --- Display Chat History ---
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

    # --- Chat Input ---
    if prompt := st.chat_input("Ask something about the scraped content..."):
        # Add user message to UI immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            with st.spinner("Thinking..."):
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt.strip(),
                }
                resp = requests.post(f"{BACKEND_URL}/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("response", "")
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response_text)
            
            # Save to history
            st.session_state.chat_history.append(
                {"user": prompt.strip(), "assistant": response_text}
            )
            
        except Exception as e:
            st.error(f"Chat failed: {e}")

