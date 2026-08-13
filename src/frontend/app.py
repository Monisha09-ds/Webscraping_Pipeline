# src/frontend/app.py - Single entrypoint for the whole UI.
#
# One Streamlit server, one port. The scraper is the landing page; the chat page
# is only reached when the user explicitly asks for it (sidebar nav, or the
# "Chat with this content" button the scraper shows after a successful crawl).

import streamlit as st

st.set_page_config(
    page_title="Site Intelligence",
    page_icon="🕷️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Shared session state (created once, before any page runs) ---
_DEFAULTS = {
    "session_id": None,      # deterministic id derived from the content folder
    "scrape_last": None,     # last /scrape response
    "pending_folder": None,  # folder handed from the scraper page to the chat page
    "chat_initialized": False,
    "chat_history": [],
}
for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

SCRAPER_PAGE = "scraper_interface.py"
CHAT_PAGE = "chat_interface.py"

pages = [
    st.Page(SCRAPER_PAGE, title="Scrape a site", icon="🕷️", default=True),
    st.Page(CHAT_PAGE, title="Chat with content", icon="💬"),
]

st.navigation({"Pipeline": pages}).run()
