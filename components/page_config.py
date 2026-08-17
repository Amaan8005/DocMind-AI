import streamlit as st


def set_page_config():
    if "sidebar_state" not in st.session_state:
        st.session_state["sidebar_state"] = "expanded"

    st.set_page_config(
        page_title="DocMind AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state=st.session_state["sidebar_state"],
        menu_items={
            "Get Help": "https://github.com/Ashita-no-Kaushar/DocMind-AI/discussions",
            "Report a bug": "https://github.com/Ashita-no-Kaushar/DocMind-AI/issues",
        },
    )

    # Remove the Streamlit `Deploy` button from the Header
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    /* Give the sidebar more width so Data Sources rows never feel cramped */
    section[data-testid="stSidebar"] {
        min-width: 320px;
    }
    /* Breathing room between sidebar expanders */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        margin-bottom: 0.75rem;
        border-radius: 8px;
    }
    /* Hide the in-box "Press Enter to apply" overlay; the app shows this
       instruction below the input instead */
    [data-testid="InputInstructions"] {
        display: none;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
