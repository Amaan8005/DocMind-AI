import streamlit as st

import utils.rag_pipeline as rag
import utils.helpers as func
from components.ingestion_prerequisites import (
    ingestion_is_configured,
)

from urllib.parse import urlparse


def ensure_https(url):
    parsed = urlparse(url)

    if not bool(parsed.scheme):
        return f"https://{url}"

    return url


def add_website_from_input():
    new_website = st.session_state.get("new_website", "").strip()
    if new_website == "":
        return

    try:
        validated_website = func.validate_website_urls([ensure_https(new_website)])[0]
    except ValueError as err:
        st.session_state["website_input_error"] = str(err)
        return

    st.session_state["websites"].append(validated_website)
    st.session_state["websites"] = sorted(set(st.session_state["websites"]))
    st.session_state["new_website"] = ""
    st.session_state["website_input_error"] = None


def clear_websites():
    st.session_state["websites"] = []


def website():
    if not ingestion_is_configured():
        with st.container(border=True):
            st.caption(
                "Configure a chat model and an embedding model in Settings to "
                "enable website ingestion."
            )
            st.text_input(
                "Website URL",
                placeholder="https://docs.python.org/3/",
                disabled=True,
                key="new_website_disabled",
            )
            st.button(
                "Add URL",
                icon=":material/add_link:",
                disabled=True,
                use_container_width=True,
            )
        return

    with st.container(border=True):
        st.caption(
            "Add up to 5 public HTTPS URLs. A hostname without a scheme gets "
            "`https://` added automatically."
        )

        st.text_input(
            "Website URL",
            placeholder="https://docs.python.org/3/",
            key="new_website",
            on_change=add_website_from_input,
        )
        add_button = st.button(
            "Add URL",
            icon=":material/add_link:",
            use_container_width=True,
        )

        if add_button:
            add_website_from_input()

        if st.session_state.get("website_input_error"):
            st.error(st.session_state["website_input_error"])

        if len(st.session_state.get("websites", [])) > 0:
            st.write("")
            st.markdown("**Website(s)**")
            for site in st.session_state["websites"]:
                st.caption(f"- {site}")
            st.write("")

        col_proc, col_clr = st.columns([1, 1])
        with col_proc:
            process_button = st.button(
                "Process",
                icon=":material/task_alt:",
                key="process_website",
                use_container_width=True,
            )
        with col_clr:
            st.button(
                "Clear List",
                icon=":material/delete_sweep:",
                key="clear_websites_button",
                on_click=clear_websites,
                use_container_width=True,
            )

    if process_button:
        if st.session_state.get("new_website", "").strip():
            add_website_from_input()

        if len(st.session_state.get("websites", [])) == 0:
            st.warning("Please enter a website URL (e.g. https://docs.python.org/3/) before processing.")
            return

        status_container = st.empty()
        completed_stages = []

        with st.spinner("Processing..."):
            try:
                rag.render_pipeline_status(
                    status_container, completed_stages, "Fetching Websites"
                )
                documents = func.load_website_documents(st.session_state["websites"])
                completed_stages.append("Websites Fetched")
                rag.render_pipeline_status(status_container, completed_stages)
            except Exception as err:
                st.error(f"Failed to fetch website content: {err}")
                st.stop()

            if len(documents) > 0:
                # Initiate the RAG pipeline, providing documents to be saved on disk if necessary
                error = rag.rag_pipeline(
                    documents=documents,
                    status_container=status_container,
                    initial_stages=completed_stages,
                    status_state_key="website_ingestion_stages",
                    documents_loaded_stage="Website Content Loaded",
                )

                # Display errors (if any) or proceed
                if error is not None:
                    st.exception(error)
                else:
                    st.write("Site processing completed. Let's chat! 😎")
