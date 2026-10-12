
import streamlit as st
from RAG import retrieve_answer


# Configure the web application's page.
st.set_page_config(
    page_title="PolicyPilot",
    page_icon="📘",
    layout="centered"
)

st.title("📘 PolicyPilot")
st.caption("Evidence-Based AI Policy Assistant")
st.write("Ask a question about the company policy.")

# Store the user's question.
question = st.text_input(
    "Your question",
    placeholder="How many paid leave days do employees get?"
)

# Run RAG only when the user clicks the button.
if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching policy and generating answer..."):
            try:
                answer, evidence = retrieve_answer(question)

                st.subheader("Answer")
                st.write(answer)

                if evidence:
                    with st.expander("Supporting Evidence"):
                        st.text(evidence)

            except Exception:
                st.error(
                    "Something went wrong. Please check that "
                    "Ollama is running and try again."
                )

st.divider()
st.caption("Local AI · RAG · ChromaDB · Ollama")
