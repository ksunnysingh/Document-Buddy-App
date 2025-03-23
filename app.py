import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vectors import EmbeddingsManager  # Import the EmbeddingsManager class
from chatbot import ChatbotManager     # Import the ChatbotManager class

# Function to display the PDF of a given file
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session_state variables if not already present
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "chatbot_manager" not in st.session_state:
    st.session_state["chatbot_manager"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="Document Buddy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")

    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 Document Buddy App")
    st.markdown(
        """
        Welcome to **Document Buddy App**! 🚀

        **Built using Open Source Stack (Llama 3.2, BGE Embeddings, and Qdrant running locally within a Docker Container.)**

        - **Upload Documents**: Easily upload your PDF documents.
        - **Summarize**: Get concise summaries of your documents.
        - **Chat**: Interact with your documents through our intelligent chatbot.

        Enhance your document management experience with Document Buddy! 😊
        """
    )

# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface (Llama 3.2 RAG 🦙)")
    st.markdown("---")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: File Uploader and Preview
    with col1:
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} Files Uploaded Successfully!")
            st.session_state["uploaded_files"] = uploaded_files

            # Display uploaded files and save them temporarily
            temp_files = []
            for file in uploaded_files:
                temp_file_path = f"temp_{file.name}"
                temp_files.append(temp_file_path)
                with open(temp_file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.markdown(f"**Filename:** {file.name} ({file.size} bytes)")
                displayPDF(file)

            # Store temp file paths
            st.session_state["temp_files"] = temp_files

    # Column 2: Create Embeddings for All Uploaded Documents
    with col2:
        st.header("🧠 Create Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings for All Documents")

        if create_embeddings:
            if not st.session_state["uploaded_files"]:
                st.warning("⚠️ Please upload PDFs first.")
            else:
                try:
                    # Process each document individually
                    for file, temp_pdf_path in zip(st.session_state["uploaded_files"], st.session_state["temp_files"]):
                        collection_name = f"vector_db_{file.name}"  # Unique collection per document
                        embeddings_manager = EmbeddingsManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            qdrant_url="http://qdrant:6333",
                            collection_name=collection_name,
                        )

                        with st.spinner(f"🔄 Creating embeddings for {file.name}..."):
                            result = embeddings_manager.create_embeddings(temp_pdf_path)
                            time.sleep(1)

                        st.success(f"✅ Embeddings created for {file.name}")

                    # Initialize the ChatbotManager after embeddings are created
                    if st.session_state["chatbot_manager"] is None:
                        st.session_state["chatbot_manager"] = ChatbotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="llama3:8b",
                            llm_temperature=0.7,
                            qdrant_url="http://qdrant:6333",
                            collection_name="vector_db",
                        )

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Column 3: Chatbot Interface
    with col3:
        st.header("💬 Chat with Documents")

        if st.session_state["chatbot_manager"] is None:
            st.info("🤖 Please upload PDFs and create embeddings to start chatting.")
        else:
            for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).markdown(msg["content"])

            user_input = st.chat_input("Type your message here...")

            if user_input:
                st.chat_message("user").markdown(user_input)
                st.session_state["messages"].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        # Retrieve from all collections
                        all_collections = [
                            f"vector_db_{file.name}" for file in st.session_state["uploaded_files"]
                        ]

                        retrieved_docs = []
                        for collection in all_collections:
                            temp_db = Qdrant(
                                client=st.session_state["chatbot_manager"].client,
                                embeddings=st.session_state["chatbot_manager"].embeddings,
                                collection_name=collection,
                            )
                            retrieved_docs.extend(
                                temp_db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(user_input)
                            )

                        if retrieved_docs:
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            chatbot_prompt = f"Based on the extracted document information:\n\n{context}\n\n{user_input}"
                        else:
                            chatbot_prompt = user_input  # No context, just use LLM

                        answer = st.session_state["chatbot_manager"].get_response(chatbot_prompt)
                        time.sleep(1)

                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"

                st.chat_message("assistant").markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

# Contact Page
elif choice == "📧 Contact":
    st.title("📬 Contact Us")
    st.markdown(
        """
        We'd love to hear from you! Reach out for questions or feedback.

        - **Email:** [developer@example.com](mailto:aianytime07@gmail.com)
        - **GitHub:** [Contribute on GitHub](https://github.com/AIAnytime/Document-Buddy-App) 🛠️
        """
    )

# Footer
st.markdown("---")
st.markdown("© 2024 Document Buddy App by AI Anytime. All rights reserved. 🛡️")

