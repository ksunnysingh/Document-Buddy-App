from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import re

# Extracts keywords dynamically from the query
# Uses your EmbeddingsManager to rerank chunks
# Displays matched keywords and retrieved chunks in the Streamlit UI

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3:8b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
        embeddings_manager=None
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embeddings_manager = embeddings_manager

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
        )

        self.prompt_template = """Use the following extracted document information to answer the user's question.
            If the document does not provide an answer, say 'The document does not contain relevant information.'

            Document Context: {context}
            User Question: {question}

            Provide a well-structured, informative, and detailed response. If relevant, include specific details from the document.
            Always attempt to provide insights and explanations rather than short answers.

            Answer:
            """

        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
        self.chain_type_kwargs = {"prompt": self.prompt}

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        try:
            # Step 1: Retrieve from Qdrant
            raw_docs = self.retriever.get_relevant_documents(query, k=20)

            # Step 2: Rerank based on ererank_combined() + show match info in UI

            docs = self.embeddings_manager.rerank_combined(query, raw_docs, top_k=5)

            #############################################################################################################################################
            for i, doc in enumerate(docs):
                print(f"\n🔍 Chunk {i+1}:\n{doc.page_content[:500]}...\n")

            # Deduplicate by content
            seen = set()
            unique_chunks = []
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_chunks.append(doc)

            #############################################################################################################################################

            # rerank_combined() does not show match info in the UI by itself.
            # It’s purely a backend ranking function. It:
            #   1. Applies keyword-based filtering (via rerank_by_keyword)
            #   2. Applies CrossEncoder scoring
            #   3. Returns the top-k most relevant chunks

            # If we want to Show Match Info in Streamlit (UI)
            # we need to manually display it after the call to rerank_combined()

            # Show retrieved chunks in the UI
            # The retrieved chunks now highlight matched keywords in yellow using HTML
            #
            # support multiple keywords or switch to bold + underline styles
            #   We highlight just one keyword (your query) using:
            #   <span style='background-color: #ffff66'>keyword</span>
            #
            # This works great for simple queries like:
            # “Does Sunny know Kubernetes?”
            # But what if your query contains multiple important terms? for example: “Does Sunny have experience with Kubernetes, Docker, or Helm?”
            # You might want to highlight all 3 terms: Kubernetes, Docker, Helm
            # We’d break query into important terms and highlight each one in the chunk text.
            #
            # Instead of just yellow background, you could also:
            # Style	Example
            #   Bold + underline	<u><b>keyword</b></u>
            #   🔵 Color text	<span style='color:blue'>keyword</span>
            #   🟨 Background highlight	<span style='background-color:#ffff66'>keyword</span>
            #
            # Below code highlights  multiple keywords from the user query Custom styling: bold, underlined, and yellow background
            #
            # The code below does all of the above plus ignores stopwords
            #     1) Extracts keywords while ignoring common stopwords (like “does”, “have”, “is”)
            #     2) Preserves quoted phrases as exact keywords (e.g., "cloud architecture" or 'risk engine')
            #     3) Highlights them in the UI with custom styles

            # Keyword analytics
            import collections
            keywords = [word.lower() for word in re.findall(r'\b\w+\b', query.lower()) if len(word) > 3]
            keyword_to_chunks = collections.defaultdict(list)
            for doc in docs:
                for kw in keywords:
                    if kw in doc.page_content.lower():
                        keyword_to_chunks[kw].append(doc)

            # Sort keywords by frequency
            sorted_keywords = sorted(keyword_to_chunks.items(), key=lambda x: -len(x[1]))

            with st.expander("📄 Retrieved Context Chunks Grouped by Keyword"):
                if sorted_keywords:
                    st.markdown("### 🔑 Matched Keywords:")
                    st.markdown(" ".join([f"<span style='background-color:#e0f7fa;padding:6px 10px;border-radius:10px;margin:4px;display:inline-block'>{kw}</span>" for kw, _ in sorted_keywords]), unsafe_allow_html=True)

                for kw, chunk_list in sorted_keywords:
                    st.markdown(f"### 🔍 Keyword: {kw} ({len(chunk_list)} match(es))")
                    for i, doc in enumerate(chunk_list):
                        st.markdown(f"**Chunk {i+1}:**")
                        highlighted = doc.page_content[:1000]
                        highlighted = re.sub(f"(?i)({re.escape(kw)})", r"<u><b><span style='background-color: #ffff66'>\\1</span></b></u>", highlighted)
                        st.markdown(highlighted, unsafe_allow_html=True)


            # Step 3: Create prompt

            # Now join only unique chunks into the prompt context
            #context = "\n\n".join(unique_chunks)
            context = "\n\n".join([doc.page_content for doc in unique_chunks])

            # Step 3: Format the final prompt using the template
            filled_prompt = self.prompt.format(context=context, question=query)

            # 🔍 Print it to terminal / log it
            print("\n" + "="*30)
            print("🧠 Prompt Sent to LLM:")
            print(filled_prompt)
            print("="*30 + "\n")


            # Step 4: Get response from LLM
            response = self.llm.invoke(filled_prompt)

            return response.content if hasattr(response, 'content') else response

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            return "⚠️ Sorry, an error occurred while processing your request."