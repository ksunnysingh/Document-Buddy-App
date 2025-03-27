from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

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
            # Step 1: Retrieve from Qdrant. 
            # Get top-k documents from Qdrant via retriever ; in the constructor self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
            docs = self.retriever.get_relevant_documents(query)

            # Step 2: Rerank based on extracted keywords + show match info in UI
            docs = self.embeddings_manager.rerank_by_keyword(docs, keyword=query)

            # Step 3: Create prompt
            # context = "\n\n".join([doc.page_content for doc in docs])
            # prompt = self.prompt.format(context=context, question=query)

            # Step 4: Get response from LLM
            # response = self.llm.invoke(prompt)

            for i, doc in enumerate(docs):
                print(f"\n🔍 Chunk {i+1}:\n{doc.page_content[:500]}...\n")

            # Deduplicate by content
            unique_chunks = list({doc.page_content for doc in docs})

            # Now join only unique chunks into the prompt context
            context = "\n\n".join(unique_chunks)

            # Step 3: Format the final prompt using the template
            filled_prompt = self.prompt.format(context=context, question=query)

            # 🔍 Print it to terminal / log it
            print("\n" + "="*30)
            print("🧠 Prompt Sent to LLM:")
            print(filled_prompt)
            print("="*30 + "\n")

            # Step 4: Call the LLM directly (bypass RetrievalQA)
            response = self.llm.invoke(filled_prompt)

            # LLM responses (like from LangChain or OpenAI) may come back as either:

            # 1) An object (e.g., AIMessage) → with a .content attribute containing the text
            # 2) A plain string → already the text (no .content needed)
            #
            # If the response object has a .content attribute, return that.
            # Otherwise, just return the response itself
        
            return response.content if hasattr(response, 'content') else response

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            return "⚠️ Sorry, an error occurred while processing your request."
