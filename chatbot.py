# chatbot.py

import os
import torch

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
#/usr/local/lib/python3.9/site-packages/langchain/__init__.py:30: 
#UserWarning: Importing PromptTemplate from langchain root module is no longer supported. 
#Please use langchain_core.prompts.PromptTemplate instead
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = None, # allow auto-detection if not passed 

        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3:8b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """

        if not device:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Local LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
            # Add other parameters if needed
        )

        # Define the prompt template
#        self.prompt_template = """Use the following pieces of information to answer the user's question.
#If you don't know the answer, just say that you don't know, don't try to make up an answer.
#
#Context: {context}
#Question: {question}
#
#Only return the helpful answer. Answer must be detailed and well explained.
#Helpful answer:
#"""
#        self.prompt_template = """Use the following extracted document information to answer the user's question.
#If the document does not provide an answer, say 'The document does not contain relevant information.'
#
#Document Context: {context}
#User Question: {question}
#
#Provide a response strictly based on the document. Avoid speculation.
#Response:
#"""

        self.prompt_template = """Use the following extracted document information to answer the user's question.
If the document does not provide an answer, say 'The document does not contain relevant information.'

Document Context: {context}
User Question: {question}

Provide a well-structured, informative, and detailed response. If relevant, include specific details from the document.
Always attempt to provide insights and explanations rather than short answers.

Answer:
"""

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=True
        )

    def get_responseOLD(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            response = self.qa.run(query)
            return response  # 'response' is now a string containing only the 'result'
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
        

    def get_response(self, query: str) -> str:
        try:
            # Step 1: Get top-k documents from Qdrant via retriever
            docs = self.retriever.get_relevant_documents(query)

            # Step 2: Build context string
            #context = "\n\n".join([doc.page_content for doc in docs])

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

            return response

        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."

    
