import os
import torch
import hashlib
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import CrossEncoder

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = None,
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "http://qdrant:6333",
        collection_name: str = "vector_db",
    ):
        if not device:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def hash_pdf(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def create_embeddings(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        doc_hash = self.hash_pdf(pdf_path)

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        # Check if this hash already exists in Qdrant
        # Search the vector_db collection and return 1 document that has a payload field called doc_hash with the value matching the current PDF’s hash. 
        # If we find one, that means we’ve already embedded this document — and we can skip reprocessing it."

        #
        # self.client.scroll(...)	Calls Qdrant to scroll through stored vectors
        # collection_name=...	Tells it which collection to look in (e.g., vector_db)
        # scroll_filter=...	Applies a filter to find only points where doc_hash == current hash
        # "must"	Acts like an AND condition (standard Qdrant filtering syntax)
        # {"key": "doc_hash", "match": {"value": doc_hash}}	Only return vectors where the doc_hash payload matches our current PDF
        # limit=1	We just want to check if any match exists — so we stop at 1 result
        # existing_points, _ = ...	Extract the matching points; we discard the next_page_offset using _

        # existing_points will contain that vector chunk, and we will skip re-indexing.
        # existing_points, _ ---> a Python pattern known as tuple unpacking
        # In Python, _ is a convention that means: "I’m intentionally ignoring this value."
        # It’s used when a function returns multiple values, but you only care about some of them.

        # The .scroll() method in Qdrant returns two values: 
        # (points, next_page_offset) = self.client.scroll(...)
        # points: a list of matching documents (vectors + payload)
        # next_page_offset: used for pagination (i.e., getting the next batch if there are more results)

        if self.collection_name in collection_names:
            try:
                existing_points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={"must": [{"key": "doc_hash", "match": {"value": doc_hash}}]},
                    limit=1
                )
                if existing_points:
                    return "✅ Embeddings already exist for this document. Skipping reprocessing."
                else:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=Filter(
                            must_not=[
                                FieldCondition(
                                    key="doc_hash",
                                    match={"value": doc_hash}
                                )
                            ]
                        )
                    )
            except UnexpectedResponse as e:
                if "doesn't exist" in str(e):
                    print("Collection not found during scroll.")
                else:
                    raise

        # What is next_page_offset?
        # It’s a marker (an internal offset) that tells Qdrant where to continue the scroll from.
        # 
        # When you scroll with limit=N, Qdrant returns: 
        #  1) N points 
        #  2) next_page_offset (if there are more points)
        # 
        # If next_page_offset is None, you're done — you've reached the end.

        # all_points = []
        # next_offset = None

        # while True:
        #    points, next_offset = client.scroll(
        #        collection_name="vector_db",
        #        limit=100,
        #        offset=next_offset,  # 👈 Start from here
        #        with_payload=True,
        #        with_vectors=True
        #    )
        #    all_points.extend(points)
        #
        #    if next_offset is None:
        #        break  # No more data left

        # Load and preprocess the document
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")

        # Add the hash to the payload of each document
        for doc in splits:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["doc_hash"] = doc_hash

        # Create and store embeddings in Qdrant
        try:
            qdrant = Qdrant.from_documents(
                splits,
                embedding=self.embeddings,
                url=self.qdrant_url,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )

            print(qdrant.client.get_collection(collection_name=self.collection_name))

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "✅ Vector DB successfully created and stored in Qdrant!"

    def rerank_by_keyword(self, documents, keyword):
        keyword_lower = keyword.lower()
        ranked = sorted(documents, key=lambda d: keyword_lower in d.page_content.lower(), reverse=True)
        return ranked

    def rerank_with_cross_encoder(self, query, documents, top_k=5):
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: -x[0])]
        return ranked_docs[:top_k]
    
    # rerank_combined() gives you a list of top-k documents that:
    #   1) Contain relevant keywords, and
    #   2) Are highly aligned with the query’s intent

    # It does 1) Keyword-Aware Filtering and then 2) CrossEncoder Reranking

    # 1) Keyword-Aware Filtering
    # --------------------------
    # self.rerank_by_keyword(documents, keyword=query): This does a basic string match on each document’s page_content, like: query.lower() in doc.page_content.lower()
    # Then it reorders the documents to put the ones containing the query terms near the top

    # 2) CrossEncoder Reranking
    # -------------------------
    # Then it uses a deep relevance model (cross-encoder/ms-marco-MiniLM-L-6-v2) to rerank those documents again based on true semantic match with the query:
    #   pairs = [(query, doc.page_content) for doc in keyword_filtered_docs]
    #   scores = model.predict(pairs)
    #
    # The CrossEncoder:
    #   1) Takes the full query + document pair
    #   2) Applies deep attention across both
    #   3) Outputs a relevance score
    #   4) You then sort the documents by this score and keep the best top_k

    def rerank_combined(self, query, documents, top_k=5):
        # First apply keyword-based soft filter
        keyword = query.lower().strip("? ")
        keyword_filtered = self.rerank_by_keyword(documents, keyword)

        # Then rerank with cross encoder for precision
        return self.rerank_with_cross_encoder(query, keyword_filtered, top_k=top_k)

