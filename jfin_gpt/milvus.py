import logging
import threading
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import FieldSchema, DataType, CollectionSchema

from jfin_gpt.constants import EMBEDDING_MODEL_NAME, DEVICE_TYPE, MILVUS_URL
from jfin_gpt.documents import documents_service
from jfin_gpt.exceptions import CollectionDoesNotExistException

from backend.jfin_gpt.constants import INITIAL_SOURCES_DIRECTORY


class MilvusService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, alias='default', url=MILVUS_URL, collection_name='jfin_llm',
                 reindex_documents=True):
        self._alias = alias
        self._collection_name = collection_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder='/app/embedding_models',
            model_kwargs={"device": DEVICE_TYPE},
        )
        self._vector_store = Milvus(
            embedding_function=self._embeddings,
            collection_name=self._collection_name,
            connection_args={"uri": url},
            consistency_level='Strong',
            auto_id=True,
            enable_dynamic_field=True
        )
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={
                "k": 3,
            }
        )
        self._url = url
        if reindex_documents:
            logging.info("Reindexing database...")
            self.reset_documents(True)

    def _clear_collection(self):
        logging.info(f"Clearing collection {self._collection_name}...")

        if not self._vector_store.client.has_collection(self._collection_name):
            logging.error(f"No collection found for {self._collection_name}")
            self.create_collection()
            return
        self._vector_store.client.delete(self._collection_name, filter="text like '%%'")

    def create_collection(self):
        logging.info(f"Creating collection {self._collection_name}...")
        if not self._vector_store.client.has_collection(self._collection_name):
            Milvus.from_documents(documents_service.split_to_documents(f'{INITIAL_SOURCES_DIRECTORY}/blank.pdf'),
                                  embedding=self._embeddings,
                                  collection_name=self._collection_name, auto_id=True,
                                  enable_dynamic_field=True)
            self._clear_collection()
            logging.info(f"Created collection {self._collection_name}")
        else:
            logging.warning(f"Collection {self._collection_name} already exists.")

    def has_documents(self):
        if not self._vector_store.client.has_collection(self._collection_name):
            logging.error(f"No collection found for {self._collection_name}")
            raise CollectionDoesNotExistException(self._collection_name)

        # Attempt to retrieve a small number of documents
        try:
            results = self._vector_store.client.query("jfin_llm", filter="text like '%%'", limit=1)
            return len(results) > 0
        except Exception as e:
            logging.error(f"Error checking documents in collection: {e}")
            return False

    def delete_document(self, filename: str):
        logging.info(f"Deleting document {filename} from {self._collection_name}...")
        if not self._vector_store.client.has_collection(self._collection_name):
            logging.error(f"No collection found for {self._collection_name}")
            raise CollectionDoesNotExistException(self._collection_name)
        if self.has_documents():
            self._vector_store.client.delete(self._collection_name, filter=f"source like '%{filename}'")

    def insert_file_or_directory(self, path: str):
        documents = documents_service.split_to_documents(path)
        self._insert_documents(documents)

    def _insert_documents(self, documents: List[Document]):
        logging.info(f"Inserting {len(documents)} documents into {self._collection_name}...")
        if not self._vector_store.client.has_collection(self._collection_name):
            logging.error(f"No collection found for {self._collection_name}")
            raise CollectionDoesNotExistException(self._collection_name)
        if len(documents) != 0:
            self._vector_store.add_documents(documents)

    def reset_documents(self, insert_documents=False):
        logging.info(f"Resetting documents in {self._collection_name}...")
        self._clear_collection()
        if insert_documents:
            documents = documents_service.split_to_documents()
            self._insert_documents(documents)

    def get_retriever(self):
        return self._retriever


milvus_service = MilvusService()
