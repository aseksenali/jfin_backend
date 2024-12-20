import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, CSVLoader, OutlookMessageLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from jfin_gpt.constants import SOURCE_DIRECTORY


class DocumentsService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, directory=SOURCE_DIRECTORY):
        self.sources_directory = directory
        self.filter_keywords = ['Inhaltsverzeichnis', 'Table of Contents', 'Index']

    def split_to_documents(self, path=None) -> List[Document]:
        if path is None:
            path = self.sources_directory
        if os.path.isdir(path):
            logging.info(f"Loading directory {path}")
            documents = PyPDFDirectoryLoader(path).load()
        else:
            match Path(path).suffix.lower():
                case '.pdf':
                    logging.info(f"Loading PDF file {path}")
                    documents = PyPDFLoader(path).load()
                case '.csv':
                    logging.info(f"Loading CSV file {path}")
                    documents = CSVLoader(path).load()
                case '.msg':
                    logging.info(f"Loading MSG file {path}")
                    documents = OutlookMessageLoader(path).load()
                case _:
                    raise ValueError("Unsupported file format")

        filtered_documents = [
            doc for doc in documents if not self._contains_filter_keywords(doc.page_content)
        ]
        logging.info(f"Loaded {len(filtered_documents)} documents from {SOURCE_DIRECTORY}")
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1024,
                                                       chunk_overlap=100)
        texts = text_splitter.split_documents(filtered_documents)
        logging.info(f"Split into {len(texts)} chunks of text")
        return texts

    def get_files(self, directory=None):
        if directory is None:
            directory = self.sources_directory
        documents = []
        if not os.path.exists(directory):
            return []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                creation_time = os.path.getctime(path)
                creation_date = datetime.fromtimestamp(creation_time).strftime('%d.%m.%Y %H:%M')
                documents.append({
                    'name': Path(path).name,
                    'creation_date': creation_date
                })
        return documents

    def has_file(self, filename):
        return os.path.exists(os.path.join(self.sources_directory, filename))

    def save_file(self, file) -> tuple[str, str, str]:
        filename: str = file.filename
        logging.info(f"Saving file {filename}")
        file_path = os.path.join(self.sources_directory, filename)
        if os.path.exists(filename):
            raise FileExistsError()
        file.save(file_path)
        creation_time = os.path.getctime(file_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime('%d.%m.%Y %H:%M')
        return file_path, filename, creation_date

    def reset_sources(self):
        if os.path.exists(self.sources_directory):
            for filename in os.listdir(self.sources_directory):
                file_path = os.path.join(self.sources_directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.error(f"Failed to remove {file_path}. Reason: {e}")

    def delete_file(self, filename: str):
        if os.path.exists(os.path.join(self.sources_directory, filename)):
            os.remove(os.path.join(self.sources_directory, filename))
            logging.info(f"Deleted file {filename} from sources")
            return True
        else:
            logging.warning(f"Tried to delete non-existing file {filename} from sources")
            return False

    def _contains_filter_keywords(self, text: str):
        for keyword in self.filter_keywords:
            if keyword.lower() in text.lower():
                return True
        return False


documents_service = DocumentsService()
