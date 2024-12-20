import os
from pathlib import Path

import torch
from langchain_community.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, \
    Docx2txtLoader

TRUTHY_VALUES = ('true', '1', 't')  # Add more entries if you want, like: `y`, `yes`, `on`, ...
FALSY_VALUES = ('false', '0', 'f')  # Add more entries if you want, like: `n`, `no`, `off`, ...
VALID_VALUES = TRUTHY_VALUES + FALSY_VALUES


def get_bool_env_variable(name: str, default_value: bool | None = None) -> bool:
    value = os.getenv(name) or default_value
    if value is None:
        raise ValueError(f'Environment variable "{name}" is not set!')
    value = str(value).lower()
    if value not in VALID_VALUES:
        raise ValueError(f'Invalid value "{value}" for environment variable "{name}"!')
    return value in TRUTHY_VALUES


ROOT_DIRECTORY = Path(__file__).parent.resolve()

SOURCE_DIRECTORY = os.getenv('SOURCE_DIRECTORY', "./sources")

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'intfloat/multilingual-e5-base')
MODEL_NAME = os.getenv('MODEL_NAME', 'llama3.2')

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5110")
REINDEX_REQUIRED = get_bool_env_variable("REINDEX", True)
