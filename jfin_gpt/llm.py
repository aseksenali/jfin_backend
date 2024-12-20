import logging
import threading

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langdetect import DetectorFactory, detect

from jfin_gpt.constants import MODEL_NAME, OLLAMA_URL
from jfin_gpt.milvus import milvus_service


def get_prompt_language(question: str):
    try:
        # Convert to lowercase for better language detection accuracy
        question_lower = question.lower()
        language = detect(question_lower)

        if language == 'de':
            return 'deutsch'
        else:
            return 'english'
    except Exception as e:
        logging.warning(f"Error detecting language: {e}")
        return 'deutsch'


class LLMService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name=MODEL_NAME, ollama_url=OLLAMA_URL):
        self.model = ChatOllama(
            model=model_name,
            base_url=ollama_url,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.1,
        )

        DetectorFactory.seed = 0  # For consistent results

    def ask_question(self, prompt: str, history: list[tuple[str]]):
        logging.info(f"Initialized {self.model} model")
        rephrase_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="Given the chat history and the latest user question, reformulate the question to be standalone."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=milvus_service.get_retriever(),
            prompt=rephrase_prompt
        )

        qa_system_prompt = (
            "You are an assistant and your task is to answer questions based only on the provided context.\n"
            "Try to give accurate and informative document if there some information important in the context but the user didnt mention them, you can explain them breifly\n"
            "Give answer in {language}\n"
            "If you don't know the answer, just say that you don't know.\n"
            "Use Markdown formatting:\n"
            "##  for headers\n"
            "- for bullet points\n"
            "** for bold text\n"
            "* for italic text\n"
            "Use headers only when you need them not in all answers\n"
            "Example:\n"
            "##  Title\n"
            "Main text with **bold** and *italic*\n"
            "- Point 1\n"
            "- Point 2\n\n"
            "Context: {context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm=self.model, prompt=qa_prompt)

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain
        )

        language = get_prompt_language(prompt)
        logging.info(f"Selected {language} language")
        return rag_chain.invoke({"input": prompt, "language": language, "chat_history": history})


llm_service = LLMService()
