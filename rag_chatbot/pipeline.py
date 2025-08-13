from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt,
    get_so_prompt,
    get_pydantic,
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
import re


class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._model_name = ""
        self._category = None
        self._system_prompt = get_system_prompt(is_rag_prompt=False)
        self._engine = LocalChatEngine(host=host, category=self._category)
        self._default_model = LocalRAGModel.set(self._model_name, host=host)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore().vector_store
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def get_model_name(self):
        return self._model_name

    def set_model_name(self, model_name: str):
        self._model_name = model_name

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or get_system_prompt(
            is_rag_prompt=self._ingestion.check_nodes_exist()
        )

    def set_model(self):
        Settings.llm = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._host,
        )
        self._default_model = Settings.llm

    def reset_engine(self):
        self._query_engine = self._engine.set_engine(llm=self._default_model, nodes=[])

    def reset_documents(self):
        self._ingestion.reset()

    def clear_conversation(self):
        self._query_engine.reset()

    def reset_conversation(self):
        self.reset_engine()
        self.set_system_prompt(get_system_prompt(is_rag_prompt=False))

    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(self._host, model_name)

    def pull_embed_model(self):
        return LocalEmbedding.pull(self._host)

    def check_exist(self, model_name: str) -> bool:
        return LocalRAGModel.check_model_exist(self._host, model_name)

    def check_exist_embed(self) -> bool:
        return LocalEmbedding.check_model_exist(self._host)

    def store_nodes(self, input_files: list[str] = None) -> None:
        self._ingestion.store_nodes(input_files=input_files)

    def set_chat_mode(self, system_prompt: str | None = None):
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()

    def set_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            vector_store=self._vector_store,
        )

    # Fix later
    def get_category(self, message: str):
        llm = Settings.llm
        prompt = get_so_prompt()
        schema = get_pydantic()

        response = llm.structured_predict(
            schema,
            prompt,
            query=message,
        )

        response = re.sub(r"[\d\.]+", " ", response.classifier).strip()
        return response

    # Fix later
    def set_category(self):
        self._engine = LocalChatEngine(host=self._host, category=self._category)

    def get_history(self, chatbot: list[list[str]]):
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append(ChatMessage(role=MessageRole.USER, content=chat[0]))
                history.append(ChatMessage(role=MessageRole.ASSISTANT, content=chat[1]))
        return history

    def query(
        self, mode: str, message: str, chatbot: list[list[str]]
    ) -> StreamingAgentChatResponse:
        if mode == "chat":
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
        else:
            self._category = self.get_category(message)
            # Fix later: double engine init. During _document_processing and here!
            self.set_category()
            self.set_engine()
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
