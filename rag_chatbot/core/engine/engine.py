from typing import List

from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
    SimpleChatEngine,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import SimpleVectorStore

from .retriever import LocalRetriever
from ...setting import RAGSettings


class LocalChatEngine:
    """Chat engine for handling conversational interactions."""

    def __init__(
        self,
        setting: RAGSettings | None = None,
        category: str = None,
        host: str = "host.docker.internal",
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(setting=self._setting, category=category)
        self._host = host

    def update_category(self, category: str) -> None:
        """Update the category for document filtering."""
        self._retriever.update_category(category)

    def set_engine(
        self,
        llm: LLM,
        nodes: List[BaseNode],
        vector_store: SimpleVectorStore | None = None,
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # Normal chat engine
        if len(nodes) == 0:
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                ),
            )

        # Chat engine with documents
        retriever = self._retriever.get_retrievers(
            llm=llm,
            nodes=nodes,
            vector_store=vector_store,
        )
        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            skip_condense=True,
            memory=ChatMemoryBuffer(token_limit=self._setting.ollama.chat_token_limit),
        )
