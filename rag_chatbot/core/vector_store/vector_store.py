import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel
from typing import Any
from dotenv import load_dotenv
from ...setting import RAGSettings

load_dotenv()


class LocalVectorStore(BaseModel):

    vector_store: Any = None
    _host: str = "host.docker.internal"
    _setting: RAGSettings | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setting = self._setting or RAGSettings()
        self.vector_store = self.setup(self._host, self._setting)

    def setup(self, host, setting):
        chroma_client = chromadb.HttpClient(host=host, port=setting.storage.port)
        chroma_collection = chroma_client.get_or_create_collection(
            setting.storage.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
