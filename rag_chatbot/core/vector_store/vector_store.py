import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from ...setting import RAGSettings

load_dotenv()


class LocalVectorStore:
    def __init__(self) -> None:
        pass

    def setup(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ):
        self._setting = setting or RAGSettings()
        chroma_client = chromadb.HttpClient(host=host, port=self._setting.storage.port)
        chroma_collection = chroma_client.get_or_create_collection(
            self._setting.storage.collection_name
        )
        vectore_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vectore_store

    def get_index(self, nodes):
        if len(nodes) == 0:
            return None
        index = VectorStoreIndex(nodes=nodes)
        return index
