import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from ...setting import RAGSettings

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ) -> None:
        self._setting = setting or RAGSettings()

    def create_vector_store(self):
        chroma_client = chromadb.PersistentClient(
            path=self._setting.storage.persist_dir_storage
        )
        chroma_collection = chroma_client.create_collection(
            self._setting.storage.collection_name
        )

        vectore_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vectore_store

    def get_index(self, nodes):
        if len(nodes) == 0:
            return None
        index = VectorStoreIndex(nodes=nodes)
        return index
