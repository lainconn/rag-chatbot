from llama_index.core import Document, Settings, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from dotenv import load_dotenv
from typing import Any, List
from tqdm import tqdm
from ...setting import RAGSettings
from ..vector_store import LocalVectorStore
from unstructured.partition.auto import partition

load_dotenv()


class LocalDataIngestion:
    def __init__(
        self,
        setting: RAGSettings | None = None,
    ) -> None:
        self._setting = setting or RAGSettings()
        self._vector_store = LocalVectorStore().setup()
        self._document_store = SimpleDocumentStore()
        self._storage_context = StorageContext
        self._doc_id = []

    def store_nodes(
        self,
        input_files: list[str],
        embed_model: Any | None = None,
    ) -> List[BaseNode]:
        if len(input_files) == 0:
            return []
        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex,
        )
        Settings.embed_model = embed_model or Settings.embed_model
        for input_file in tqdm(input_files):
            file_name = input_file.strip().split("/")[-1]
            # node_ids = self._vector_store.client.get(where={"file_name": file_name})[
            #     "ids"
            # ]
            # self._ingested_file.update(node_ids)
            # if node_ids:
            #     return self._node_store.get_nodes(node_ids=node_ids)
            # else:
            elements = partition(
                filename=input_file,
                languages=["rus", "eng"],
                strategy="fast",
                skip_infer_table_types=["jpg", "png", "heic"],
            )

            text = " "
            for element in elements:
                text += "\n\n" + element.text

            document = Document(
                text=text,
                metadata={
                    "file_name": file_name,
                },
            )

            nodes = splitter([document], show_progress=True)
            nodes = Settings.embed_model(nodes, show_progress=True)
            self._document_store.add_documents(nodes)
            # node_ids = self._node_store.add(nodes)
            # self._ingested_file.update(node_ids)
        return []

    def reset(self):
        return []
        # return self._node_store.delete_nodes(node_ids=list(self._ingested_file))

    def check_nodes_exist(self):
        return len(self._document_store.docs.values()) > 0

    def get_all_nodes(self):
        return self._document_store.docs.values()

    def get_ingested_nodes(self):
        return list(self._document_store.docs.values())

    def get_storage_context(self):
        return self._storage_context.from_defaults(
            vector_store=self._vector_store,
            docstore=self._document_store,
        )
