from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
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
        self._ingested_ids = set()

    def store_nodes(
        self,
        input_files: list[str],
        embed_model: Any | None = None,
    ) -> List[BaseNode]:
        if input_files in [None, []]:
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
            node_ids = self._vector_store.client.get(where={"file_name": file_name})[
                "ids"
            ]
            self._ingested_ids.update(node_ids)
            if node_ids:
                return self._vector_store.get_nodes(node_ids=node_ids)
            else:
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
                node_ids = self._vector_store.add(nodes)
                self._ingested_ids.update(node_ids)
        return self._vector_store.get_nodes(node_ids=node_ids)

    def reset(self):
        return self._vector_store.delete_nodes(node_ids=list(self._ingested_ids))

    def check_nodes_exist(self):
        return len(self._ingested_ids) > 0

    def get_all_nodes(self):
        return self._vector_store.get_nodes(
            node_ids=self._vector_store.client.get()["ids"]
        )

    def get_ingested_nodes(self):
        try:
            return self._vector_store.get_nodes(node_ids=list(self._ingested_ids))
        except:
            return []

    def get_vector_store(self):
        return self._vector_store
