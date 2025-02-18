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
        chroma: LocalVectorStore | None = None,
    ) -> None:
        self._setting = setting or RAGSettings()
        self._node_store = chroma.create_vector_store()
        self._ingested_file = []

    def store_nodes(
        self,
        input_files: list[str],
        embed_nodes: bool = True,
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
        if embed_nodes:
            Settings.embed_model = embed_model or Settings.embed_model
        for input_file in tqdm(input_files):
            file_name = input_file.strip().split("/")[-1]
            # if file_name in self._node_store:
            #     return_nodes.extend(self._node_store[file_name])
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
            if embed_nodes:
                nodes = Settings.embed_model(nodes, show_progress=True)
            self._node_store.add(nodes)
        return self._node_store.get_nodes()

    def reset(self):
        self._node_store = {}
        self._ingested_file = []

    def check_nodes_exist(self):
        return len(self._node_store.values()) > 0

    def get_all_nodes(self):
        return_nodes = []
        for nodes in self._node_store.values():
            return_nodes.extend(nodes)
        return return_nodes

    def get_ingested_nodes(self):
        return_nodes = []
        for file in self._ingested_file:
            return_nodes.extend(self._node_store[file])
        return return_nodes
