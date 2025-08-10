from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.program import LLMTextCompletionProgram
from dotenv import load_dotenv
from typing import Any, List
from tqdm import tqdm
from ...setting import RAGSettings
from ..vector_store import LocalVectorStore
from ..prompt import get_so_prompt, get_pydantic
import json
from pathlib import Path
from unstructured.partition.auto import partition

load_dotenv()


class LocalDataIngestion:
    def __init__(
        self,
        setting: RAGSettings | None = None,
    ) -> None:
        self._setting = setting or RAGSettings()
        self._vector_store = LocalVectorStore().vector_store
        self._ingested_ids = set()
        self._customizer = get_pydantic().customize

    def store_nodes(
        self,
        input_files: list[str],
        embed_model: Any | None = None,
    ) -> None:
        if input_files in [None, []]:
            return []
        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex,
        )
        # program = LLMTextCompletionProgram.from_defaults(
        #     output_cls=get_pydantic(),
        #     llm=Settings.llm,
        #     prompt_template_str=get_so_prompt(),
        #     verbose=True,
        # )
        Settings.embed_model = embed_model or Settings.embed_model
        for input_file in tqdm(input_files):
            file_name = input_file.strip().split("/")[-1]
            node_ids = self._vector_store.client.get(where={"file_name": file_name})[
                "ids"
            ]
            if node_ids:
                self._ingested_ids.update(node_ids)
            else:
                if file_name.split(".")[-1] == "json":
                    with open(input_file, "r", encoding="utf-8") as file:
                        elements = json.load(file)
                    try:
                        for key_for_lists, inner_dict in elements.items():
                            for i in range(len(inner_dict.values())):
                                text = f"{key_for_lists}: "
                                for key, value_list in inner_dict.items():
                                    text += f"{key}: {value_list[i]}; "

                                print(text.strip())
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
                    except:
                        max_length = max(
                            len(value_list) for value_list in elements.values()
                        )

                        for i in range(max_length):
                            text = ""
                            for key, value_list in elements.items():
                                if i < len(value_list):
                                    text += f"{key}: {value_list[i]}; "

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
                else:
                    elements = partition(
                        filename=input_file,
                        languages=["rus", "eng"],
                        strategy="hi_res",
                        skip_infer_table_types=["jpg", "png", "heic"],
                    )

                    text = " "
                    for element in elements:
                        text += element.text + "\n\n"

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

        return None

    def reset(self):
        return self._vector_store.clear()

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
