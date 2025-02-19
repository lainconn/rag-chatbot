import os
import torch
import requests
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoModel, AutoTokenizer
from ...setting import RAGSettings
from dotenv import load_dotenv


load_dotenv()


class LocalEmbedding:
    @staticmethod
    def set(setting: RAGSettings | None = None, host: str = "host.docker.internal"):
        setting = setting or RAGSettings()
        model_name = setting.ingestion.embed_llm
        return OllamaEmbedding(
            model_name=model_name,
            base_url=f"http://{host}:{setting.ollama.port}",
            embed_batch_size=setting.ingestion.embed_batch_size,
        )

    @staticmethod
    def pull(host: str):
        setting = RAGSettings()
        payload = {"name": setting.ingestion.embed_llm}
        return requests.post(f"http://{host}:11434/api/pull", json=payload, stream=True)

    @staticmethod
    def check_model_exist(host: str) -> bool:
        setting = RAGSettings()
        data = requests.get(f"http://{host}:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if setting.ingestion.embed_llm in list_model:
            return True
        return False
