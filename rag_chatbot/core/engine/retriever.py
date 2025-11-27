from typing import List

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.tools import RetrieverTool
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever

from ..prompt import get_query_gen_prompt
from ...setting import RAGSettings

load_dotenv()


class TwoStageRetriever(QueryFusionRetriever):
    """Two-stage retriever with query generation and reranking."""

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        setting: RAGSettings | None = None,
        llm: str | None = None,
        query_gen_prompt: str | None = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = ...,
        num_queries: int = 4,
        use_async: bool = False,
        verbose: bool = False,
        callback_manager: CallbackManager | None = None,
        objects: List[IndexNode] | None = None,
        object_map: dict | None = None,
        retriever_weights: List[float] | None = None,
    ) -> None:
        super().__init__(
            retrievers,
            llm,
            query_gen_prompt,
            mode,
            similarity_top_k,
            num_queries,
            use_async,
            verbose,
            callback_manager,
            objects,
            object_map,
            retriever_weights,
        )
        self._setting = setting or RAGSettings()
        self._rerank_model = SentenceTransformerRerank(
            top_n=self._setting.retriever.top_k_rerank,
            model_name_or_path=self._setting.retriever.rerank_llm,
            device="cpu",  # Note that reranking in cpu bound!
            keep_retrieval_score=True,
        )
        # self._cutoff_obj = SimilarityPostprocessor(similarity_cutoff=0)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))
        # TODO: enable async calls to chromadb, currently only sync supported
        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)
        results = self._simple_fusion(results)
        results = self._rerank_model.postprocess_nodes(results, query_bundle)
        return results
        # return self._cutoff_obj.postprocess_nodes(results)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        results = self._simple_fusion(results)
        results = self._rerank_model.postprocess_nodes(results, query_bundle)
        return self._cutoff_obj.postprocess_nodes(results)


class LocalRetriever:
    """Handles document retrieval using various retrieval strategies."""

    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal",
        category: str = None,
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._host = host
        self._category = category

    # def _get_normal_retriever(
    #     self,
    #     vector_index: VectorStoreIndex,
    #     llm: LLM | None = None,
    # ):
    #     llm = llm or Settings.llm
    #     return VectorIndexRetriever(
    #         index=vector_index,
    #         similarity_top_k=self._setting.retriever.similarity_top_k,
    #         embed_model=Settings.embed_model,
    #         verbose=True,
    #     )

    def _get_hybrid_retriever(
        self,
        vector_index: VectorStoreIndex,
        nodes: List[BaseNode],
        llm: LLM | None = None,
        gen_query: bool = True,
    ):

        # VECTOR INDEX RETRIEVER
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="category", value=self._category)]
            ),
            similarity_top_k=self._setting.retriever.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True,
        )

        # bm25_retriever = BM25Retriever.from_defaults(
        #     nodes=nodes,
        #     similarity_top_k=self._setting.retriever.similarity_top_k,
        #     filters=MetadataFilters(
        #         filters=[MetadataFilter(key="category", value=self._category)]
        #     ),
        #     verbose=True,
        # )

        # FUSION RETRIEVER
        if gen_query:
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                use_async=False,
                query_gen_prompt=get_query_gen_prompt(),
                similarity_top_k=self._setting.retriever.top_k_rerank,
                num_queries=self._setting.retriever.num_queries,
                mode=self._setting.retriever.fusion_mode,
                verbose=True,
            )
        else:
            hybrid_retriever = TwoStageRetriever(
                # retrievers=[vector_retriever, bm25_retriever],
                retrievers=[vector_retriever],
                # retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=self._setting.retriever.similarity_top_k,
                num_queries=1,
                mode=self._setting.retriever.fusion_mode,
                verbose=True,
            )

        return hybrid_retriever

    def _get_router_retriever(
        self,
        vector_index: VectorStoreIndex,
        nodes: List[BaseNode],
        llm: LLM | None = None,
    ):
        # fusion_tool = RetrieverTool.from_defaults(
        #     retriever=self._get_hybrid_retriever(
        #         vector_index, nodes, llm, gen_query=True
        #     ),
        #     description=(
        #         "Используй этот инструмент, если запрос пользователя "
        #         "неоднозначен или неясен."
        #     ),
        #     name=(
        #         "Fusion Retriever with BM25 and Vector Retriever and "
        #         "LLM Query Generation."
        #     ),
        # )
        # two_stage_tool = RetrieverTool.from_defaults(
        #     retriever=self._get_hybrid_retriever(
        #         vector_index, nodes, llm, gen_query=False
        #     ),
        #     description=(
        #         "Используй этот инструмент, когда запрос пользователя "
        #         "ясен и недвусмыслен."
        #     ),
        #     name="Two Stage Retriever with BM25 and Vector Retriever and LLM Rerank.",
        # )

        retriever = self._get_hybrid_retriever(
            vector_index, nodes, llm, gen_query=False
        )

        return retriever

        # return RouterRetriever.from_defaults(
        #     selector=LLMSingleSelector.from_defaults(llm=llm),
        #     retriever_tools=[fusion_tool, two_stage_tool],
        #     llm=llm,
        # )

    def get_retrievers(
        self,
        nodes: List[BaseNode],
        vector_store: SimpleVectorStore,
        llm: LLM | None = None,
    ):
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )

        # if len(nodes) > self._setting.retriever.top_k_rerank:
        retriever = self._get_router_retriever(vector_index, nodes, llm)
        # else:
        # retriever = self._get_normal_retriever(vector_index, llm)

        return retriever
