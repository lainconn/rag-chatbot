from ..rag_chatbot.ui import LocalChatbotUI
from ..rag_chatbot.pipeline import LocalRAGPipeline
from ..rag_chatbot.logger import Logger


# Test document uploading pipeline
def test_document_upload():
    ui = LocalChatbotUI(pipeline=LocalRAGPipeline, logger=Logger)
    ui._processing_document(document=[])
    assert isinstance(ui._pipeline._query_engine, None)
