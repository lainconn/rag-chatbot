def get_context_prompt() -> str:
    return CONTEXT_PROMPT_RUS


def get_system_prompt(is_rag_prompt: bool = True) -> str:
    return SYSTEM_PROMPT_RAG_RUS if is_rag_prompt else SYSTEM_PROMPT_RUS


SYSTEM_PROMPT_RUS = """\n
Это чат между пользователем и помощником с искусственным интеллектом.\n 
Помощник дает полезные, подробные и вежливые ответы на вопросы пользователя, исходя из контекста.\n
Помощник также должен указать, когда ответ не может быть найден в контексте."""

SYSTEM_PROMPT_RAG_RUS = """\
Это чат между пользователем и помощником с искусственным интеллектом.\n
Помощник дает полезные, подробные и вежливые ответы на вопросы пользователя, исходя из контекста.\n
Помощник также должен указать, когда ответ не может быть найден в контексте."""

CONTEXT_PROMPT_RUS = """\
Вот соответствующие документы для контекста::

{context_str}

Инструкция: На основании вышеуказанных документов предоставь подробный ответ на вопрос пользователя ниже.\n
Ответь «не знаю», если его нет в документе."""

CONDENSED_CONTEXT_PROMPT_EN = """\
Given the following conversation between a user and an AI assistant and a follow up question from user,
rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:\
"""
