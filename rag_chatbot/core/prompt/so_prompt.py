from typing import List

from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field


def get_so_prompt() -> PromptTemplate:
    return PromptTemplate(prompt_template_str)


def get_pydantic() -> type[Block]:
    return Block


class Block(BaseModel):

    classifier: str = Field(
        description=(
            "Output only the resulting name of the category from the prompt. "
            "Do not include the number!"
        )
    )

    # def customize(self):
    #     output = []
    #     for row in self.rows:
    #         formatted_row = []
    #         for i, cell in enumerate(row):
    #             formatted_row.append(f"{self.header[i]}: {cell}")
    #         output.append("; ".join(formatted_row))
    #     return "\n\n".join(output)


prompt_template_str = """\
    You are a classification LLM.
    Your task is to classify the provided user query into the following categories.
    The categories:
    ###
        1. Открытие счетов.
        2. Сервисное обслуживание текущих счетов.
        3. Выдача наличных со счетов.
        4. Прием наличных денежных средств на счета.
        5. Зачисление денежных средств на счета.
        6. Зачисление денежных средств от физического лица в белорусских рублях
        и иностранной валюте на счета и карточки вутри Банка.
        7. Зачисление денежных средств в белорусских рублях и иностранной валюте
        на счета и карточки физических лиц из банков Республики Беларусь
        и через платежные сервисы.
        8. Зачисление денежных средств в иностранной и ной валюте на счета
        и карточки, поступивших из-за границы.
        9. Зачисление денежных средств на счет физического лица от юридического лица.
        10. Переводы со счетов внутри банка.
        11. Переводы внутри Банка.
        12. Переводы в иные банки Республики Беларусь.
        13. Международные переводы.
        14. Плановый платеж.
        15. Предоставление выписок и справок.
        16. Оформление дополнительных завещательных соглашения к договору
        по инициативе Клиента с согласия Банка.
        17.	Банковские сейфы ( ячейки)  - хранение ценностей.
        18. Неподвижные счета.
        19. Курьерская доставка.
        20. Переводы по номеру мобильного телефона.
    ### \n
    The query:
    ###
    {query}
    ### \n
    Example:
    ###
    The query:
    ###
    Сколько стоит открыть счет в американских долларах?
    ### \n
    Response:
    ###
    {"classifier":"Открытие счетов"}
    ### \n
    ###
    """
