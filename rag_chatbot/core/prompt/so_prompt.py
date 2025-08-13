from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field
from typing import List


def get_so_prompt():
    return PromptTemplate(prompt_template_str)


def get_pydantic():
    return Block


class Block(BaseModel):

    classifier: str = Field(
        description="Output only the resulting name of the category from the prompt. Do not include the number!"
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
        6. Переводы со счетов.
        7. Переводы в пределах Банка.
        8. Переводы в иные банки Республики Беларусь.
        9. Международные переводы.
        10. Плановый платеж.
        11. Предоставление выписок и справок.
        12. Оформление дополнительных  завещательных  соглашения к договору по инициативе Клиента с согласия Банка.
        13. Банковские сейфы ( ячейки)  - хранение ценностей.
        14. Неподвижные счета.
        15. Курьерская доставка.
        16. Переводы по номеру мобильного телефона.	
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
