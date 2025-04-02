from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field
from typing import List


def get_so_prompt():
    return prompt_template_str


def get_pydantic():
    return Block


class Block(BaseModel):

    header: List[str] = Field(
        description="This block is used to store header information.\
        For context: every table has a first 'header' row to define the name of different columns."
    )
    rows: List[List[str]] = Field(
        description=(
            "This block is used to store row-wise information.\
        For context: every table has a few rows to store the information under specific columns."
        )
    )

    def customize(self):
        output = []
        for row in self.rows:
            formatted_row = []
            for i, cell in enumerate(row):
                formatted_row.append(f"{self.header[i]}: {cell}")
            output.append("; ".join(formatted_row))
        return "\n\n".join(output)


prompt_template_str = """\
You are a table serialization agent.\n
Your task is to create a set of contextually independent blocks of information based on the provided table and surrounding text.\n
These blocks must be totally context-independent because they will be used as separate chunk to populate database.\
Context:\n{table}
"""
