from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.formatter import format_docs


class Answer(BaseModel):
    message: str = Field(description="Output message from LLM")
    source: str | None = Field(description="Source link used")


def build_chain(retriever, llm):

    parser = PydanticOutputParser(pydantic_object=Answer)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
{format_instruction}
""",
        partial_variables={"format_instruction": parser.get_format_instructions()},
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )

    return chain