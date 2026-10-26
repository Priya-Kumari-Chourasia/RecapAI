"""Single place that talks to the LLM (Groq through LangChain).

Everything else calls `complete()` or `complete_json()`, which keeps prompts
readable and lets the tests swap the model for a fake one.
"""
import json
import os
import re
from typing import Type, TypeVar

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from utils.llm_retry import invoke_with_retry

DEFAULT_MODEL = "openai/gpt-oss-120b"
T = TypeVar("T", bound=BaseModel)

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
        )
    return _llm


def complete(system: str, human: str) -> str:
    # Pass the text as variables rather than templates so braces in a
    # transcript or a JSON schema are never treated as prompt placeholders.
    prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{human}")])
    chain = prompt | get_llm() | StrOutputParser()
    return invoke_with_retry(chain, {"system": system, "human": human}).strip()


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence first."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if start != -1 and end > start else text


def complete_json(system: str, human: str, schema: Type[T]) -> T:
    """Ask for JSON matching a Pydantic model; retry once with the error if it doesn't validate."""
    schema_text = json.dumps(schema.model_json_schema())
    full_system = (
        f"{system}\n\nRespond with ONLY a JSON object that matches this JSON schema "
        f"(no markdown, no commentary):\n{schema_text}"
    )
    raw = complete(full_system, human)
    try:
        return schema.model_validate_json(_extract_json(raw))
    except (ValidationError, ValueError) as err:
        repair = (
            f"{human}\n\nYour previous reply was not valid JSON for the schema.\n"
            f"Error: {err}\nPrevious reply:\n{raw}\n\nReturn the corrected JSON only."
        )
        return schema.model_validate_json(_extract_json(complete(full_system, repair)))
