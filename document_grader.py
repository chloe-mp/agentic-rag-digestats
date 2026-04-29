from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import MessagesState
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from models import gemini

GRADE_PROMPT = (
    "You are a grader assessing whether a retrieved document can DIRECTLY ANSWER a user question.\n\n"
    "Retrieved document:\n{context}\n\n"
    "User question: {question}\n\n"
    "Score 'yes' ONLY if the document contains specific information that directly answers the question "
    "(e.g. a specific value, threshold, procedure, or explicit rule).\n"
    "Score 'no' if the document is only thematically related but does not contain the specific answer.\n"
    "Give a binary score 'yes' or 'no'."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question", "search_web"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    # Compter combien de fois on a déjà fait retrieve
    tool_attempts = sum(1 for m in state["messages"] if m.type == "tool")

    config: RunnableConfig = get_config()
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = gemini.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}], config=config
    )

    if response.binary_score == "yes":
        return "generate_answer"
    elif tool_attempts >= 2:
        return "search_web"
    else:
        return "rewrite_question"


if __name__ == "__main__":
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "Quels sont les critères d'éligibilité DIG ?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_context",
                            "args": {"query": "critères éligibilité DIG"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    }
    print(grade_documents(input))

    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "Quels sont les critères d'éligibilité DIG ?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_context",
                            "args": {"query": "critères éligibilité DIG"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Les critères d'éligibilité au statut DIG comprennent : être une structure publique, justifier de l'intérêt général du projet et fournir une étude d'impact.",
                    "tool_call_id": "1",
                },
            ]
        )
    }
    print(grade_documents(input))
