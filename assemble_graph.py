from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage

from query_pipeline import generate_query_or_respond
from indexing_pipeline import retrieve_context
from document_grader import grade_documents
from rewrite_question import rewrite_question
from generate_answer import generate_answer
from web_tool import search_web

def web_search_node(state: MessagesState):
    question = state["messages"][0].content
    results = search_web.invoke({"query": question})
    return {"messages": [ToolMessage(
        content=str(results),
        tool_call_id="web_search",
        name="search_web"
    )]}


workflow = StateGraph(MessagesState)

# 1. Définition des nœuds
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_context, search_web]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)
workflow.add_node("search_web", web_search_node)

# 2. Configuration des liens
workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
        "search_web": "search_web",
    },
)

workflow.add_edge("search_web", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# 3. Compilation
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    import sys
    from langchain_core.messages import HumanMessage

    question = " ".join(sys.argv[1:])
    if not question:
        print("Usage: python assemble_graph.py <question>")
        sys.exit(1)

    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    ):
        for node, update in chunk.items():
            print(f"\n--- Nœud : {node} ---")
            update["messages"][-1].pretty_print()
