# src/helper_agent/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, END

from .config import AppConfig
from .llm_client import LLMClient
from .models import AgentState
from .retrievers import OfflineDocRetriever, OnlineDocRetriever, DocRetriever


def build_retriever(config: AppConfig, llm_client: LLMClient) -> DocRetriever:
    return OfflineDocRetriever(config, llm_client) if config.mode == "offline" \
        else OnlineDocRetriever(config, llm_client)


def node_analyze_question(state: AgentState) -> AgentState:
    return state


def node_retrieve_docs(state: AgentState, retriever: DocRetriever) -> AgentState:
    question = state["question"]
    state["retrieved_docs"] = retriever.retrieve(question, k=5)
    return state


def node_draft_answer(state: AgentState, llm_client: LLMClient) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    ctx = "\n\n".join(
        f"[doc {i+1}] {d.source}\n{d.content}"
        for i, d in enumerate(docs)
    )

    system = (
        "You are a LangGraph/LangChain expert. "
        "Use documentation context when helpful. "
        "Provide clean explanations and working Python examples."
    )

    user = f"Question:\n{question}\n\nDocs:\n{ctx}"

    answer = llm_client.generate(user, system_instruction=system)

    state["final_answer"] = answer
    return state


def create_agent_app(config: AppConfig):
    llm = LLMClient(config)
    retriever = build_retriever(config, llm)

    builder = StateGraph(AgentState)

    builder.add_node("analyze", node_analyze_question)
    builder.add_node("retrieve", lambda s: node_retrieve_docs(s, retriever))
    builder.add_node("draft", lambda s: node_draft_answer(s, llm))

    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", END)

    return builder.compile()


def run_agent_once(app, config: AppConfig, question: str) -> str:
    initial: AgentState = {"question": question, "mode": config.mode}
    final = app.invoke(initial)
    return final.get("final_answer", "(no answer)")
