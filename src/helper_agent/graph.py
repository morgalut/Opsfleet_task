# src/helper_agent/graph.py
from __future__ import annotations

import re
from typing import List

from langgraph.graph import StateGraph, END

from .config import AppConfig
from .llm_client import LLMClient
from .models import AgentState, Document
from .retrievers import OfflineDocRetriever, OnlineDocRetriever, DocRetriever
from .retry_utils import with_retry
from .retry_node import RetryNode


# ------------------------------------------------------------
# Helper: build retriever from mode
# ------------------------------------------------------------
def build_retriever(config: AppConfig, llm_client: LLMClient) -> DocRetriever:
    return (
        OfflineDocRetriever(config, llm_client)
        if config.mode == "offline"
        else OnlineDocRetriever(config, llm_client)
    )


# ------------------------------------------------------------
# Common / basic pipeline nodes
# ------------------------------------------------------------
def node_analyze_question(state: AgentState) -> AgentState:
    """
    Currently a light node – could be extended with classification.
    Ensures we always have a reasoning_style in state.
    """
    if "reasoning_style" not in state:
        state["reasoning_style"] = "basic"
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
        "Provide clear explanations and working Python examples."
    )

    prompt = f"Question:\n{question}\n\nDocs:\n{ctx}"
    state["final_answer"] = llm_client.generate(prompt, system_instruction=system)
    return state


# ------------------------------------------------------------
# ORC-style nodes: plan → retrieve → answer
# ------------------------------------------------------------
def node_orc_plan(state: AgentState, llm_client: LLMClient) -> AgentState:
    question = state["question"]

    system = (
        "You are a senior LangGraph/LangChain engineer. "
        "Break the question into 2–6 sub-questions. "
        "Return only a numbered list."
    )

    plan = llm_client.generate(
        prompt=question,
        system_instruction=system,
        max_output_tokens=512,
        temperature=0.1,
    )

    subqs: List[str] = []
    for line in plan.splitlines():
        m = re.match(r"^\d+\.\s*(.+)$", line.strip())
        if m:
            subqs.append(m.group(1))
    if not subqs:
        subqs = [question]

    state["orc_plan"] = plan
    state["orc_subquestions"] = subqs
    return state


def node_orc_retrieve(state: AgentState, retriever: DocRetriever) -> AgentState:
    subqs = state.get("orc_subquestions", []) or [state["question"]]
    all_docs: List[Document] = []
    seen = set()

    for sq in subqs:
        for d in retriever.retrieve(sq, k=3):
            key = (d.source, d.content[:180])
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    state["retrieved_docs"] = all_docs
    return state


def node_orc_answer(state: AgentState, llm_client: LLMClient) -> AgentState:
    question = state["question"]
    plan = state.get("orc_plan", "")
    docs = state.get("retrieved_docs", [])

    ctx = "\n\n".join(
        f"[doc {i+1}] {d.source}\n{d.content}"
        for i, d in enumerate(docs)
    )

    system = (
        "You are a LangGraph/LangChain expert. "
        "Use the ORC plan + documentation to produce a structured answer."
    )

    prompt = f"Question:\n{question}\n\nPlan:\n{plan}\n\nDocs:\n{ctx}"
    state["final_answer"] = llm_client.generate(
        prompt, system_instruction=system, max_output_tokens=2048
    )
    return state


# ------------------------------------------------------------
# ReAct-style nodes: Thought / Action / Observation loop
# ------------------------------------------------------------
MAX_REACT_STEPS = 4


def _ensure_react_init(state: AgentState) -> None:
    if "scratchpad" not in state:
        state["scratchpad"] = ""
    if "iterations" not in state:
        state["iterations"] = 0


def node_react_decide(state: AgentState, llm_client: LLMClient) -> AgentState:
    """
    One ReAct decision step:
      - Observes the scratchpad (Thought/Action/Observation so far)
      - Either chooses an Action: search[...] OR Final: ...
    """
    _ensure_react_init(state)
    state["iterations"] = int(state.get("iterations", 0)) + 1

    # Safety: stop if too many steps
    if state["iterations"] > MAX_REACT_STEPS:
        system = (
            "You attempted several ReAct steps. "
            "Now produce the final best answer."
        )
        prompt = (
            f"Question:\n{state['question']}\n\n"
            f"Scratchpad:\n{state['scratchpad']}"
        )
        state["final_answer"] = llm_client.generate(
            prompt, system_instruction=system
        )
        state["react_mode"] = "finish"
        return state

    question = state["question"]
    scratchpad = state.get("scratchpad", "")

    tools_desc = (
        "You have exactly one tool you can call:\n\n"
        "Tool: search[<query>]\n"
        "  - Uses either local LangGraph/LangChain docs (offline) or web search (online)\n"
        "  - <query> should be a short description of what you want to look up\n\n"
        "Your output MUST be in one of the following formats:\n\n"
        "1) To call the tool:\n"
        "Thought: <your reasoning>\n"
        "Action: search[<query>]\n\n"
        "2) To finish with a final answer:\n"
        "Thought: <your reasoning>\n"
        "Final: <your final answer>\n"
    )

    system = (
        "You are a LangGraph/LangChain expert using the ReAct pattern "
        "(Reasoning + Acting). Think step by step, decide whether you "
        "need to call the search tool, or you can answer now."
    )

    prompt = (
        f"Question:\n{question}\n\n"
        f"Scratchpad so far (Thought/Action/Observation):\n{scratchpad}\n\n"
        f"{tools_desc}\n"
        "Decide your next step."
    )

    output = llm_client.generate(
        prompt=prompt,
        system_instruction=system,
        max_output_tokens=512,
        temperature=0.2,
    )

    # Append raw model output to scratchpad for transparency
    scratchpad += f"\n\nStep {state['iterations']} model output:\n{output}\n"
    state["scratchpad"] = scratchpad

    # Parse for Final:
    final_match = re.search(r"Final:\s*(.+)", output, re.DOTALL)
    if final_match:
        state["final_answer"] = final_match.group(1).strip()
        state["react_mode"] = "finish"
        return state

    # Parse for Action: search[...]
    action_match = re.search(r"Action:\s*search\[(.+?)\]", output, re.DOTALL)
    if action_match:
        query = action_match.group(1).strip()
        state["react_query"] = query
        state["react_mode"] = "act"
        return state

    # If we can't parse anything, just finish with the whole output as answer
    state["final_answer"] = output.strip()
    state["react_mode"] = "finish"
    return state


def node_react_act(state: AgentState, retriever: DocRetriever) -> AgentState:
    """
    Execute the chosen ReAct action: search[query].
    Append an Observation to the scratchpad.
    """
    _ensure_react_init(state)
    query = state.get("react_query", state["question"])

    docs = retriever.retrieve(query, k=3)

    # Create a compact observation string
    obs_parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        snippet = d.content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        obs_parts.append(f"[{i}] {snippet} (source={d.source})")

    obs_text = "\n".join(obs_parts) if obs_parts else "(no results)"

    scratchpad = state.get("scratchpad", "")
    scratchpad += (
        f"\nAction: search[{query}]\n"
        f"Observation:\n{obs_text}\n"
    )
    state["scratchpad"] = scratchpad

    # After acting, we go back to react_decide
    return state


def node_react_finish(state: AgentState, llm_client: LLMClient) -> AgentState:
    """
    Terminal node for ReAct branch.
    We usually already have final_answer set in node_react_decide.
    """
    # If for some reason final_answer is missing, synthesize one from scratchpad.
    if not state.get("final_answer"):
        question = state["question"]
        scratchpad = state.get("scratchpad", "")

        system = (
            "You are a LangGraph/LangChain expert. "
            "Summarize your reasoning and observations into a final answer."
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Scratchpad:\n{scratchpad}\n\n"
            "Now provide a clear final answer."
        )
        state["final_answer"] = llm_client.generate(
            prompt=prompt,
            system_instruction=system,
            max_output_tokens=1024,
            temperature=0.2,
        )
    return state


# ------------------------------------------------------------
# Routing helpers
# ------------------------------------------------------------
def route_from_analyze(state: AgentState) -> str:
    """
    Decide which branch of the graph to take based on reasoning_style.
    """
    style = state.get("reasoning_style", "basic")
    if style in ("orc", "react", "hybrid"):
        return style
    return "basic"


def route_react_next(state: AgentState) -> str:
    """
    Decide whether the ReAct loop should 'act' or 'finish' after a decision step.
    """
    mode = state.get("react_mode", "finish")
    if mode == "act":
        return "act"
    return "finish"


# ------------------------------------------------------------
# OPTIONAL: example retry-enabled nodes (not wired into main flow)
# ------------------------------------------------------------
def risky_langgraph_node(state: AgentState) -> AgentState:
    """
    Example node that always fails to demonstrate retries.
    Not wired into the main graph, but useful for testing.
    """
    raise RuntimeError("Simulated failure for retry testing")


risky_retry_node = with_retry(
    risky_langgraph_node,
    max_attempts=3,
    base_delay=1.0,
    exception_types=(RuntimeError,),
    node_name="risky_langgraph_node",
)


def call_llm_node(state: AgentState) -> AgentState:
    """
    Example node that represents a risky LLM/tool call.
    You can replace the body with a real call.
    """
    raise TimeoutError("Simulated timeout in LLM call")


llm_retry_node = RetryNode(
    base_node=call_llm_node,
    name="llm_call_node",
    max_attempts=4,
    base_delay=1.3,
    exception_types=(TimeoutError,),
).as_node()


# ------------------------------------------------------------
# Build compiled LangGraph app
# ------------------------------------------------------------
def create_agent_app(config: AppConfig):
    llm = LLMClient(config)
    retriever = build_retriever(config, llm)

    builder = StateGraph(AgentState)

    builder.set_entry_point("analyze")
    builder.add_node("analyze", node_analyze_question)

    # OPTIONAL retry examples (available to use, not in main path)
    builder.add_node("risky_node", risky_retry_node)
    builder.add_node("llm_call", llm_retry_node)

    # basic
    builder.add_node("retrieve_basic", lambda s: node_retrieve_docs(s, retriever))
    builder.add_node("draft_basic", lambda s: node_draft_answer(s, llm))

    # ORC
    builder.add_node("orc_plan", lambda s: node_orc_plan(s, llm))
    builder.add_node("orc_retrieve", lambda s: node_orc_retrieve(s, retriever))
    builder.add_node("orc_answer", lambda s: node_orc_answer(s, llm))

    # ReAct
    builder.add_node("react_decide", lambda s: node_react_decide(s, llm))
    builder.add_node("react_act", lambda s: node_react_act(s, retriever))
    builder.add_node("react_finish", lambda s: node_react_finish(s, llm))

    # branch selection
    builder.add_conditional_edges(
        "analyze",
        route_from_analyze,
        {
            "basic": "retrieve_basic",
            "orc": "orc_plan",
            "react": "react_decide",
            "hybrid": "orc_plan",
        },
    )

    # basic path
    builder.add_edge("retrieve_basic", "draft_basic")
    builder.add_edge("draft_basic", END)

    # ORC path
    builder.add_edge("orc_plan", "orc_retrieve")
    builder.add_edge("orc_retrieve", "orc_answer")
    builder.add_edge("orc_answer", END)

    # ReAct path
    builder.add_conditional_edges(
        "react_decide",
        route_react_next,
        {"act": "react_act", "finish": "react_finish"},
    )
    builder.add_edge("react_act", "react_decide")
    builder.add_edge("react_finish", END)

    return builder.compile()


def run_agent_once(app, config: AppConfig, question: str) -> str:
    initial: AgentState = {
        "question": question,
        "mode": config.mode,
        "reasoning_style": config.reasoning_style,
    }
    final = app.invoke(initial)
    return final.get("final_answer", "(no answer)")
