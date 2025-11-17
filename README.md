# LangGraph Helper Agent

## 1. What you’re actually being asked to build

You’re building a developer-facing Q&A agent that helps people work with **LangGraph** and **LangChain**, with two distinct runtime modes:

### Offline mode

- No internet / web access.
- But you **can** still call LLMs via API (Google Gemini).
- Knowledge comes from **locally stored docs**, starting with the `llms.txt`-based LangGraph & LangChain documentation (or `llms-full.txt` snapshot).
- You must define how you **downloaded, cleaned, and indexed** that data.
- If you add more docs, you must explain how they **stay fresh**.

### Online mode

- Internet allowed.
- Agent can use **web search / APIs** to fetch up-to-date docs or examples (e.g. Tavily, DuckDuckGo, Exa, etc.), all on free tiers.
- README must clearly say **what external services** you use, how to get their API keys, and why you picked them.

In both modes, the agent is supposed to answer **practical, technical questions** about using LangGraph / LangChain, including reasonably advanced topics:

- Persistence & checkpointers  
- StateGraph vs MessageGraph  
- Human-in-the-loop flows  
- Error handling & retries  
- Best practices for state management, etc.

Additional constraints:

- **LLM:** must use **Google Gemini** via the free API (from Google AI Studio).
- **Language:** Python.
- **Orchestration:** platform of your choice (this project uses **LangGraph**).
- **Portability:** project must run easily on another machine (clean env setup, clear instructions).

Deliverables (via a public GitHub repository):

- Working code (both offline and online modes).
- Documentation (README) covering:
  - Architecture
  - Operating modes
  - Data strategy and update/freshness
  - Setup and example usage

> The rest of this README will describe:
>
> - Architecture overview (graph design, state, nodes)
> - Offline vs online mode implementations
> - Data preparation and update strategy
> - Setup instructions and example commands

```sh
python -m src.helper_agent.prepare_docs
```

```sh
# Basic (original) pipeline
python main.py --mode offline --strategy basic \
  "What is the difference between StateGraph and MessageGraph?"

python main.py --mode online --strategy basic \
  "What is the difference between StateGraph and MessageGraph?"
```

---
```sh
# ORC-style planning
python main.py --mode offline --strategy orc \
  "How do I add persistence to a LangGraph agent?"

python main.py --mode online --strategy orc \
  "How do I add persistence to a LangGraph agent?"
```
---
```sh
# ReAct-style reasoning
python main.py --mode online --strategy react \
  "Show me human-in-the-loop with LangGraph"


python main.py --mode online --strategy react \
  "Show me human-in-the-loop with LangGraph"
```

```sh
# Hybrid (currently = ORC pipeline entry; you can extend)
export AGENT_REASONING=hybrid
## debug - Debug mode shows the agents' thought processes and how the system works.
python main.py --mode online --strategy hybrid --debug \
  "How do I handle retries in LangGraph nodes?"


python main.py --mode offline \
  "How do I handle retries in LangGraph nodes?"


python main.py --mode online --strategy hybrid \
  "How do I handle retries in LangGraph nodes?"

```