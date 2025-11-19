
# **LangGraph Helper Agent — Documentation**

A developer-facing AI assistant that explains and demonstrates **LangGraph**, **LangChain**, persistence, retries, state machines, and advanced orchestration patterns.

The system runs in two modes:

* **Offline mode:** local RAG over pre-indexed documentation
* **Online mode:** live web search + offline fallback


---

# Built in Python and orchestrated with **LangGraph**.
1. Architecture Overview**

Built in Python and orchestrated with **LangGraph**.


Below is the high-level architecture showing the flow from CLI → Config → LangGraph → Retrieval → Final Answer:

```
┌────────────────────────┐
│         CLI            │
│   main.py entrypoint   │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│       ConfigLoader     │
│ (AppConfig + env vars) │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│      LLMClient         │
│  (Gemini text/embed)   │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│    Retriever Builder   │───────┐
│ offline / online mode  │       │
└────────────┬───────────┘       │
             │                   │
             ▼                   │
      ┌──────────────┐    ┌───────────────┐
      │ Offline RAG  │    │ Online Search │
      │ embeddings   │    │ Tavily / DDG  │
      └──────────────┘    └───────────────┘
             │                    │
             └──────────┬────────┘
                        ▼
              Provides docs → LangGraph
```

---

## **LangGraph Workflow**

The core LangGraph agent uses four orchestrators:

* **basic** → simple RAG + answer
* **ORC** → planning, retrieval per subquestion, structured answer
* **ReAct** → Thought/Action/Observation loop
* **hybrid** → ORC entry but extensible

```
                           ┌───────────────┐
                           │   analyze     │
                           │ (select path) │
                           └───────┬───────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────────┐        ┌─────────────────┐
│   basic      │          │       ORC        │        │     ReAct       │
└──────┬───────┘          └─────────┬────────┘        └────────┬────────┘
       │                            │                          │
       ▼                            ▼                          ▼
retrieve_basic → draft_basic      orc_plan → orc_retrieve → orc_answer    react_decide → react_act ↔ react_decide → react_finish
       │                            │                           │
       └────────────── END ◄────────╯                           └────── END
```

---

## **Retry Architecture**

The project includes two retry wrappers:

```
with_retry(fn)           RetryNode(base_node)
```

Retry behavior uses:

* exponential backoff
* jitter
* attach error to state instead of crashing

Diagram:

```
         ┌───────────────┐
         │  Node (fn)    │
         └───────┬───────┘
                 │
                 ▼
     ┌──────────────────────────┐
     │  Retry Wrapper (3–4 tries│
     │  backoff + jitter        │
     └───────────┬──────────────┘
                 │ success
                 ▼
              Next node
                 │ failure (max)
                 ▼
      state["error"] added safely
```

---

# **2. Operating Modes**

## **Offline Mode (Local RAG)**

```
AGENT_MODE=offline
```

Characteristics:

* No internet
* Still uses Gemini API (allowed)
* RAG runs on **local vector index**
* Deterministic and fast

**Index files used:**

```
data/index/embeddings.npy
data/index/docs.jsonl
```

---

## **Online Mode (Web + RAG Hybrid)**

```
AGENT_MODE=online
```

Uses:

* Tavily (if API key available)
* DuckDuckGo (ddgs or duckduckgo-search)
* Scoring:

  * boosts LangChain/LangGraph domains
  * keyword boosting
* Low-quality detection → automatic offline fallback

### Why?

* Fresh documentation
* Can reference latest LangGraph/LangChain releases
* Enables advanced ReAct workflows that query the web

---

## **Switching Modes**

CLI:

```
--mode offline
--mode online
```

Environment:

```
AGENT_MODE=offline
AGENT_MODE=online
```

---

# **3. Data Freshness Strategy**

## **Offline Data Freshness**

Offline docs come from `.txt` snapshots stored in:

```
data/raw/*.txt
```

To update the offline dataset:

1. Replace/append new `.txt` docs in `data/raw/`
2. Rebuild index:

```
python -m src.helper_agent.prepare_docs
```

### Indexing steps:

```
raw text
   │
   ▼
chunk_text()  (safe: ≤30kB for Gemini)
   │
dedup via blake2b hash
   │
Gemini embedding model
   │
save → embeddings.npy + docs.jsonl
```

ASCII diagram:

```
┌──────────────┐
│ raw txt docs │
└───────┬──────┘
        ▼
 Chunker (safe byte limits)
        ▼
Deduper ──────────┐
        ▼         │
Embedder (Gemini) │
        ▼         │
Save JSONL + NPY  │
        ▼         │
Local RAG index ◄─┘
```

---

## **Online Data Freshness**

Online mode fetches live docs from web.

Priority:

1. Tavily → preferred if key
2. DDG → free fallback
3. Offline RAG → if results weak

Scoring ensures results come from:

* `langchain-ai.github.io`
* `python.langchain.com`
* `langgraph.readthedocs.io`
* GitHub LangChain repos

---

# **4. Setup Instructions**

## **1. Clone and install**

```
git clone https://github.com/morgalut/Opsfleet_task.git
cd Opsfleet_task
```

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## **2. Add `.env`**

```
GEMINI_API_KEY=your_key_here
```

---

## **3. Build offline index**

```
python -m src.helper_agent.prepare_docs
```

---

## **4. Run examples**

### Basic RAG

```
python main.py --mode offline --strategy basic \
  "What is the difference between StateGraph and MessageGraph?"
```

### ORC Planning

```
python main.py --mode online --strategy orc \
  "How do I add persistence to a LangGraph agent?"
```

### ReAct tool use

```
python main.py --mode online --strategy react \
  "Show me human-in-the-loop with LangGraph"
```

### Hybrid + Debug

```
export AGENT_REASONING=hybrid

python main.py --mode online --strategy hybrid --debug \
  "How do I handle retries in LangGraph nodes?"
```

### Fully offline

```
python main.py --mode offline \
  "How do I handle retries in LangGraph nodes?"
```

---

# **5. Version Requirements**

| Component                | Version       |
| ------------------------ | ------------- |
| Python                   | 3.10+         |
| LangGraph                | Latest stable |
| LangChain-Core           | Latest stable |
| google-generativeai      | ≥ 0.7.0       |
| numpy                    | ≥ 1.24        |
| ddgs / duckduckgo-search | optional      |
| tqdm                     | ≥ 4.0         |
| python-dotenv            | required      |

---

# **6. Summary**

This project provides:

- A two-mode intelligent LangGraph agent  
- Offline RAG with a clean vector index  
- Online search with scoring and fallback  
- Gemini LLM integration with automatic retry and truncation recovery  
- ORC planner, ReAct agent, and a hybrid reasoning pipeline  
- A full retry framework for LangGraph nodes  
- Easy portability and environment setup  





