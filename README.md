# 🧪 Promptfoo RAG Validation — Conversational AI Chatbot

A production-ready validation suite for **LLM/RAG chatbots**, testing both
**OpenAI GPT-4o** and **Anthropic Claude Sonnet** side-by-side using
[Promptfoo](https://promptfoo.dev).

---

## 📁 Repo Structure

```
promptfoo-rag-validation/
├── promptfooconfig.yaml        # Main Promptfoo config (providers + tests)
├── prompts/
│   └── rag_system_prompt.txt   # System prompt template with {{variables}}
├── providers/
│   └── rag_provider.py         # Python RAG pipeline (retrieval + LLM call)
├── tests/
│   ├── happy_path.yaml         # Core correct-answer tests
│   ├── edge_cases.yaml         # Boundary & stress conditions
│   ├── hallucination.yaml      # Fabrication detection
│   ├── safety.yaml             # Prompt injection & adversarial tests
│   └── test_rag_provider.py    # Pytest unit tests for the provider
├── scripts/
│   └── run_eval.py             # CLI runner with human-readable summary
├── results/                    # Eval output (gitignored)
├── .env.example
├── package.json
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Clone & install dependencies

```bash
git clone <your-repo>
cd promptfoo-rag-validation

# Python deps
pip install -r requirements.txt

# Promptfoo (Node.js required)
npm install
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

Promptfoo auto-reads `.env`, or export keys in your shell:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the full evaluation

```bash
# Option A: via Python runner (with pretty summary)
python scripts/run_eval.py

# Option B: via npm scripts
npm run eval

# Option C: directly
npx promptfoo eval
```

### 4. View interactive report

```bash
npx promptfoo view
# Opens a browser UI with side-by-side comparison of GPT-4o vs Claude
```

---

## 🔬 Test Suites

| Suite | File | What it validates |
|---|---|---|
| Happy Path | `tests/happy_path.yaml` | Correct answers, multi-turn coherence |
| Edge Cases | `tests/edge_cases.yaml` | Out-of-scope, long history, ambiguity |
| Hallucination | `tests/hallucination.yaml` | Model stays within context, no fabrication |
| Safety | `tests/safety.yaml` | Prompt injection, PII handling, jailbreaks |

Run a single suite:
```bash
python scripts/run_eval.py --test happy
python scripts/run_eval.py --test hallucination
npm run eval:safety
```

---

## 🤖 Providers

| Label | Model | Notes |
|---|---|---|
| GPT-4o | `openai:gpt-4o` | Used for generation + LLM-as-judge |
| Claude Sonnet | `anthropic:claude-sonnet-4-5` | Tested in parallel |

Both use `temperature: 0.2` for deterministic outputs.

---

## 🔌 Swapping in Your Real RAG Pipeline

The mock retrieval in `providers/rag_provider.py` uses keyword matching.
Replace the `retrieve_context()` function body with your real retriever:

```python
# Example: Pinecone
import pinecone
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question: str, top_k: int = 2) -> str:
    embedding = model.encode(question).tolist()
    index = pinecone.Index("your-index-name")
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return "\n\n".join(m["metadata"]["text"] for m in results["matches"])
```

```python
# Example: ChromaDB
import chromadb

client = chromadb.Client()
collection = client.get_collection("your-collection")

def retrieve_context(question: str, top_k: int = 2) -> str:
    results = collection.query(query_texts=[question], n_results=top_k)
    return "\n\n".join(results["documents"][0])
```

---

## 🧪 Unit Tests

```bash
pytest tests/test_rag_provider.py -v
```

Tests cover:
- Retrieval correctness and fallback
- Prompt template substitution
- OpenAI / Anthropic routing
- Error handling
- Context variable override

---

## 📊 Assertion Types Used

| Type | Purpose |
|---|---|
| `contains` | Fast string match for critical values |
| `not-contains` | Ensure hallucinated content is absent |
| `llm-rubric` | GPT-4o judges quality of full response |

---

## 📈 Extending the Suite

Add new test cases to any YAML file following this pattern:

```yaml
- description: "Your test description"
  vars:
    context: |
      Paste the retrieved context chunk here.
    history: |
      Prior conversation turns (or leave blank).
    question: "The user's message"
  assert:
    - type: contains
      value: "expected keyword"
    - type: llm-rubric
      value: "Describe what a correct answer looks like."
    - type: not-contains
      value: "text that should NOT appear"
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for GPT-4o generation + LLM judge |
| `ANTHROPIC_API_KEY` | Yes | Used for Claude Sonnet generation |

---

## 📄 License

MIT
