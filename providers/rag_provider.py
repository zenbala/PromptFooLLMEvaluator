"""
providers/rag_provider.py
─────────────────────────
A Promptfoo-compatible Python provider that simulates a RAG pipeline:
  1. Retrieves relevant context chunks (mock vector store here — swap for
     your real retriever: Pinecone, Chroma, pgvector, etc.)
  2. Injects context into the prompt template
  3. Calls the configured LLM (OpenAI or Anthropic)

Usage in promptfooconfig.yaml:
  providers:
    - id: "python:providers/rag_provider.py"
      config:
        model: openai   # or "anthropic"
"""

import os
import json
import re
import sys
from typing import Any

# ── Mock knowledge base (replace with your vector DB retrieval) ──────────────
KNOWLEDGE_BASE = [
    {
        "id": "kb-001",
        "text": "Our return policy allows returns within 30 days of purchase. "
                "Items must be unused and in original packaging. "
                "Refunds are processed within 5–7 business days.",
        "keywords": ["return", "refund", "policy", "days"],
    },
    {
        "id": "kb-002",
        "text": "The ProPlan subscription includes: unlimited storage, priority support, "
                "advanced analytics dashboard, and up to 10 team members. "
                "Price: $49/month billed annually or $59/month billed monthly.",
        "keywords": ["pro", "plan", "proplan", "storage", "analytics", "team"],
    },
    {
        "id": "kb-003",
        "text": "Basic Plan: $9/month. Pro Plan: $49/month (annual) or $59/month (monthly). "
                "Enterprise: contact sales for custom pricing.",
        "keywords": ["basic", "price", "cost", "pricing", "enterprise", "plan"],
    },
    {
        "id": "kb-004",
        "text": "Cancellation policy: Users can cancel anytime. Cancellation takes effect "
                "at the end of the current billing cycle. No partial refunds.",
        "keywords": ["cancel", "cancellation", "billing", "cycle"],
    },
    {
        "id": "kb-005",
        "text": "AcmeCorp provides SaaS project management tools. "
                "We offer Basic, Pro, and Enterprise plans. "
                "For support, visit our Help Center at help.acmecorp.com.",
        "keywords": ["acmecorp", "support", "help", "plans", "saas"],
    },
]


def retrieve_context(question: str, top_k: int = 2) -> str:
    """
    Naive keyword-based retrieval — swap this function body for your
    real vector similarity search (e.g. cosine similarity on embeddings).
    """
    q_lower = question.lower()
    scored = []
    for doc in KNOWLEDGE_BASE:
        score = sum(1 for kw in doc["keywords"] if kw in q_lower)
        if score > 0:
            scored.append((score, doc["text"]))

    scored.sort(reverse=True)
    top = [text for _, text in scored[:top_k]]

    if not top:
        return "No specific documentation found for this topic."
    return "\n\n".join(top)


def load_prompt_template(path: str = "prompts/rag_system_prompt.txt") -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        # Fallback inline template
        return (
            "You are a helpful support chatbot. Use ONLY the context below.\n\n"
            "Context:\n{{context}}\n\nHistory:\n{{history}}\n\nUser: {{question}}\nAssistant:"
        )


def build_prompt(template: str, context: str, history: str, question: str) -> str:
    return (
        template
        .replace("{{context}}", context)
        .replace("{{history}}", history or "(none)")
        .replace("{{question}}", question)
    )


def call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def call_anthropic(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── Promptfoo entry point ────────────────────────────────────────────────────

def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Called by Promptfoo for each test case.
    `prompt`  — the rendered prompt string (after variable substitution)
    `options` — provider config from promptfooconfig.yaml
    `context` — test vars and metadata
    """
    try:
        vars_ = context.get("vars", {})
        question = vars_.get("question", prompt)
        history = vars_.get("history", "")

        # If context var is provided in the test, use it; otherwise retrieve
        ctx_override = vars_.get("context", "")
        retrieved_context = ctx_override if ctx_override else retrieve_context(question)

        template = load_prompt_template()
        full_prompt = build_prompt(template, retrieved_context, history, question)

        model_backend = (options.get("config") or {}).get("model", "openai")

        if model_backend == "anthropic":
            output = call_anthropic(full_prompt)
        else:
            output = call_openai(full_prompt)

        return {
            "output": output,
            "metadata": {
                "retrieved_context": retrieved_context,
                "model_backend": model_backend,
            },
        }

    except Exception as e:
        return {"error": str(e), "output": ""}


# ── CLI: run a quick manual test ─────────────────────────────────────────────

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is your return policy?"
    print(f"\n🔍 Question: {question}")

    ctx = retrieve_context(question)
    print(f"\n📄 Retrieved context:\n{ctx}\n")

    template = load_prompt_template()
    prompt = build_prompt(template, ctx, "", question)
    print(f"📝 Prompt sent to LLM:\n{prompt}\n")

    result = call_api(prompt, {"config": {"model": "openai"}}, {"vars": {"question": question}})
    print(f"🤖 Response:\n{result.get('output', result.get('error'))}\n")
