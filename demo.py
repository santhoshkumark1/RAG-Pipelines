"""
demo.py
=======
Runs the RAG pipeline end-to-end on a small sample corpus, with one
answerable question and one unanswerable ("false premise") question, to show
the abstain-when-unsure behavior.

Run with the offline MockLLMClient (no setup needed):
    python demo.py

Run with a real LLM for much better answer quality:
    python demo.py --backend openai --api-key sk-...       [--model gpt-4o-mini]
    python demo.py --backend anthropic --api-key sk-ant-... [--model claude-sonnet-5]
"""

import argparse
import json

from rag_pipeline import (
    Passage, MockLLMClient, OpenAIClient, AnthropicClient, build_pipeline,
)

SAMPLE_PASSAGES = [
    Passage(
        id="p1",
        title="Scott Derrickson",
        text=(
            "Scott Derrickson (born July 16, 1966) is an American director, "
            "screenwriter and producer. He is best known for horror films such "
            "as Sinister and The Exorcism of Emily Rose, and later directed the "
            "Marvel Studios film Doctor Strange."
        ),
    ),
    Passage(
        id="p2",
        title="Ed Wood",
        text=(
            "Edward Davis Wood Jr. was an American filmmaker, actor, writer, "
            "producer, and director. He is known for his low-budget science "
            "fiction and horror films, and his work was the subject of a 1994 "
            "biographical film directed by Tim Burton."
        ),
    ),
    Passage(
        id="p3",
        title="Ed Wood (film)",
        text=(
            "Ed Wood is a 1994 American biographical period comedy-drama film "
            "directed and produced by Tim Burton, starring Johnny Depp as the "
            "cult filmmaker Ed Wood. The film focuses on the period of Wood's "
            "life when he made his best-known films."
        ),
    ),
    Passage(
        id="p4",
        title="Marie Curie",
        text=(
            "Marie Curie was a physicist and chemist who conducted pioneering "
            "research on radioactivity. She was the first woman to win a Nobel "
            "Prize and the only person to win Nobel Prizes in two different "
            "scientific fields, physics and chemistry."
        ),
    ),
]

QUESTIONS = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What was the name of the spaceship Marie Curie flew to the Moon?",
]


def make_llm(args):
    if args.backend == "openai":
        return OpenAIClient(api_key=args.api_key, model=args.model or "gpt-4o-mini")
    if args.backend == "anthropic":
        return AnthropicClient(api_key=args.api_key, model=args.model or "claude-sonnet-5")
    return MockLLMClient()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mock", "openai", "anthropic"], default="mock")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm = make_llm(args)
    print(f"[setup] backend={args.backend} contextualize=True")

    agent = build_pipeline(SAMPLE_PASSAGES, llm, contextualize=(args.backend != "mock"))
    print(f"[index] built hybrid index over sample corpus ({len(SAMPLE_PASSAGES)} passages)\n")

    for q in QUESTIONS:
        result = agent.answer(q)
        final = result["final"]
        print(f"Q: {q}")
        print(f"  route={result['route']} hops={result.get('hops', 0)} "
              f"status={final.status} reason={final.reason} min_support={final.min_support:.2f}")
        print(f"  A: {final.answer}")
        if final.citations:
            print(f"  citations: {final.citations}")
        print()


if __name__ == "__main__":
    main()
