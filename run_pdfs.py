import argparse
import glob
import os

from pypdf import PdfReader

from rag_pipeline import Passage, MockLLMClient, OpenAIClient, AnthropicClient, build_pipeline


def load_pdfs(folder: str) -> list[Passage]:
    passages = []
    pdf_files = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdf_files:
        print(f"[warning] no PDF files found in '{folder}/'")
        return passages

    for i, path in enumerate(pdf_files):
        title = os.path.splitext(os.path.basename(path))[0]
        try:
            reader = PdfReader(path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            text = text.strip()
        except Exception as e:
            print(f"[warning] failed to read {path}: {e}")
            continue

        if len(text) < 40:
            print(f"[warning] '{title}' has almost no extractable text (skipped) "
                  f"-- it may be a scanned/image PDF that needs OCR")
            continue

        passages.append(Passage(id=f"doc{i}", title=title, text=text))
        print(f"[loaded] {title}  ({len(text)} characters)")

    return passages


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
    parser.add_argument("--folder", default="pdfs")
    parser.add_argument("--no-context", action="store_true",
                         help="Skip the per-chunk LLM contextualizing step (much faster indexing).")
    args = parser.parse_args()

    print(f"[setup] loading PDFs from '{args.folder}/' ...")
    passages = load_pdfs(args.folder)
    if not passages:
        print("[error] no usable documents found -- add PDFs to the folder and try again.")
        return

    llm = make_llm(args)
    do_contextualize = (args.backend != "mock") and not args.no_context
    print(f"\n[setup] backend={args.backend}  contextualize={do_contextualize}")
    if do_contextualize:
        print("[index] building hybrid index -- this makes one LLM call per chunk "
              "(can take a few minutes for large PDFs). Use --no-context to skip this and index instantly.")
    agent = build_pipeline(passages, llm, contextualize=do_contextualize)
    print(f"[index] ready -- {len(passages)} documents indexed.\n")

    print("Type a question and press Enter. Type 'quit' to exit.\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        result = agent.answer(q)
        final = result["final"]
        print(f"  status={final.status} reason={final.reason} min_support={final.min_support:.2f}")
        print(f"  A: {final.answer}")
        if final.citations:
            print(f"  citations: {final.citations}")
        print()


if __name__ == "__main__":
    main()
