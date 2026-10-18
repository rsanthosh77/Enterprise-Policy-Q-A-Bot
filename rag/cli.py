import argparse
import json
from rag.corpus import ROOT, load_corpus, save_json
from rag.engine import Engine, ingest
from rag.evaluate import PRESETS, evaluate

def main():
    parser = argparse.ArgumentParser(description='Enterprise RAG starter')
    parser.add_argument('action', choices=['preview', 'ingest', 'evaluate'])
    parser.add_argument('--generate', action='store_true', help='Evaluate cited answers and abstention too (additional LLM calls)')
    parser.add_argument('--presets', nargs='+', choices=list(PRESETS), default=['dense', 'sparse', 'hybrid'])
    args = parser.parse_args()
    if args.action == 'preview':
        chunks = load_corpus()
        save_json(ROOT / 'data' / 'preview.json', chunks)
        print(f'{len(chunks)} chunks from {len({c["metadata"]["document_id"] for c in chunks})} documents. Preview: data/preview.json')
    elif args.action == 'ingest':
        m = ingest()
        print(f'Active snapshot: {m["namespace"]}; {len(m["chunks"])} chunks')
    else:
        print(json.dumps(evaluate(Engine(), args.presets, args.generate, print), indent=2))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, TimeoutError) as exc:
        raise SystemExit(str(exc))
