"""Opt-in live integration check. Run from the project root; uses account credits."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.corpus import ROOT, save_json
from rag.engine import Engine
from rag.evaluate import evaluate


def main():
    engine = Engine()
    report = evaluate(engine, ['dense', 'sparse', 'hybrid'], generate=False,
                      progress=lambda message: print(message, flush=True))
    rows = report['rows']
    for preset in ('dense', 'sparse', 'hybrid'):
        recalls = [r['section_recall'] for r in rows if r['preset'] == preset and r['section_recall'] is not None]
        print(f'{preset} mean section recall: {sum(recalls) / len(recalls):.3f}', flush=True)
    checks = []
    for question, expected in [
        ('How should my integration handle HTTP 429?', 'answered'),
        ('I joined 95 days ago. Can I work remotely from Singapore?', 'answered'),
        ('What is the exact travel per diem rate for Singapore?', 'unknown'),
    ]:
        retrieval = engine.retrieve(question)
        answer = engine.answer(question, retrieval)
        checks.append({'question': question, 'expected_status': expected, 'answer': answer,
                       'retrieval': retrieval})
        print(f'Answer check: {answer["status"]}; expected: {expected}', flush=True)
    save_json(ROOT / 'data' / 'claude-live-smoke.json', {'namespace': engine.manifest['namespace'], 'checks': checks})
    if any(c['answer']['status'] != c['expected_status'] for c in checks):
        raise ValueError('Live answer check failed. Inspect data/claude-live-smoke.json.')
    print('Live Claude answers, citations, and missing-evidence abstention passed.', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        message = str(exc)
        for key in ('PINECONE_API_KEY', 'ANTHROPIC_API_KEY'):
            if os.getenv(key):
                message = message.replace(os.environ[key], '[REDACTED]')
        print(type(exc).__name__ + ': ' + message[:1500])
        raise SystemExit(1)
