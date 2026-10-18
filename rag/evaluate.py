import json
import math
import time
from dataclasses import asdict
from rag.corpus import ROOT, save_json
from rag.engine import Settings

PRESETS = {
    'dense': Settings(dense_weight=1),
    'sparse': Settings(dense_weight=0),
    'hybrid': Settings(),
    'hybrid-reranked': Settings(rerank=True),
}

def cases():
    return json.loads((ROOT / 'evaluation.json').read_text())

def section_metrics(expected, chunks):
    expected = {tuple(s) for s in expected}
    actual = {(c['metadata']['document_id'], c['metadata']['section_number']) for c in chunks}
    return (len(expected & actual) / len(expected), expected <= actual) if expected else (None, None)

def ranked_metrics(expected, chunks, k=12):
    """Binary relevance over unique document/section pairs; fixed K precision."""
    if k < 1:
        raise ValueError('k must be positive')
    gold = {tuple(s) for s in expected}
    names = ('recall_at_k', 'precision_at_k', 'mrr', 'ndcg_at_k')
    if not gold:
        return dict.fromkeys(names)
    ranked = list(dict.fromkeys((c['metadata']['document_id'], c['metadata']['section_number']) for c in chunks))[:k]
    relevance = [int(section in gold) for section in ranked]
    hits = sum(relevance)
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return {'recall_at_k': hits / len(gold), 'precision_at_k': hits / k,
            'mrr': next((1 / (rank + 1) for rank, rel in enumerate(relevance) if rel), 0),
            'ndcg_at_k': dcg / ideal}


def evaluate(engine, presets=None, generate=False, progress=None):
    rows = []
    for name in presets or ['dense', 'sparse', 'hybrid']:
        setting = PRESETS[name]
        for case in cases():
            start = time.monotonic()
            retrieval = engine.retrieve(case['question'], setting)
            recall, complete = section_metrics(case['expected_sections'], retrieval['chunks'])
            candidate_recall, _ = section_metrics(case['expected_sections'], retrieval['candidates'])
            rank_metrics = ranked_metrics(case['expected_sections'], retrieval['candidates'], setting.context_k)
            row = {**rank_metrics, 'metric_k': setting.context_k, 'preset': name, 'case': case['id'], 'question': case['question'],
                   'section_recall': recall, 'all_required_sections': complete,
                   'candidate_recall': candidate_recall, 'retrieval_seconds': retrieval['retrieval_seconds'],
                   'retrieved_sections': sorted({c['metadata']['document_id'] + ':' + c['metadata']['section_number'] for c in retrieval['chunks']})}
            if generate:
                answer = engine.answer(case['question'], retrieval)
                text = ' '.join(c['text'] for c in answer['claims'] + answer['conflicts']).lower()
                row.update(answer=answer, status_matches=answer['status'] == case['expected_status'],
                           required_terms_coverage=sum(t.lower() in text for t in case['required_terms']) / len(case['required_terms']) if case['required_terms'] else None)
            row['total_seconds'] = time.monotonic() - start
            rows.append(row)
            if progress:
                progress(f'{name}: {case["id"]}')
    report = {'namespace': engine.manifest['namespace'], 'generate': generate,
              'presets': {n: asdict(PRESETS[n]) for n in presets or ['dense', 'sparse', 'hybrid']}, 'rows': rows}
    stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    save_json(ROOT / 'data' / f'evaluation-{stamp}.json', report)
    return report
