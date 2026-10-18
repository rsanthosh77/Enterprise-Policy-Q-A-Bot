import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pinecone import DenseVectorQuery, SparseVectorQuery, SparseValues
from pinecone._internal.documents_helpers import _build_search_documents_body
from rag.corpus import ROOT, load_corpus
from rag.engine import Engine, Settings, fuse, Answer, Claim, Citation, Verification, embedding_text
from rag.evaluate import section_metrics, cases
from rag.sparse import BM25


def test_chunks_preserve_exact_source_and_all_lines():
    chunks = load_corpus()
    assert len({c['metadata']['document_id'] for c in chunks}) == 4
    covered = {}
    for c in chunks:
        m = c['metadata']
        lines = (ROOT / m['source_path']).read_text().splitlines()
        assert c['text'] == '\n'.join(lines[m['line_start'] - 1:m['line_end']]).strip()
        covered.setdefault(m['filename'], set()).update(range(m['line_start'] - 1, m['line_end']))
    for name, indices in covered.items():
        lines = (ROOT / 'Sample Documents' / name).read_text().splitlines()
        assert all(i in indices for i, line in enumerate(lines) if line.strip())
    assert [c['id'] for c in chunks] == [c['id'] for c in load_corpus()]
    onboarding = [c for c in chunks if c['metadata']['document_id'] == 'onboarding_guide']
    assert all(c['metadata']['section_number'] == 'preamble' or '.' in c['metadata']['section_number'] for c in onboarding)


def test_saved_gold_sections_exist():
    chunks = load_corpus()
    present = {(c['metadata']['document_id'], c['metadata']['section_number']) for c in chunks}
    for case in cases():
        assert set(map(tuple, case['expected_sections'])) <= present


def test_sparse_exact_error_code_and_unknown_vocabulary():
    bm = BM25(['Retry-After HTTP 429 jitter', 'PTO carryover'])
    query = bm.query('HTTP 429')
    scores = []
    for text in ['Retry-After HTTP 429 jitter', 'PTO carryover']:
        vector = bm.document(text)
        values = dict(zip(vector['indices'], vector['values']))
        scores.append(sum(values.get(i, 0) * w for i, w in zip(query['indices'], query['values'])))
    assert scores[0] > scores[1] == 0
    assert bm.query('unseenword')['indices'] == []
    loaded = BM25.__new__(BM25)
    loaded.__dict__.update(json.loads(json.dumps(bm.__dict__)))
    assert loaded.query('429') == bm.query('429')


def test_fusion_preserves_union_and_endpoint_modes():
    dense = [{'id': 'a', 'score': .8}, {'id': 'b', 'score': .7}]
    sparse = [{'id': 'c', 'score': 50}, {'id': 'b', 'score': 3}]
    assert {c['id'] for c in fuse(dense, sparse)} == {'a', 'b', 'c'}
    assert [c['id'] for c in fuse(dense, sparse, 1)] == ['a', 'b']
    assert [c['id'] for c in fuse(dense, sparse, 0)] == ['c', 'b']
    assert fuse(dense, sparse)[0]['id'] == 'b'


def test_sdk_search_wire_shapes():
    for clause in [dict(type='dense_vector', field='dense', values=[.1, .2]),
                   dict(type='sparse_vector', field='sparse', sparse_values={'indices': [1], 'values': [.5]})]:
        body = _build_search_documents_body(score_by=[clause], top_k=10, include_fields=['chunk_id'], filter=None)
        assert body['top_k'] == 10


def test_section_recall_requires_all_conditions():
    c = {'metadata': {'document_id': 'hr_policy', 'section_number': '3.1'}}
    assert section_metrics([['hr_policy', '3.1'], ['hr_policy', '3.5']], [c]) == (.5, False)


def test_answer_rejects_fabricated_quotes(monkeypatch):
    chunk = load_corpus()[0]
    answer = Answer(status='answered', claims=[Claim(text='Invented', citations=[Citation(source=1, quote='FABRICATED QUOTE')])], conflicts=[])
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = answer
    monkeypatch.setattr('langchain_anthropic.ChatAnthropic', Mock(return_value=model))
    engine = Engine.__new__(Engine)
    assert engine.answer('Question', {'chunks': [chunk]})['status'] == 'unknown'


def test_answer_rejects_unsupported_inference(monkeypatch):
    chunk = load_corpus()[0]
    answer = Answer(status='answered', claims=[Claim(text='Unjustified inference', citations=[Citation(source=1, quote=chunk['text'][:50])])], conflicts=[])
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = [answer, Verification(supported=False)]
    monkeypatch.setattr('langchain_anthropic.ChatAnthropic', Mock(return_value=model))
    assert Engine.__new__(Engine).answer('Question', {'chunks': [chunk]})['status'] == 'unknown'


def test_retrieval_filters_and_section_expansion():
    chunks = load_corpus(max_chars=200)
    chosen = next(c for c in chunks if c['metadata']['document_id'] == 'hr_policy' and c['metadata']['section_number'] == '3.1')
    engine = Engine.__new__(Engine)
    engine.manifest = {'chunks': chunks, 'namespace': 'test'}
    engine.by_id = {c['id']: c for c in chunks}
    engine.encoder = BM25([embedding_text(c) for c in chunks])
    engine.embeddings = Mock()
    engine.embeddings.embed_query.return_value = [.1]
    engine.index = Mock()
    engine.index.documents.search.return_value = SimpleNamespace(matches=[SimpleNamespace(id=chosen['id'], score=.7)])
    result = engine.retrieve('remote eligibility', Settings(candidate_k=10, context_k=1), ['hr'])
    assert len(result['chunks']) > 1
    assert all(c['metadata']['section_number'] == '3.1' for c in result['chunks'])
    for call in engine.index.documents.search.call_args_list:
        assert call.kwargs['filter'] == {'document_type': {'$in': ['hr']}}
        assert call.kwargs['namespace'] == 'test'
