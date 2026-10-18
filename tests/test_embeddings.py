from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rag.embeddings import PineconeEmbeddings
from rag.engine import credentials


def test_pinecone_passage_query_modes_and_batching():
    pc = Mock()
    def embed(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(values=[.1, .2]) for _ in kwargs['inputs']])
    pc.inference.embed.side_effect = embed
    embeddings = PineconeEmbeddings(pc, dimensions=2)
    assert len(embeddings.embed_documents(['document'] * 65)) == 65
    assert len(pc.inference.embed.call_args_list) == 2
    assert pc.inference.embed.call_args_list[0].kwargs['parameters'] == {'input_type': 'passage', 'dimension': 2, 'truncate': 'NONE'}
    assert embeddings.embed_query('question') == [.1, .2]
    assert pc.inference.embed.call_args.kwargs['parameters']['input_type'] == 'query'


def test_embedding_dimension_and_count_mismatch_rejected():
    pc = Mock()
    pc.inference.embed.return_value = SimpleNamespace(data=[SimpleNamespace(values=[.1])])
    with pytest.raises(ValueError, match='dimensions'):
        PineconeEmbeddings(pc, dimensions=2).embed_documents(['document'])
    pc.inference.embed.return_value = SimpleNamespace(data=[])
    with pytest.raises(ValueError, match='count'):
        PineconeEmbeddings(pc, dimensions=2).embed_query('question')


def test_claude_and_pinecone_credentials_without_openai(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
    monkeypatch.setenv('PINECONE_API_KEY', 'test')
    credentials()
    monkeypatch.delenv('ANTHROPIC_API_KEY')
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        credentials()
