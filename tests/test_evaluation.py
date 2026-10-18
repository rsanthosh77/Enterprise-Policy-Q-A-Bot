from rag.evaluate import ranked_metrics, cases

def chunk(doc, sec):
    return {'metadata':{'document_id':doc,'section_number':sec}}

def test_ranked_metrics_deduplicate_sections_and_fixed_k():
    rows=[chunk('hr','1'),chunk('hr','1'),chunk('hr','2'),chunk('hr','3')]
    result=ranked_metrics([['hr','1'],['hr','3']], rows, k=2)
    assert result['recall_at_k']==.5
    assert result['precision_at_k']==.5
    assert result['mrr']==1
    assert 0 < result['ndcg_at_k'] < 1

def test_unanswerable_metrics_are_not_perfect_scores():
    assert all(v is None for v in ranked_metrics([], [], 12).values())

def test_gold_set_is_15_unique_cases_with_rubrics():
    gold=cases()
    assert len(gold)==len({c['id'] for c in gold})==15
    assert all(c['required_facts'] for c in gold if c['expected_status']=='answered')
