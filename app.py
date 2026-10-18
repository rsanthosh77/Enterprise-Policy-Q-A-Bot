import json
import time
import streamlit as st
from rag.corpus import load_corpus
from rag.engine import Engine, MANIFEST, Settings, ingest
from rag.evaluate import PRESETS, cases, evaluate

st.set_page_config(page_title='Enterprise Document Q&A', page_icon='📚', layout='wide')
st.title('Enterprise Document Q&A')
st.caption('Answers grounded in HR, compliance, onboarding, and product documents, with source citations.')

with st.sidebar:
    st.header('Documents & retrieval')
    st.caption('Local document preview works without API keys. Configure .env to ingest and ask questions.')
    if st.button('Ingest sample documents', type='primary'):
        try:
            with st.spinner('Embedding documents and preparing a Pinecone snapshot…'):
                manifest = ingest()
            st.success(f'Indexed {len(manifest["chunks"])} chunks')
            st.session_state.pop('result', None)
        except Exception as exc:
            st.error(str(exc))
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        st.caption(f'Active snapshot: {manifest["namespace"]}')
    alpha = st.slider('Dense semantic weight', 0., 1., .5, .05, help='0 = sparse keywords only; 1 = dense only. Hybrid uses weighted reciprocal rank fusion.')
    candidate_k = st.slider('Candidates per search', 12, 60, 30)
    context_k = st.slider('Sections/chunks sent to answer step', 4, candidate_k, min(12, candidate_k))
    rerank = st.checkbox('Rerank candidates', help='Uses Pinecone bge-reranker-v2-m3; adds latency and service usage.')
    expand = st.checkbox('Include complete split sections', value=True)
    filters = st.multiselect('Limit document types', ['hr', 'compliance', 'product', 'onboarding'], help='Leave empty to search all documents.')
    st.caption('Two evidence-checking LLM calls per answer. Retrieval-only mode skips generation.')

ask, documents, evaluations = st.tabs(['Ask & inspect', 'Document explorer', 'Compare retrieval'])
with ask:
    sample = st.selectbox('Example question', [c['question'] for c in cases()])
    with st.form('question-form'):
        question = st.text_area('Your question', value=sample)
        retrieval_only = st.checkbox('Retrieve only')
        submitted = st.form_submit_button('Ask', type='primary')
    if submitted:
        try:
            with st.spinner('Searching document evidence…'):
                start = time.monotonic()
                engine = Engine()
                retrieval = engine.retrieve(question, Settings(alpha, candidate_k, context_k, rerank, expand), filters)
                answer = None if retrieval_only else engine.answer(question, retrieval)
                st.session_state['result'] = {'question': question, 'retrieval': retrieval, 'answer': answer,
                                              'total_seconds': time.monotonic() - start}
        except Exception as exc:
            st.session_state.pop('result', None)
            st.error(str(exc))
    if 'result' in st.session_state:
        result = st.session_state['result']
        retrieval, answer = result['retrieval'], result['answer']
        st.subheader(result['question'])
        if answer:
            if answer['status'] == 'unknown':
                st.info(answer['message'])
            else:
                for claim in answer['claims']:
                    citations = ' '.join(f'[{c["source"]}](#source-{c["source"]})' for c in claim['citations'])
                    st.markdown(claim['text'] + ' ' + citations)
                if answer['conflicts']:
                    st.warning('Conflicting statements found in the retrieved documents.')
                    for claim in answer['conflicts']:
                        st.markdown(claim['text'] + ' ' + ' '.join(f'[{c["source"]}](#source-{c["source"]})' for c in claim['citations']))
        cols = st.columns(3)
        cols[0].metric('Retrieval', f'{retrieval["retrieval_seconds"]:.2f}s')
        cols[1].metric('Total', f'{result["total_seconds"]:.2f}s')
        cols[2].metric('Evidence chunks', len(retrieval['chunks']))
        st.subheader('Source excerpts')
        for i, chunk in enumerate(retrieval['chunks'], 1):
            m = chunk['metadata']
            st.markdown(f'<span id="source-{i}"></span>', unsafe_allow_html=True)
            with st.expander(f'[{i}] {m["filename"]} · §{m["section_number"]} {m["section_title"]} · lines {m["line_start"]}–{m["line_end"]}'):
                st.caption(f'Version {m["version"]} | {m.get("effective_date", m.get("updated_date", "Date unspecified"))}')
                st.code(chunk['text'], language=None)
                if answer:
                    for claim in answer['claims'] + answer['conflicts']:
                        for citation in claim['citations']:
                            if citation['source'] == i:
                                st.text('Cited quote: ' + citation['quote'])
                st.json(m)
        with st.expander('Candidate rankings and retrieval settings'):
            st.json(retrieval['settings'])
            st.dataframe([{'document': c['metadata']['filename'], 'section': c['metadata']['section_number'],
                           'dense': c.get('dense_score'), 'sparse': c.get('sparse_score'),
                           'fusion': c.get('fusion_score'), 'rerank': c.get('rerank_score')} for c in retrieval['candidates']], width='stretch')
        st.download_button('Download question trace', json.dumps(result, indent=2), 'question-trace.json', 'application/json')

with documents:
    chunks = load_corpus()
    doc = st.selectbox('Document', sorted({c['metadata']['filename'] for c in chunks}))
    for c in chunks:
        if c['metadata']['filename'] == doc:
            m = c['metadata']
            with st.expander(f'§{m["section_number"]} {m["section_title"]} — lines {m["line_start"]}–{m["line_end"]}'):
                st.code(c['text'], language=None)
                st.json(m)
    st.download_button('Download chunk & metadata preview', json.dumps(chunks, indent=2), 'chunks.json', 'application/json')

with evaluations:
    st.write('Compare saved questions across dense, sparse, and hybrid settings. Section recall measures whether the expected conditions reach the answer context.')
    selected = st.multiselect('Presets', list(PRESETS), default=['dense', 'sparse', 'hybrid'])
    generate = st.checkbox('Also evaluate answers and “I don’t know” behavior', help='Adds up to two LLM calls per question per preset.')
    st.dataframe(cases(), width='stretch')
    if st.button('Run comparison', disabled=not selected):
        try:
            with st.status('Running saved evaluation…') as status:
                report = evaluate(Engine(), selected, generate, lambda message: status.update(label=message))
                status.update(label='Evaluation complete', state='complete')
                st.session_state['report'] = report
        except Exception as exc:
            st.error(str(exc))
    if 'report' in st.session_state:
        import pandas as pd
        report = st.session_state['report']
        df = pd.DataFrame(report['rows'])
        metrics = ['section_recall', 'all_required_sections', 'candidate_recall', 'recall_at_k', 'precision_at_k', 'mrr', 'ndcg_at_k', 'retrieval_seconds']
        if report['generate']:
            metrics += ['status_matches', 'required_terms_coverage']
        summary = df.groupby('preset')[metrics].mean(numeric_only=False)
        st.dataframe(summary, width='stretch')
        st.bar_chart(summary[['section_recall', 'candidate_recall']])
        st.dataframe(df.drop(columns=['answer'], errors='ignore'), width='stretch')
        st.caption('Required-term coverage is a diagnostic, not a semantic correctness score. Inspect cited answers manually, especially conflicts and missing conditions.')
        st.download_button('Download evaluation report', json.dumps(report, indent=2), 'evaluation-report.json', 'application/json')
