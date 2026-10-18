"""Offline report builder: consumes frozen run JSON, never loads service keys."""
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'docs'/'evaluation_results.json'
PDF=ROOT/'docs'/'Enterprise_RAG_15_Question_Evaluation_Report.pdf'
BLUE=colors.HexColor('#173d63');TEAL=colors.HexColor('#087f8c')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='T',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=BLUE,spaceAfter=14))
styles.add(ParagraphStyle(name='H',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=BLUE,spaceBefore=8,spaceAfter=9,keepWithNext=True))
styles.add(ParagraphStyle(name='S',fontName='Helvetica-Bold',fontSize=10,leading=14,textColor=TEAL,spaceBefore=7,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='B',fontName='Helvetica',fontSize=9,leading=12.5,spaceAfter=7))
styles.add(ParagraphStyle(name='C',fontName='Helvetica',fontSize=7.5,leading=10))
styles.add(ParagraphStyle(name='CH',parent=styles['C'],textColor=colors.white,fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Small',fontName='Helvetica',fontSize=7.8,leading=10,spaceAfter=5))
story=[]
def p(s,style='B'):story.append(Paragraph(s,styles[style]))
def plain(s,style='B'):p(escape(str(s)),style)
def heading(s):plain(s,'H')
def sub(s):plain(s,'S')
def page(s):story.append(PageBreak());heading(s)
def table(headers, rows, widths):
    data=[[Paragraph(escape(str(c)),styles['CH'] if i==0 else styles['C']) for c in row] for i,row in enumerate([headers]+rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4f8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t);story.append(Spacer(1,8))
def fmt(value):return '-' if value is None else f'{value:.3f}'
def avg(rows,key):
    values=[r[key] for r in rows if isinstance(r.get(key),(int,float))]
    return statistics.mean(values) if values else None

def categories(row):
    out=[]
    if row.get('error'):return ['service/runtime error']
    if row.get('judge_error'):out.append('scoring error')
    if row.get('candidate_recall') is not None and row['candidate_recall']<1:out.append('candidate retrieval miss')
    elif row.get('context_recall') is not None and row['context_recall']<1:out.append('ranking/context loss')
    answer=row.get('answer',{})
    if row['expected_status']=='answered' and answer.get('status')=='unknown':out.append('false abstention')
    if row['expected_status']=='unknown' and answer.get('status')=='answered':out.append('unsupported answer on absent/adversarial query')
    judgment=row.get('judgment',{})
    if row.get('case') == 'learning' and row.get('fact_coverage') is not None and row['fact_coverage'] < 1:
        out.append('rubric scope artifact')
    if answer.get('status')=='answered':
        if judgment.get('correct') is False:out.append('answer correctness')
        if judgment.get('citation_support') is False:out.append('citation support')
        if row.get('fact_coverage') is not None and row['fact_coverage']<1 and row.get('case') != 'learning':out.append('incomplete answer')
    return out

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#cfdae5'));c.line(48,39,547,39)
    c.setFillColor(colors.HexColor('#526174'));c.setFont('Helvetica',8)
    c.drawString(48,26,'Enterprise Document Q&A | 15-question evaluation | 18 October 2026')
    c.drawRightString(547,26,str(doc.page))
class Doc(SimpleDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name in ('T','H'):
            key='section-'+str(self.seq.nextf('h'));self.canv.bookmarkPage(key);self.canv.addOutlineEntry(f.getPlainText(),key,level=0)

def main():
    report=json.loads(SOURCE.read_text());gold=report['gold_cases'];rows=report['rows'];identity=report['identity'];presets=list(identity['presets'])
    if len(gold)!=15 or len(rows)!=60 or len({(r['preset'],r['case']) for r in rows})!=60:
        raise ValueError('Report requires all 60 unique trials')
    review_path=ROOT/'docs'/'evaluation_review.json'
    adjustments={a['case']:set(a['excluded_fact_indices']) for a in json.loads(review_path.read_text()).get('rubric_adjustments',[])} if review_path.exists() else {}
    for r in rows:
        facts=[f for f in r.get('judgment',{}).get('facts',[]) if f['index'] not in adjustments.get(r['case'],set())]
        r['question_aligned_fact_coverage']=sum(f['met'] for f in facts)/len(facts) if facts else None
        r['question_aligned_pass']=bool(r.get('status_matches') and r.get('judgment',{}).get('correct') and (r['expected_status']=='unknown' or (r.get('judgment',{}).get('citation_support') and r['question_aligned_fact_coverage']==1)))
    by={(r['preset'],r['case']):r for r in rows};hybrid=[by['hybrid',g['id']] for g in gold]
    failures=[r for r in rows if not r.get('answer_pass')]
    p('15-question evaluation report','T')
    p('Working enterprise Q&A bot: retrieval quality, answer quality and failure analysis','S')
    plain('Four source documents | 15 frozen questions | Four retrieval presets | 60 live trials')
    sub('Deliverable status')
    plain(f"Completed {len(rows)} trials. {sum('error' not in r for r in rows)} completed without a retrieval/generation service error; {sum('judgment' in r for r in rows)} have an answer review. All questions have per-preset retrieval results and answer outcomes below.")
    sub('Primary configuration: hybrid, without reranking')
    plain(f"Mean section Recall@12: {fmt(avg(hybrid,'recall_at_k'))}. All-required-context rate: {fmt(avg(hybrid,'all_required_sections'))}. Strict answer pass: {sum(bool(r.get('answer_pass')) for r in hybrid)}/15. Strict pass requires correct answered/unknown status, no judged factual error, citation support and every rubric fact.")
    plain(f"Question-aligned answer pass after the documented rubric-scope correction: {sum(r['question_aligned_pass'] for r in hybrid)}/15. The frozen strict score remains unchanged; see the review notes.")
    sub('Scope and grading caveat')
    plain('The report evaluates the existing bot without tuning its retrieval or generation behavior during the run. Answer judgments use a separate Claude rubric-grading call, with the same model family as the bot; these are model-assisted scores, not independent human-certified ground truth. The coding assistant additionally inspects primary answers and failures. Exact gold labels and raw traces are included in the accompanying JSON.')
    sub('Files to submit together')
    plain('This PDF; docs/evaluation_results.json (frozen gold, configuration and all 60 traces); docs/evaluation_summary.csv (one row per trial). The working app and reproducible runner remain in the repository.')
    sub('Run identity')
    started=datetime.fromisoformat(report['started_at_utc']).astimezone(ZoneInfo('Asia/Singapore'))
    plain(f"Started: {started.strftime('%d %B %Y %H:%M %Z')}. Snapshot: {identity['namespace']}. Embeddings: {identity['embedding_model']} / {identity['embedding_dimensions']} dimensions. Generation, evidence verification and rubric judge: {identity['chat_model']}.")
    plain('This is a small, developer-authored diagnostic set. Ten cases build on the existing suite; five are new. It is not an independently held-out benchmark, repeated-trial reliability study or production certification. Source dates are 2024-2025; policy claims are evaluated against those documents only.','Small')

    page('1. Questions, gold labels and method')
    table(['# / ID','Category','Expected'],[(f'{i+1}. {g["id"]}',g['category'],g['expected_status']) for i,g in enumerate(gold)],[193,210,84])
    plain('Gold cases and required-fact rubrics were written before live requests. Twelve questions are answerable; two absent-information questions and one instruction-injection question should abstain. The training gold was narrowed before the run to onboarding 3.1 because it directly contains all requested training; onboarding 2.3 corroborates a Day-1 setup detail and is not mandatory for this question.')
    plain('Apparent policy conflict is tested through different scopes (receiving gifts versus providing hospitality), not fabricated contradictory versions. The ambiguous token question should distinguish access and refresh tokens. There is no synthetic true-conflict document in this corpus; a future contradictory-version fixture remains needed.')
    sub('Presets held constant')
    table(['Preset','Dense weight','Candidates/signal','Context K','Rerank'],[(n,str(s['dense_weight']),str(s['candidate_k']),str(s['context_k']),str(s['rerank'])) for n,s in identity['presets'].items()],[135,83,112,77,80])
    plain('All presets enable complete split-section expansion. Ranking deduplicates document/section pairs for the K-based metrics. The current 86-chunk corpus has no split sections at the configured 1,800-character ceiling. The runner uses three worker threads; latency is wall-clock under shared concurrent load, not isolated benchmark latency. No automatic retries of failed entire trials are hidden; SDK-level retries can occur.')

    page('2. What each score means')
    table(['Metric','Definition / interpretation'],[
    ('Recall@12','Required gold sections in the first 12 unique ranked sections / total required sections.'),
    ('Precision@12','Gold-required sections in the first 12 unique ranked sections / 12. Fixed denominator, even if fewer results. Gold is a minimum evidence set; useful extra context may count as non-gold.'),
    ('MRR@12','Reciprocal rank of the first gold-required section within the first 12, or zero if absent.'),
    ('nDCG@12','Binary gold relevance with log2 rank discount / ideal DCG for the number of gold sections.'),
    ('Candidate recall','Gold sections anywhere in the fused candidate union, before final truncation.'),
    ('Context recall / all required','Gold coverage in the actual expanded answer context; all-required is a Boolean.'),
    ('Correctness','Rubric judge sees no factually wrong or invented substantive claim. Completeness scored separately.'),
    ('Fact coverage','Required compound rubric items met / total items; faithful paraphrases accepted. A compound item needs every material part.'),
    ('Citation support','Judge finds supporting cited evidence for every substantive claim. Exact quote/source checks also occur inside the bot.'),
    ('Status match','Returned answered/unknown matches the expected label.'),
    ('Strict answer pass','Status match AND correct AND, for answerable questions, full fact coverage plus citation support.'),
    ('Abstention success','Unknown on each of the three absent/adversarial cases; not included in retrieval relevance averages.')],[135,352])
    plain('Unanswerable cases have null retrieval relevance scores, rather than artificial perfect scores. False abstentions on answerable questions receive fact coverage zero and fail the strict score. Citation-support averages apply only to answered responses with a judge score; they exclude unknowns. Service errors count as strict failures and are reported explicitly, not rewritten as unknown.')
    plain('Low Precision@12 is expected for gold sets containing only one to three required sections and a broad 12-section context. It is a conservative gold-hit density measure, not a complete relevance annotation. The goal here prioritizes finding all conditions; compare recall and complete-context rate alongside precision.')

    page('3. Aggregate retrieval and answer results')
    table(['Preset','R@12','P@12','MRR@12','nDCG@12','All context'],[(n,fmt(avg([r for r in rows if r['preset']==n],'recall_at_k')),fmt(avg([r for r in rows if r['preset']==n],'precision_at_k')),fmt(avg([r for r in rows if r['preset']==n],'mrr')),fmt(avg([r for r in rows if r['preset']==n],'ndcg_at_k')),fmt(avg([r for r in rows if r['preset']==n],'all_required_sections'))) for n in presets],[127,66,66,72,80,76])
    plain('Retrieval averages use only the 12 answerable questions with available metric results. Detailed trials reveal any missing/error measurements; denominators are not silently treated as successful scores.')
    table(['Preset','Status','Strict pass','Mean facts','Citations','Unknowns'],[(n,f"{sum(bool(r.get('status_matches')) for r in rows if r['preset']==n)}/15",f"{sum(bool(r.get('answer_pass')) for r in rows if r['preset']==n)}/15",fmt(avg([r for r in rows if r['preset']==n],'fact_coverage')),fmt(avg([r['judgment'] for r in rows if r['preset']==n and 'judgment' in r],'citation_support')),f"{sum(r.get('answer',{}).get('status')=='unknown' for r in rows if r['preset']==n and r['expected_status']=='unknown')}/3") for n in presets],[127,65,78,75,73,69])
    table(['Preset','Mean retrieval','Mean total incl. judge','Runtime errors','Judge errors'],[(n,fmt(avg([r['retrieval'] for r in rows if r['preset']==n and 'retrieval' in r],'retrieval_seconds'))+'s',fmt(avg([r for r in rows if r['preset']==n],'total_seconds'))+'s',sum('error' in r for r in rows if r['preset']==n),sum('judge_error' in r for r in rows if r['preset']==n)) for n in presets],[127,100,118,76,66])
    plain('Total seconds includes post-answer rubric grading, which is not part of interactive app latency. Provider billing/token usage was not separately instrumented; the timing table is not a cost report. Each nonempty answered trial can use generation, in-app verification, and separate rubric grading calls.')
    table(['Preset','Frozen strict pass','Question-aligned pass'],[(n,f"{sum(bool(r.get('answer_pass')) for r in rows if r['preset']==n)}/15",f"{sum(r['question_aligned_pass'] for r in rows if r['preset']==n)}/15") for n in presets],[167,160,160])
    plain('Question-aligned pass excludes only L&D fact index 1 (allowed purchases), which the question did not ask. It is an explicitly post-run rubric correction by the coding assistant, applied equally to all presets. No original results are rewritten. It does not resolve the potential training-conflict ambiguity noted below.')
    sub('Selection rule')
    plain('Prefer a configuration that preserves required evidence and improves strict answer outcomes on this set, then validate on new held-out questions before changing app defaults. A higher retrieval ranking score alone does not establish better answers. This report makes recommendations, not an automatic production configuration change.')

    page('4. Failure analysis and recommended actions')
    counts=Counter(category for r in rows for category in categories(r))
    table(['Observed failure category','Trials (can overlap)'],list(counts.items()) or [('None detected by the scoring rules',0)],[355,132])
    plain('Failure categories are not mutually exclusive. Candidate retrieval miss means required evidence never entered the union. Ranking/context loss means candidate evidence was present but absent from final context. Incomplete answer means a rubric condition was omitted despite an answered response. False abstention means an answerable question returned unknown. The engine does not expose whether generation or its evidence verifier triggered unknown, so this report cannot honestly split those two causes without further instrumentation.')
    failed_table=[]
    for r in failures:
        j=r.get('judgment',{})
        omissions=[str(f['index']+1) for f in j.get('facts',[]) if not f['met']]
        failed_table.append((r['preset']+' / '+r['case'],', '.join(categories(r)) or 'strict grading failure', ','.join(omissions) or '-'))
    table(['Failed trial','Diagnosis','Missing fact #'],failed_table or [('All strict checks passed','No detected failure','-')],[205,225,57])
    sub('Prioritized improvements')
    plain('1. Correct rubric scope artifacts before interpreting bot defects. Review omitted rubric facts and their retrieved passages before changing embeddings. If evidence is already present, improve the generation/completeness checks rather than increasing candidate count blindly.')
    plain('2. Where candidate recall fails, test query reformulation or larger pools. Where context alone loses evidence, test reranking or document/section coverage selection. Preserve this baseline and rerun on additional held-out questions.')
    plain('3. Return an internal reason code for unknown (empty evidence, model abstention, invalid quote, verifier rejection), and verifier reasons. This would distinguish conservative verification from generation failure without exposing unsupported text to end users.')
    plain('4. Add true contradictory-version fixtures and independent human labels for relevance, factuality and citation entailment. Expand beyond this 15-question dataset before relying on aggregate percentages.')
    review=ROOT/'docs'/'evaluation_review.json'
    if review.exists():
        sub('Additional source inspection by the coding assistant')
        for item in json.loads(review.read_text()).get('observations',[]):plain(item)
    sub('Detailed cases')
    plain('Each following case includes frozen expected evidence, all four configurations, the primary hybrid answer, and specific failure explanations. Full answers, quote citations and retrieved text for all 60 trials are available in evaluation_results.json.')

    for number,g in enumerate(gold,1):
        page(f'Q{number:02d}. {g["id"]}')
        plain(g['question'],'S');plain('Category: '+g['category']+' | Expected status: '+g['expected_status'])
        plain('Gold required sections: '+(', '.join(d+' §'+sec for d,sec in g['expected_sections']) or 'None: abstain; no supported substantive answer.'))
        if g['required_facts']:
            sub('Expected answer requirements')
            for i,fact in enumerate(g['required_facts'],1):plain(f'{i}. {fact}')
        table(['Preset','R@12 / P@12','nDCG','Status','Facts','Frozen','Aligned'],[(n,fmt(by[n,g['id']].get('recall_at_k'))+' / '+fmt(by[n,g['id']].get('precision_at_k')),fmt(by[n,g['id']].get('ndcg_at_k')),by[n,g['id']].get('answer',{}).get('status','ERROR'),fmt(by[n,g['id']].get('fact_coverage')),'PASS' if by[n,g['id']].get('answer_pass') else 'FAIL','PASS' if by[n,g['id']].get('question_aligned_pass') else 'FAIL') for n in presets],[100,100,58,65,51,56,57])
        primary=by['hybrid',g['id']];answer=primary.get('answer',{})
        if g['id'] in adjustments:
            plain('Post-run rubric review: eligible-purchase examples were not asked. The aligned column excludes that criterion uniformly; frozen scores and fact coverage still show the original rubric result.','Small')
        sub('Primary hybrid response')
        if primary.get('error'):plain(primary['error'])
        elif answer.get('status')=='unknown':plain(answer.get('message',"I don't know based on the supplied documents."))
        else:
            for claim in answer.get('claims',[])+answer.get('conflicts',[]):
                cites=[]
                for cite in claim['citations']:
                    c=primary['retrieval']['chunks'][cite['source']-1];m=c['metadata']
                    cites.append(f"{m['filename']} §{m['section_number']} L{m['line_start']}-{m['line_end']}")
                plain(claim['text']+' ['+'; '.join(dict.fromkeys(cites))+']')
        sub('Analysis across configurations')
        for n in presets:
            r=by[n,g['id']];j=r.get('judgment',{})
            if r.get('error') or r.get('judge_error'):
                plain(n+': '+(r.get('error') or r['judge_error']));continue
            if not r.get('answer_pass') or (r.get('context_recall') is not None and r['context_recall']<1):
                plain(n+': '+j.get('analysis','')+' Diagnosis: '+', '.join(categories(r)))
                for f in j.get('facts',[]):
                    if not f['met']:plain(f"Missing fact {f['index']+1}: {f['explanation']}",'Small')
        if all(by[n,g['id']].get('answer_pass') and by[n,g['id']].get('context_recall',1) in (None,1) for n in presets):
            plain('All four configurations pass the strict answer rubric; no required-evidence loss detected. This does not rule out unannotated edge cases.')
        if 'judgment' in primary:
            plain('Hybrid rubric review: '+primary['judgment']['analysis'],'Small')

    page('Appendix. Reproduce and inspect the evidence')
    plain('Run with the project virtual environment so Pinecone SDK 10 and the Claude integration are selected. Configure existing service keys in .env. The scripts never print the keys; this report builder never reads .env.')
    for line in ['.venv/bin/python -m pytest -q','.venv/bin/python scripts/run_deliverable_evaluation.py','/tmp/rag-pdf-venv/bin/python scripts/build_evaluation_report.py']:
        plain(line,'Small')
    plain('The live runner resumes the current report only if snapshot, gold hash, preset settings and model identifiers match. Completed trials, including recorded failures, are not rerun automatically. To make a fresh independent run, archive evaluation_results.json first. The PDF builder needs ReportLab; PyMuPDF was used separately for layout validation, not runtime bot execution.')
    sub('Frozen run details')
    plain('Gold SHA-256: '+identity['gold_hash'],'Small')
    for file,sha in report['source_hashes'].items():plain(file+': '+sha,'Small')
    plain('Completed UTC: '+report.get('completed_at_utc','not marked complete'),'Small')
    sub('Raw result contract')
    plain('identity records model, snapshot, gold hash and preset settings. gold_cases contains questions, categories, expected sections/status and required facts. Each row records ranked candidates, context chunks, raw scores, citations and answer text, metric scores, timing, structured rubric review and explicit service/scoring errors. No API credentials are included.')
    plain('Semantic labels are machine-assisted and sensitive to model variation and rubric wording. Gold section precision treats unannotated relevant passages as non-gold. A strict failure can be an incomplete but otherwise correct answer. The separate review JSON documents the uniform post-run L&D rubric correction; both frozen and question-aligned scores are reported. No automatic retraining, reingestion or answer tuning was performed as part of this evaluation.')
    Doc(str(PDF),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=45,bottomMargin=55,title='Enterprise RAG: 15-question evaluation report',author='Enterprise Document Q&A evaluation').build(story,onFirstPage=footer,onLaterPages=footer)
    fields=['case','preset','expected_status','actual_status','recall_at_k','precision_at_k','mrr','ndcg_at_k','candidate_recall','context_recall','all_required_sections','status_matches','fact_coverage','correct','citation_support','answer_pass','question_aligned_fact_coverage','question_aligned_pass','retrieval_seconds','total_seconds','failure_categories','error','judge_error']
    with (ROOT/'docs'/'evaluation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            out={k:r.get(k) for k in fields};out.update(actual_status=r.get('answer',{}).get('status'),correct=r.get('judgment',{}).get('correct'),citation_support=r.get('judgment',{}).get('citation_support'),retrieval_seconds=r.get('retrieval',{}).get('retrieval_seconds'),failure_categories='; '.join(categories(r)))
            w.writerow(out)
    print(PDF)

if __name__=='__main__': main()
