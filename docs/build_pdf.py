"""Build the project walkthrough PDF. Requires reportlab; never reads .env."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import statistics
import textwrap
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable, Preformatted

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'docs'/'Enterprise_RAG_Architecture_and_Code_Guide.pdf'
BLUE=colors.HexColor('#163b63'); TEAL=colors.HexColor('#087f8c'); GRAY=colors.HexColor('#526174')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=BLUE,spaceAfter=12))
styles.add(ParagraphStyle(name='HCustom',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=BLUE,spaceAfter=10))
styles.add(ParagraphStyle(name='SubCustom',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=9,spaceAfter=5))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=5))
styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=8.5,leading=11))
styles.add(ParagraphStyle(name='CodeCustom',fontName='Courier',fontSize=7,leading=9,spaceAfter=8))
story=[]

def p(text,style='BodyCustom'):
    story.append(Paragraph(text, styles[style]))
def h(text): p(text,'SubCustom')
def code(text):
    lines=[]
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line,width=97,replace_whitespace=False,drop_whitespace=False) or [''])
    story.append(Preformatted('\n'.join(lines),styles['CodeCustom']))
def table(headers, rows, widths=None):
    header_style=ParagraphStyle(name='HeaderCell',parent=styles['CellCustom'],textColor=colors.white,fontName='Helvetica-Bold')
    cells=[[Paragraph(escape(str(x)), header_style if i == 0 else styles['CellCustom']) for x in row] for i,row in enumerate([headers]+rows)]
    t=Table(cells,colWidths=widths or [499/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,BLUE)]))
    # Header paragraphs need explicit white; Paragraph overrides table text color.
    for cell in cells[0]: cell.style=ParagraphStyle(name='HeaderCell',parent=styles['CellCustom'],textColor=colors.white,fontName='Helvetica-Bold')
    story.append(t);story.append(Spacer(1,8))
def page(title):
    story.append(PageBreak());p(title,'HCustom')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#d3dce5'));canvas.line(48,39,547,39)
    canvas.setFont('Helvetica',8);canvas.setFillColor(GRAY)
    canvas.drawString(48,26,'Enterprise Document Q&A | Implementation guide | 14 October 2026')
    canvas.drawRightString(547,26,str(doc.page))

class Architecture(Flowable):
    def __init__(self):
        super().__init__()
        self.width = 499
        self.height = 420
    def draw(self):
        c=self.canv
        def label(text,x,y,size=9,color=BLUE):
            c.setFillColor(color);c.setFont('Helvetica-Bold',size);c.drawString(x,y,text)
        def box(x,y,w,title,lines,fill='#edf4fa',height=56):
            c.setFillColor(colors.HexColor(fill));c.setStrokeColor(colors.HexColor('#b9cedd'));c.roundRect(x,y,w,height,6,stroke=1,fill=1)
            c.setFillColor(BLUE);c.setFont('Helvetica-Bold',9);c.drawString(x+8,y+height-16,title)
            c.setFillColor(GRAY);c.setFont('Helvetica',8)
            for i,line in enumerate(lines): c.drawString(x+8,y+height-30-i*11,line)
        def arrow(x1,y1,x2,y2):
            import math
            c.setStrokeColor(TEAL);c.setFillColor(TEAL);c.setLineWidth(1.3);c.line(x1,y1,x2,y2)
            angle=math.atan2(y2-y1,x2-x1);a=5
            path=c.beginPath();path.moveTo(x2,y2);path.lineTo(x2-a*math.cos(angle-.5),y2-a*math.sin(angle-.5));path.lineTo(x2-a*math.cos(angle+.5),y2-a*math.sin(angle+.5));path.close();c.drawPath(path,fill=1,stroke=0)
        label('INGESTION / Run when documents or embeddings change',0,405)
        box(0,329,146,'Four TXT documents',['HR / compliance','onboarding / product'])
        box(174,329,151,'Parse + section chunks',['corpus.py: source lines','text + metadata + IDs'])
        box(353,329,146,'Dense + sparse vectors',['Pinecone Llama: dense','local BM25: sparse'])
        arrow(146,357,174,357);arrow(325,357,353,357)
        box(353,247,146,'Pinecone index',['dense cosine + sparse','snapshot namespace'])
        box(174,247,151,'Local active.json',['chunks + BM25 encoder','active snapshot pointer'])
        arrow(426,329,426,303);arrow(353,275,325,275)
        label('QUESTION / Runs on each submitted question',0,224)
        box(0,149,146,'Streamlit question',['settings + optional filters','app.py -> Engine'])
        box(174,149,151,'Hybrid retrieval',['query embeddings + BM25','two Pinecone searches'])
        box(353,149,146,'Fuse + select context',['weighted RRF; rerank*','complete split sections'])
        arrow(146,177,174,177);arrow(325,177,353,177)
        arrow(353,247,305,205)
        arrow(249,247,249,205)
        box(353,65,146,'Claude generation',['structured claims + quotes','evidence only'])
        box(174,65,151,'Validate + verify',['quote checks in Python','second Claude check'])
        box(0,65,146,'Answer or abstention',['citations + source excerpts',"or: I don't know"])
        arrow(426,149,426,121);arrow(353,93,325,93);arrow(174,93,146,93)
        label('* Reranking is optional. Evaluations exercise the same Engine.',0,35,8,GRAY)
        label('Hosted: Pinecone embeddings/storage/reranking; Claude generation/verification.',0,17,8,GRAY)
        label('Local: parsing, BM25, fusion, metadata snapshots, UI, tests and reports.',0,1,8,GRAY)

p('Solution architecture','TitleCustom')
p('RAG-powered enterprise Q&A: architecture, code, decisions and measured results','SubCustom')
story.append(Architecture());story.append(Spacer(1,12))
p('The current implementation uses <b>Claude API authentication</b> for answering and verification, and <b>Pinecone API authentication</b> for dense embeddings, vector search and optional reranking. BM25 keyword vectors are generated locally. This diagram describes the code actually in this project, rather than a proposed future architecture.')
p('Prepared from the current source and saved local evaluation artifacts. Application code and stored keys are unchanged by this documentation task. Credentials are excluded from this report.','SmallCustom')

page('1. What we built, and why')
p('You requested a guided RAG starter over enterprise HR policies, compliance manuals, product documentation and onboarding guides. RAG means retrieval-augmented generation: first find supporting passages, then ask a language model to answer from those passages. The model is not trained on these documents during ingestion.')
table(['Requirement agreed together','Implementation'],[
('Find conditions across documents','Dense + sparse retrieval; broad candidate pool; section expansion; gold section recall'),
('No department/role restrictions','All four documents searchable; document-type filters are optional scope controls'),
('Citations and conflicting evidence','Structured claims with exact source quotes; conflicts kept separately'),
('Missing information means abstention',"Document-only prompts, quote validation and evidence verification; unknown response"),
('Stress-test and compare settings','Streamlit controls, saved evaluation cases, downloadable traces and reports'),
('Claude rather than OpenAI','ChatAnthropic + Pinecone-hosted dense embeddings; only two service keys')],[180,319])
h('Why this stack')
p('Python keeps parsing, embeddings, retrieval and evaluation in one language. Streamlit makes those internals reviewable through a small local UI. LangChain provides the Claude integration and structured-output handling; orchestration is explicit Python, so you can inspect and replace every stage. Pinecone supplies managed storage and inference. This is a small guided starter, not a large LangChain chain or agent framework.')
h('How the design evolved')
p('The initial working starter used OpenAI embeddings and answers. At your request, it was migrated to Claude for both generation and verification, with Pinecone-hosted Llama embeddings replacing OpenAI embeddings. Dependencies, provider tests, environment placeholders, documentation and live ingestion were updated together. Workload Identity Federation was discussed only; the implemented authentication uses API keys.')
h('Reading order')
p('Sections 2-5 explain data and chunking; 6-10 explain embedding, storage, retrieval and answers; 11-14 explain the interface, evaluation, tests and operations; 15-16 explain tuning and limits. The appendix contains line-numbered source listings so you can connect the explanations to each piece of code.')

page('2. Repository map and the actual documents')
table(['File / artifact','Responsibility'],[
('rag/corpus.py','Read TXT files, extract metadata, chunk by headings, generate IDs, save JSON atomically'),
('rag/embeddings.py','Pinecone-hosted dense passage/query embeddings; batching and dimension validation'),
('rag/sparse.py','Local corpus-fitted BM25 sparse vectors'),
('rag/engine.py','Settings, fusion, ingestion, snapshot loading, retrieval, Claude answers and checks'),
('rag/evaluate.py + evaluation.json','Reusable comparison presets, ten gold cases and report metrics'),
('rag/cli.py','preview / ingest / evaluate command-line entry points'),
('app.py','Streamlit question, document and evaluation views'),
('tests/','Nine RAG tests, three embedding/provider tests and an opt-in live check'),
('.env.example / requirements files','Safe configuration template and reproducible dependencies'),
('data/active.json','Active namespace, source chunks, encoder and embedding configuration'),
('data/evaluation-*.json','Recorded retrieval results with settings and namespace'),
('data/claude-live-smoke.json','Three saved live answer checks with evidence traces')],[190,309])
table(['Source','Version / date','Current chunks'],[
('hr_policy.txt','v4.2; effective 1 January 2025','33'),
('compliance_manual.txt','v2.1; effective 15 March 2025','27'),
('onboarding_guide.txt','v2025.1; no explicit effective date','16'),
('tech_docs.txt','v3.1; updated March 2024','10')],[190,239,70])
p('These are four UTF-8 text files, not an OCR/PDF ingestion pipeline. The technical manual uses integer headings such as “2. RATE LIMITING”; the policy and onboarding manuals use decimal subsection headings such as “3.1 Eligibility”. The parser deliberately handles those two observed formats.')

page('3. corpus.py: extraction and provenance')
h('ROOT, TYPES and digest(value)')
p('ROOT anchors paths at the repository rather than the caller’s working directory. TYPES maps the four known filename stems to hr, compliance, product and onboarding. Unrecognized names receive “other”. digest() computes SHA-256 for source fingerprints and deterministic identities.')
h('load_corpus(folder, max_chars=1800, overlap_lines=2)')
p('The function validates settings, reads sorted *.txt paths, keeps the original lines, and inspects the first four lines for the title, version, owner/reviewer and source dates. The onboarding header is handled explicitly as People Operations. Unsupported or missing dates and owners are omitted rather than invented. A missing version becomes “unknown”.')
p('Date parsing preserves precision: an exact day becomes YYYY-MM-DD, while “March 2024” becomes YYYY-MM. The code does not make an unstated effective date from a version label such as 2025.1.')
h('Heading boundaries')
code("pattern = r'^(\\d+\\.)\\s+(.+)$' if path.stem == 'tech_docs' \\\n    else r'^(\\d+\\.\\d+)\\s+(.+)$'")
p('The parser records heading indices, adds the beginning and end of the file, then walks adjacent boundaries. Text preceding the first subsection is the “preamble”. This avoids treating onboarding’s ordinary numbered list items as subsection headings. It is a filename-aware regular-expression parser, not a general semantic document parser.')
h('The chunk loop')
p('Within each subsection, the loop accumulates complete lines until the character target is reached. A long first line is allowed to exceed the target so progress is guaranteed. If more text remains, the next chunk starts up to two lines before the previous end. max(cursor + 1, ...) prevents overlap from causing an infinite loop. Whitespace at the outer edges is stripped; tests compare each chunk with the equivalent stripped source-line slice.')
h('Identity and saving')
p('A chunk ID uses the document stem, complete-file content hash, and start/end line offsets, then truncates the SHA-256 hexadecimal digest to 32 characters. IDs are stable for unchanged source and boundaries; editing any part of a document changes the IDs of all its chunks. save_json() writes a sibling .tmp file and replaces the target, reducing the risk of a partially written active manifest. It is intended for one writer at a time.')

page('4. How the chunk size was decided')
p('<b>The 1,800-character target was a practical starter choice, not an empirically optimized value.</b> The observed documents have useful numbered subsections, so the primary choice was to preserve those subsection boundaries. The character ceiling is a fallback for longer subsections.')
h('Reasoning behind the starting point')
p('A chunk should be small enough for precise retrieval and citations, but large enough to keep an eligibility rule with its approvals, deadlines or exceptions. Around 1,800 characters gives a modest passage size for these English text manuals. Character counting is easy to inspect and preserves original line references. It is not a token budget: model token counts depend on text and tokenizer.')
p('The two-line overlap is a low-complexity safeguard when a subsection is split. It is not a measured optimum either. Sections are never merged merely to fill the character target; small subsections remain small, and their title is included in embedding input.')
h('What the current corpus actually produces')
active=json.loads((ROOT/'data'/'active.json').read_text())
chunks=active['chunks'];lengths=[len(c['text']) for c in chunks]
table(['Observed measure','Value'],[
('Total chunks',str(len(chunks))),('Shortest / longest',f'{min(lengths)} / {max(lengths)} characters'),
('Median / mean',f'{statistics.median(lengths):.1f} / {statistics.mean(lengths):.0f} characters'),
('Subsections split at 1,800 characters','0'),
('900-character local parsing experiment','89 chunks; 3 subsections split'),
('3,000-character local parsing experiment','86 chunks; 0 subsections split')],[300,199])
p('The 900/1,800/3,000 comparison above is a <b>local chunk-count experiment performed for this guide</b>. It did not reembed, reindex or compare answer quality. At the current default, section boundaries determine every chunk; the overlap and complete-section expansion behavior are implemented but not exercised by split sections in this corpus.')
h('How to decide a better value')
p('Hold the question set, embeddings and retrieval settings constant. Try several limits, reingest each variant into an isolated snapshot, and compare all-required-section recall, answer completeness, irrelevant context, latency and usage. Use held-out questions. For larger or more varied documents, introduce tokenizer-aware limits and paragraph/code-block boundaries. Do not claim a larger chunk is better merely because it retrieves more text.')

page('5. Metadata: evidence beyond the chunk text')
p('Each Pinecone record has explicit dense and sparse ranking fields, the original chunk text, and flat metadata. Local manifest chunks store the same metadata under a nested “metadata” key. Metadata helps filtering, identity and citation navigation; it does not itself establish that an answer is supported.')
table(['Field','Meaning and use'],[
('document_id','Filename stem; stable document identity'),('chunk_id','Content/location-derived record key; links matches to local source'),
('filename / source_path / title','Readable source label, local source location and document title'),
('document_type','Explicit file mapping; filter both search signals'),('version','Version printed in the header; source context, not automatic version ordering'),
('effective_date / updated_date','Dates explicitly stated in source; precision preserved'),('owner','Stated owner/reviewer; missing values omitted'),
('section_number / section_title','Subsection citation and lookup for complete-section expansion'),
('line_start / line_end','One-based inclusive locations in original TXT'),('content_hash','Complete-file SHA-256; change detection'),
('ingested_at','UTC extraction/ingestion timestamp; not the policy effective date')],[175,324])
h('Example record shape: values shortened for teaching')
code('''{
  "_id": "<deterministic chunk id>",
  "dense": [0.12, "... 1024 float values ..."],
  "sparse": {"indices": [17, 42], "values": [0.8, 1.1]},
  "text": "3.1 Eligibility ...",
  "document_id": "hr_policy", "document_type": "hr",
  "section_number": "3.1", "version": "4.2",
  "line_start": 56, "line_end": "<inclusive source boundary>",
  "content_hash": "<file SHA-256>", "ingested_at": "<UTC timestamp>"
}''')
p('No access-group or geographic tags are inferred, matching your choice to omit role-based restrictions. Source dates are retained for inspection; there is no automatic “current policy only” filter. HR/compliance authority is specified in the answering prompt, not enforced by a metadata ranking rule.')

page('6. embeddings.py: dense semantic representation')
p('A dense embedding is a fixed-length numeric representation of text. It lets a question such as “what approvals do I need to work overseas?” find a section titled “International Remote Work” even when the phrasing differs.')
h('PineconeEmbeddings.__init__')
p('The wrapper holds the Pinecone client, configured model and dimensionality. Defaults are llama-text-embed-v2 and 1,024 dimensions. The dimensionality is the vector length, not a chunk length. It must agree with the Pinecone dense field schema.')
h('_embed(texts, input_type)')
p('The method slices texts into batches of 64 and calls pc.inference.embed(). It passes input_type, dimension and truncate=NONE. Disabling truncation makes oversized inputs fail visibly instead of silently discarding conditions. Each returned vector is converted to a Python list; both the vector count and dimensions must match expectations before any snapshot can be activated.')
code('''parameters = {
    "input_type": input_type,
    "dimension": self.dimensions,
    "truncate": "NONE"
}''')
h('embed_documents() and embed_query()')
p('Documents use the model’s “passage” mode; a question uses “query” mode. Retrieval models can treat these inputs differently, so using the intended mode is part of correctness. embed_documents() returns one vector per supplied chunk; embed_query() returns the first vector from a one-question batch.')
h('embedding_text() in engine.py')
p('Both dense and sparse encoding receive “document title / section number section title”, followed by the chunk text. A generic heading such as “Eligibility” becomes identifiable as HR remote-work eligibility through its surrounding source labels. The original text remains separately available for inspection and citations.')
h('Why hosted Pinecone embeddings')
p('This removed the need for an OpenAI key while preserving managed dense inference and the same storage service. It avoids installing and serving a local neural model in this starter. It does not mean embeddings are generated by Claude. The actual batch size and embedding settings are configurable code choices, not evidence that this model is best for every enterprise dataset.')
h('Limits')
p('There is no embedding cache or explicit pinned model-revision field. The namespace signature includes the model identifier and dimension, but future customizations should also include provider, revision, prompts, preprocessing and normalization. A changed embedding space always requires reembedding stored documents.')

page('7. sparse.py: BM25 keyword representation')
p('Sparse retrieval complements meaning-based retrieval with lexical matching: error codes, policy acronyms, header names and product identifiers. A sparse vector contains only nonzero term dimensions rather than thousands of dense coordinates.')
h('tokens(text)')
p('A regular expression lowercases text and extracts alphanumeric terms while keeping internal hyphens or underscores. “Retry-After” and “X-ACME-Signature” survive as single lexical terms. “401(k)” becomes “401” and “k”. There is no stemming, synonym expansion or stopword removal in this starter.')
h('BM25.__init__(texts)')
p('The constructor tokenizes every title-plus-chunk input, builds a sorted vocabulary mapping terms to numeric indices, calculates average document length, and counts how many chunks contain each term. Sorting makes the term-index assignment deterministic. The corpus and encoder must be kept together: a term index from one vocabulary has no reliable meaning in another.')
code('''IDF(term) = log(1 + (N - document_frequency + 0.5)
                        / (document_frequency + 0.5))''')
h('document(text) and query(text)')
p('The document vector stores saturated term frequency with length normalization. Constants correspond to k1=1.2 and b=0.75, conventional starting values that were not tuned against this dataset. The query vector stores IDF once per unique known query term. Their sparse dot product gives this implementation’s BM25-style score.')
code('''document_weight = f * (k1 + 1) / (
    f + k1 * (1 - b + b * document_length / average_length)
)
score = sum(document_weight(term) * IDF(term)
            for term in unique_known_query_terms)''')
p('Frequently repeated words eventually add little extra score because term frequency saturates. Length normalization prevents long passages from winning solely by having more words. Rare terms receive higher query weights. Unknown vocabulary terms are omitted; a query with no known terms skips sparse search.')
h('Persisted state')
p('vocabulary, avg_length and idf are serialized into active.json. Engine reconstructs the existing encoder using __new__ and restores that state rather than refitting at query time. If documents change, ingestion fits a new encoder and creates a new namespace. Sparse storage remains in Pinecone; encoding and fusion remain local.')

page('8. engine.py: ingestion and snapshot lifecycle')
h('Settings, credentials() and clients()')
p('Settings is an immutable dataclass for dense weight, candidate count, context count, reranking and section expansion. credentials() checks for PINECONE_API_KEY and ANTHROPIC_API_KEY; missing settings fail with a clear message. clients() constructs Pinecone and the dense embedding wrapper. It currently requires both keys even for ingestion or retrieval-only operations, although those operations themselves do not call Claude.')
h('ingest(): step by step')
p('1. Load source chunks and configuration. 2. Declare a schema with dense/cosine and sparse fields. 3. Create the named index if it is absent. 4. Check the existing dense field’s dimensions and metric and the presence of both fields. 5. Compute a deterministic namespace. 6. Fit BM25 on embedding_text(). 7. Embed and upsert batches of 64. 8. Poll until all expected IDs can be fetched. 9. Atomically activate the local manifest.')
code('''signature = SHA256({
  "chunks": [(chunk_id, text), ...],
  "model": embedding_model,
  "dimension": vector_dimensions,
  "encoder": "bm25-v1"
})[:24]
namespace = "corpus-" + signature''')
p('The signature excludes the ingestion timestamp, so a re-run with identical source/configuration uses the same namespace and IDs. Changed source text, dimensions or model produces a new namespace. These are snapshots, not a transactional document database or automatic latest-version resolver.')
h('Visibility check and activation')
p('Fetch polling runs in batches with a 120-second overall deadline and a two-second interval. The new active.json is written only after all chunk IDs are visible. If ingestion raises an error earlier, the prior manifest remains active. Pinecone fetch visibility does not prove that search has already converged; a short delay can still be necessary before querying.')
h('Engine.__init__')
p('Loading requires an active manifest. The constructor checks current embedding model, dimension and index name against the recorded snapshot, initializes the Pinecone index, restores BM25, and creates by_id for constant-time chunk lookup. A configuration mismatch tells you to reingest. Local chunks, rather than rereading mutable source files, provide answer evidence for the chosen snapshot.')
h('Storage and operational implications')
p('Search only returns matched IDs and scores; source text is resolved from the local manifest. This keeps the evidence paired with the encoder and namespace, but means active.json is required on the querying machine. Old namespaces are retained for rollback and may keep consuming storage. There is no automatic garbage collection, multi-writer locking, embedding cache or ingestion scheduler.')

page('9. engine.py: retrieval, fusion and reranking')
h('Engine.retrieve()')
p('The method rejects empty questions and invalid settings, starts a monotonic timer, and creates the same document_type filter for both signals. Dense search uses a query embedding against “dense”; sparse search uses the BM25 query vector against “sparse”. Each asks Pinecone for up to candidate_k matches within the active namespace. Match IDs resolve to local snapshot chunks; unrecognized IDs are ignored.')
p('The two database searches use a ThreadPoolExecutor. One subtle timing detail: dense query embedding is calculated before its search is submitted, so sparse search does not start until that embedding call returns. The implementation does not fully overlap embedding generation with sparse search.')
h('Why weighted reciprocal rank fusion')
p('Dense cosine similarities and sparse BM25 scores have different scales. fuse() uses rank positions rather than adding those raw scores. A chunk present in both ranked lists receives both contributions; chunks present in only one still enter the merged pool. Sorting ties by ID makes the output deterministic.')
code('''fusion_score = alpha / (60 + dense_rank)
             + (1 - alpha) / (60 + sparse_rank)
# Missing rank contributes zero; ranks start at 1.''')
p('alpha=0.5 balances the two lists. alpha=1 runs dense only; alpha=0 runs sparse only. The constant 60 smooths how aggressively top ranks dominate; it is a conventional RRF starting value, not a dataset-tuned parameter. Stored dense_score/sparse_score are diagnostics, not calibrated probabilities.')
h('Optional reranking')
p('When enabled, Pinecone bge-reranker-v2-m3 scores the question against title-plus-text candidates and reorders the whole fused union. Returned indices map back to original candidates; rerank_score is saved. Reranking changes ordering but cannot recover evidence missing from the union. In this code alpha primarily affects fusion order; with reranking enabled, the reranker establishes the final order.')
h('Context selection and expansion')
p('The first context_k ranked chunks are selected. With expand_sections enabled, remaining chunks from the same document/subsection are appended from the snapshot. Expanded chunks are marked and have no retrieval scores. The default 12 is a chunk count before expansion, not a guarantee of 12 unique sections or a maximum token count. The usual union has at most 60 candidates with 30 per signal, before deduplication.')
p('Defaults prioritize recall because you said finding all relevant conditions matters most. Neither 30 candidates nor 12 context chunks was optimized exhaustively. There is no automatic multi-query expansion, related-section traversal, document diversity quota or relevance-score cutoff.')

page('10. engine.py: Claude, citations and abstention')
h('Schemas')
table(['Pydantic type','Contract'],[
('Citation','source: numbered evidence ID; quote: exact supporting text'),
('Claim','text plus a list of citations'),('Answer','status limited to answered/unknown; lists of claims and conflicts'),
('Verification','supported: Boolean from the second evidence-check call')],[150,349])
h('Engine.answer(): ordered gates')
p('If no chunks were retrieved, answer() immediately returns unknown without a model call. Otherwise it assembles numbered evidence blocks, instantiates ChatAnthropic, and asks for native JSON-schema output. Defaults are the configured Claude model, max_tokens=8192, timeout=90 seconds and max_retries=2. No temperature is explicitly set.')
p('The generation prompt restricts answers to the supplied evidence, requests conditions and exceptions, forbids guessing about personal eligibility, and treats questions and documents as untrusted data. It gives formal HR/compliance manuals priority over onboarding summaries and requests actual conflicts with support from both sides.')
p('After parsing, status unknown or an empty claim list returns the standard unknown response. Every remaining claim and conflict must have citations; each source must fall within the evidence list; each quote must be nonblank and appear verbatim in that evidence block. This checks quote existence, not semantic entailment.')
p('A second Claude call checks support, whether citations support their claims, whether required conditions were omitted, and whether the question can be answered. A false result returns unknown. Passing responses are rendered as claims and citations in the UI. Up to two calls are used: zero for empty evidence, one for initial unknown/failed quote checks, two for verification.')
h('Why both checks')
p('A genuine quote can still be used to justify an unrelated or overbroad claim. Python provides exact mechanical checks; the second call evaluates meaning and completeness. The verifier is the same configured model, not an independent provider or a mathematical proof. These gates reduce unsupported output but cannot guarantee factuality or perfect abstention.')
h('Important limitations')
p('Conflicts are not programmatically required to contain two distinct source IDs; the prompt and verifier enforce that intention. Citations can quote source labels added by embedding_text(), although meaningful claims should cite actual passage text. API, refusal or schema-parsing errors propagate to the caller and are shown as errors; they are not silently converted into unsupported answers.')
p('Source documents from 2024-2025 are not independently checked for current legal or policy validity. There is no web retrieval in the runtime answer path, and no automatic legal or geographic inference rule.')

page('11. app.py: the stress-testing interface')
h('Sidebar')
p('The ingestion button calls ingest() and clears the previous answer trace on success. The active namespace is displayed for reproducibility. Sliders set dense weight, candidates per search and context chunks. Checkboxes control reranking and complete-section expansion. Optional document-type filters restrict both signals; an empty selection searches all types.')
h('Ask & inspect tab')
p('A dropdown supplies saved sample questions. A form accepts edited text and a retrieve-only option. On submission the app creates an Engine, retrieves evidence, optionally generates an answer and records the result in session_state. Errors clear the stale question result before displaying a message.')
p('Answered claims become paragraphs with numbered links to source anchors. Unknown results show the standard abstention message. Conflicts have a separate warning and paragraphs. Retrieval/total timing and evidence counts are displayed. Each excerpt expands to source text, version/date, exact cited quotes and metadata. Candidate tables retain dense, sparse, fusion and optional reranking scores. A JSON download preserves the full trace.')
h('Document explorer tab')
p('The explorer calls load_corpus() directly, groups current local chunks by filename, and displays subsection text and metadata. It works without API access. Its important distinction is that it previews current files; the answer path uses active.json. Editing a source file without reingestion can therefore make the preview differ from the active evidence snapshot.')
h('Compare retrieval tab')
p('Preset selection uses rag.evaluate.PRESETS. You can add answer evaluation, which increases model calls. st.status() receives progress callbacks. Saved results enter session_state, then pandas calculates mean metrics by preset. Tables, a recall bar chart and downloadable JSON make comparisons inspectable.')
h('Why Streamlit and session_state')
p('Streamlit offers a quick local interface for experimentation. Its script reruns as widgets change; session_state retains the latest answer and report between reruns. This is not a multi-user backend, persistent chat-history store or deployed enterprise permission model. Updating sidebar settings does not automatically recompute the last answer: submit again.')
h('How to use it productively')
p('First inspect a source and its metadata. Ask the same question using dense-only, sparse-only and hybrid. Examine candidate recall before blaming generation. Then try reranking and confirm that every required condition still reaches the final context. Export traces before changing settings so you can compare the same snapshot.')

page('12. evaluate.py: a repeatable comparison')
h('PRESETS and cases()')
p('PRESETS defines dense (alpha=1), sparse (alpha=0), hybrid (default settings) and hybrid-reranked (default settings with reranking enabled). Each uses the same candidate/context defaults. cases() reads evaluation.json each time, so you can extend the question set without changing the runner.')
h('The gold-case contract')
p('Each case has an ID, question, expected document/subsection pairs, expected status, and optional required text fragments. Seven cases are answerable; two ask for absent information, and one asks the system to invent an official policy. The last three have no expected source sections because they test abstention, not relevant-section recall.')
h('section_metrics()')
code('''expected = set((document_id, section_number), ...)
actual = set((document_id, section_number) for retrieved chunks)
section_recall = len(expected & actual) / len(expected)
all_required_sections = expected <= actual
# No expected sections -> (None, None), not an artificial perfect score.''')
p('The unit of relevance is a section, so multiple chunks from one subsection do not inflate recall. For international remote work, finding HR 3.1 but missing HR 3.5 and onboarding 5.2 gives recall 1/3 and all_required_sections=false. This reflects your preference for all conditions, rather than a single vaguely related hit.')
h('evaluate()')
p('For each selected preset and case, the runner records retrieval seconds, final-context recall, candidate-pool recall, complete-section success and the actual retrieved section IDs. If generate=true, it calls answer(), compares answered/unknown status with the gold label and calculates literal required-term coverage from claim/conflict text. Total seconds includes everything done for that case.')
p('A report contains namespace, presets/settings, whether generation ran and all per-case rows. It is atomically saved under data/evaluation-&lt;UTC timestamp&gt;.json. The callback delivers progress to the CLI or UI. Cases and presets run sequentially; each retrieval can issue two database searches.')
h('What the metrics do not prove')
p('Literal term coverage can miss correct paraphrases or accept misleading text. Status matching does not establish that an answered claim is correct. Expected sections are manually chosen and may not enumerate every supporting section. The runner does not compute precision, nDCG, costs or an independent semantic correctness score. Review full evidence and expand the gold set as the corpus grows.')

page('13. Tests and measured results')
table(['Test','What failure it catches'],[
('Source coverage / stable IDs','Lost nonblank lines, mismatched excerpts, unstable identities, wrong onboarding boundaries'),
('Gold sections exist','Broken evaluation labels after parser/source changes'),
('Sparse matching + persistence','HTTP-code lexical miss, unknown vocabulary behavior, encoder-state drift'),
('Fusion union / endpoints','Dropped one-signal hits, wrong dense-only or sparse-only behavior'),
('Pinecone search wire shapes','Invalid dense or sparse query fields for the installed SDK'),
('Section recall completeness','Metric mistakes that hide missing conditions'),
('Fabricated quote rejection','Citations quoting nonexistent evidence'),
('Unsupported-inference rejection','A valid quote being used for an unjustified claim'),
('Filters + split-section expansion','Inconsistent filters and failure to include a selected section’s other chunks'),
('Embedding modes + batching','Wrong passage/query mode or broken batch boundaries'),
('Embedding dimensions + counts','Wrong vector length or missing embedding results'),
('Two-key provider contract','Accidental continued dependency on an OpenAI key')],[190,309])
p('The first nine tests are in test_rag.py; the last three are in test_embeddings.py. Neural outputs and database results are mocked for deterministic unit tests. Split-section behavior is tested using a deliberately smaller chunk limit. Separate Streamlit AppTest checks were run for startup, missing configuration and citation rendering; these are not persisted as dedicated test files.')
report=json.loads((ROOT/'data'/'evaluation-20261014-094303.json').read_text())
rows=[]
for preset in ['dense','sparse','hybrid']:
    valid=[r for r in report['rows'] if r['preset']==preset and r['section_recall'] is not None]
    times=[r['retrieval_seconds'] for r in report['rows'] if r['preset']==preset]
    rows.append((preset,f'{statistics.mean(r["section_recall"] for r in valid):.3f}',f'{statistics.mean(r["all_required_sections"] for r in valid):.3f}',f'{statistics.mean(times):.3f}s'))
table(['Live preset','Mean section recall','All sections rate','Mean retrieval'],rows,[95,140,135,129])
p('Source: data/evaluation-20261014-094303.json. Recall averages use seven answerable cases; latency uses all ten cases. This run compared retrieval only and did not enable reranking or evaluate all generated answers. These are measurements from one small corpus/run, not a general accuracy or speed guarantee.')
p('The opt-in live_smoke.py also checked three actual Claude answers: HTTP 429 and international remote work returned answered with validated citations; the absent Singapore per diem rate returned unknown. Trace source: data/claude-live-smoke.json. Twelve automated tests passed during the migration. No new hosted calls were needed to create this report.')

page('14. CLI, configuration and dependencies')
h('cli.py main()')
p('argparse limits action to preview, ingest or evaluate. --generate enables answer/status checks, and --presets selects named comparison settings. preview writes data/preview.json without service calls. ingest uploads and prints the active namespace/chunk count. evaluate runs the same evaluator as the UI and prints its report. ValueError and TimeoutError become concise CLI exit messages; other failures may still include a traceback.')
code('''source .venv/bin/activate
python -m rag.cli preview
python -m rag.cli ingest
python -m rag.cli evaluate --presets dense sparse hybrid
python -m rag.cli evaluate --generate --presets hybrid
streamlit run app.py
python -m pytest -q
# Optional live integration check; consumes provider usage:
python tests/live_smoke.py''')
h('Environment settings: values shown are safe defaults, not your keys')
table(['Name','Role / default'],[
('ANTHROPIC_API_KEY','Claude authentication; no value shown'),('PINECONE_API_KEY','Pinecone authentication; no value shown'),
('PINECONE_INDEX','enterprise-rag; choose a compatible index name'),('PINECONE_CLOUD / REGION','aws / us-east-1'),
('EMBEDDING_MODEL / DIMENSIONS','llama-text-embed-v2 / 1024'),('CHAT_MODEL','claude-sonnet-5-5; native structured output required')],[220,279])
p('load_dotenv() reads the project .env when engine.py is imported. Existing shell variables normally take precedence. Restart the Streamlit process after changing .env to avoid retained environment values. Changing embedding model, dimension or index requires reingestion before querying; changing the answer model does not require new vectors.')
h('Supporting files')
p('.env.example is a placeholder-only template. .env was created locally with restrictive file permissions and is excluded by .gitignore; its real contents are never included in this guide. .gitignore also excludes generated data, the virtual environment and Python caches. requirements.txt uses compatible major-version ranges; requirements.lock records the installed exact versions. rag/__init__.py is empty and marks the package. README.md is the operating guide.')
p('The runtime dependencies are Pinecone, langchain-anthropic, Streamlit and python-dotenv, plus pytest for tests. PDF tooling for this report was installed into a separate temporary environment, not added to the app’s dependency lock. .venv/pyvenv.cfg describes the virtual environment; it is not the place for service credentials or retrieval settings.')

page('15. Worked example: conditions across documents')
p('<b>Question:</b> “I joined 95 days ago. Can I work remotely from Singapore?”')
h('What should retrieval bring into context?')
table(['Evidence','Why it matters'],[
('HR 3.1 Eligibility','90-day tenure plus full-time status, review rating and remote-eligible role'),
('HR 3.5 International Remote Work','VP and Chief People Officer written approval, 30-day notice, HR assessment and duration limit'),
('Onboarding 5.2 After Day 90','Summary says eligibility to apply, explicitly subject to HR policy')],[190,309])
p('Dense retrieval helps connect “Singapore” and “work remotely” to international remote work. Sparse retrieval helps match “remote”, “days” and exact identifiers. Fusion joins their candidates; optional reranking can improve ordering. A response saying only “Yes, you have worked more than 90 days” would omit required conditions.')
h('The saved live answer')
p('The actual answer identified the 90-day tenure condition while keeping full-time status, the review rating and role eligibility conditional. It included written approvals, 30-day advance notice and the 90-consecutive-day international limit. It cited the onboarding summary and additional relevant work-schedule/travel provisions. This is a conditional policy answer, not an assertion that the person is approved.')
h('Qualification versus contradiction')
p('The onboarding guide does not contradict HR merely by summarizing less detail; it explicitly defers to the policy. A real contradiction would require incompatible claims about the same scope, such as two different approval thresholds. The starter has conflict prompts and verifier checks, but no deliberately contradictory source fixture was included in the saved live validation.')
h('A missing-answer example')
p('“What is the exact travel per diem rate for Singapore?” cannot be answered from a reference to a separate Travel Policy. The referenced policy itself is absent. Retrieval may still find related passages; the generation and verification gates must return unknown rather than invent a rate. The live smoke test observed that abstention.')
h('What citations mean')
p('Each displayed number refers to one numbered evidence chunk in that specific response. The UI shows the file, subsection, version/date and source lines. These numbers are not stable document IDs across questions; chunk_id and namespace provide the durable trace identity.')

page('16. Tuning, custom models and remaining limits')
table(['Choice','Why selected / when to revisit'],[
('Section-first chunking','Observed headings give precise citations; generalize parser for new document formats'),
('1,800 chars; 2 overlap lines','Starting safeguards; current source has no split sections; tune on larger held-out cases'),
('Dense + sparse; alpha 0.5','Combine paraphrases and exact terms without raw score-scale mixing'),
('30 candidates; 12 context chunks','Recall-oriented baseline; tune missing-section recall, noise, usage and latency'),
('Optional reranking','Adds a relevance pass and service cost; benchmark before making it mandatory'),
('Two answer checks','Mechanical quotes plus semantic verification; latency/cost tradeoff and imperfect model judgment'),
('Snapshot namespaces','Prevent obsolete records competing with updated corpus; explicit cleanup still needed')],[190,309])
h('How custom embeddings fit')
p('Replace the implementation behind embed_documents() and embed_query(), update clients(), then reembed all chunks. Query and document encodings must belong to the same trained space. A changed vector width needs a compatible dense schema/index. Same-width model changes still require a new snapshot. Include provider/revision/prompts/normalization in the snapshot identity. The sparse BM25 path can remain unchanged unless you replace it too.')
h('How custom reranking fits')
p('Replace the inference.rerank() block in retrieve() with a local or hosted query-passage scorer and preserve the candidate-ID mapping. Reranking does not require reembedding the corpus. Measure whether all required sections still survive final context selection. Fine-tuning requires labeled queries, relevant passages and difficult negatives; four manuals and ten gold cases do not establish reliable training gains.')
h('Limits to address before broader deployment')
p('There is no authentication, role enforcement, upload/OCR pipeline, background ingestion or multi-writer coordination. The heading parser is tied to current formats. Context expansion is unbounded by tokens, and answer evidence is assembled into one prompt. There is no calibrated retrieval cutoff, independent factuality guarantee, automatic current-version resolution, per-stage usage accounting or evidence-verifier audit reason. Persisted traces do not fully record model revision, prompt version and corpus hashes for every generated report.')
p('A useful next experiment is to add genuinely difficult questions and actual conflicting versions, then compare reranking and chunk limits with identical held-out cases. Preserve each trace and compare completeness, abstention errors and cost, rather than treating a single perfect-recall run as conclusive.')

page('17. Source references and reproducibility')
p('This report describes the repository state inspected on 14 October 2026 (Asia/Singapore). It uses current source files and existing saved artifacts, plus local chunk-count experiments. It does not include your .env contents, rerun hosted services or alter application behavior.')
h('Local evidence')
p('data/active.json: 86 chunk records, BM25 state and the active snapshot.<br/>data/evaluation-20261014-094303.json: ten cases across dense, sparse and hybrid retrieval.<br/>data/claude-live-smoke.json: three generated-answer integration checks.<br/>evaluation.json: manually authored expected-section/status labels.<br/>tests/test_rag.py and tests/test_embeddings.py: deterministic verification contracts.')
h('Official references used during implementation')
for title,url in [
('Pinecone hybrid retrieval','https://docs.pinecone.io/guides/search/hybrid-search'),
('Pinecone SDK 10 and schema-based Documents API','https://sdk.pinecone.io/python/migration/v10-migration.html'),
('Pinecone-hosted Llama embedding model','https://docs.pinecone.io/models/llama-text-embed-v2'),
('LangChain Claude structured output','https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic/with_structured_output'),
('Claude model identifiers','https://platform.claude.com/docs/en/models/overview'),
('Sentence Transformers retrieval encoders (customization option)','https://sbert.net/docs/sentence_transformer/usage/usage.html'),
('Sentence Transformers cross-encoders (customization option)','https://sbert.net/docs/input_formats.html')]:
    p(f'<b>{escape(title)}</b><br/><link href="{url}" color="#087f8c">{url}</link>','SmallCustom')
h('Appendix conventions')
p('The following listings reproduce the project code with original line numbers. Long lines wrap for readability; continuation lines retain indentation and do not get a new source line number. The listings include all application modules, test helpers and safe configuration examples. The private .env and generated provider data are deliberately excluded. The empty rag/__init__.py has no executable logic to list.')

files=['rag/corpus.py','rag/embeddings.py','rag/sparse.py','rag/engine.py','rag/evaluate.py','rag/cli.py','app.py','tests/test_rag.py','tests/test_embeddings.py','tests/live_smoke.py','evaluation.json','.env.example','requirements.txt','.gitignore']
for file in files:
    page('Appendix: '+file)
    p('Exact source listing; explanations appear in the corresponding sections above.','SmallCustom')
    source=(ROOT/file).read_text()
    # One flowable per source line allows natural page breaks without oversized blocks.
    for no,line in enumerate(source.splitlines(),1):
        wrapped=textwrap.wrap(line.expandtabs(4),width=99,replace_whitespace=False,drop_whitespace=False,break_long_words=True,break_on_hyphens=False) or ['']
        rendered=f'{no:4d}  '+wrapped[0]
        for continuation in wrapped[1:]: rendered+='\n      '+continuation
        story.append(Preformatted(rendered,ParagraphStyle(name='Listing',fontName='Courier',fontSize=6.8,leading=8.7,spaceAfter=0)))

OUT.parent.mkdir(exist_ok=True)
class GuideDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ('HCustom', 'TitleCustom'):
            key = 'heading-' + str(self.seq.nextf('heading'))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)

doc=GuideDoc(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=52,title='Enterprise RAG: Architecture, Code and Design Decisions',author='Project implementation walkthrough',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
