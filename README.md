# Enterprise Document Q&A Starter

A local Streamlit interface over the four files in `Sample Documents`, using LangChain's Claude integration and Pinecone's schema-based Documents API (Python SDK 10 / API 2026-07). No department or role restrictions. Generation uses only retrieved documents and abstains when evidence is insufficient.

## Run

Requires Python 3.11 or later.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env
# Edit .env and supply your Pinecone and Claude API keys.
python -m rag.cli preview
streamlit run app.py
```

Open http://localhost:8501. The Document explorer works without keys. Click **Ingest sample documents** to create the configured Pinecone index if necessary, embed and upload the corpus, and activate its snapshot. Then use Ask & inspect or Compare retrieval. Alternatively:

```bash
python -m rag.cli ingest
python -m rag.cli evaluate
python -m rag.cli evaluate --generate --presets dense sparse hybrid hybrid-reranked
```

Ingestion sends document content to Pinecone Inference for dense embeddings and stores chunks and metadata in Pinecone. Answering sends selected excerpts to Anthropic Claude; optional reranking sends candidates to Pinecone inference. Keys stay in `.env`, which is ignored by git. Do not paste keys into chat. Embedding, database, reranking, and generation calls incur account usage. Pinecone provides both dense embeddings and vector storage; Claude provides answer generation and evidence verification. Only `PINECONE_API_KEY` and `ANTHROPIC_API_KEY` are required. `CHAT_MODEL` defaults to `claude-sonnet-5-5` and can be changed to a Claude model available to your account that supports native structured output.

## Guided design

1. **Inspect:** `preview` writes `data/preview.json` containing exact source text and metadata. The current parser supports UTF-8 TXT only. Add format-specific extraction with page/paragraph provenance before adding PDFs, DOCX, or HTML.
2. **Chunk:** Prefer document subsections over arbitrary windows. Default maximum is approximately 1,800 characters with two lines of overlap within a subsection; one long line can exceed this target. Headings and code blocks retain line provenance. Dense and sparse inputs include title and section labels. Keeping full sections during context expansion preserves qualifications in split sections. Change `load_corpus` settings, preview, then reingest to experiment.
3. **Embed:** Dense vectors use Pinecone-hosted `llama-text-embed-v2`, 1,024 dimensions by default. Sparse vectors use a corpus-fitted BM25 vocabulary, k1=1.2, b=.75; document vectors contain length-normalized term frequency and query vectors contain IDF. Tokenization preserves hyphenated/underscored terms and numbers. The encoder is persisted alongside its corpus so dimensions cannot drift independently. Dense documents use `input_type=passage`; questions use `input_type=query`. Input truncation is disabled, and returned vector counts/dimensions are checked before storage.
4. **Store:** One Pinecone schema declares `dense` (cosine) and `sparse` fields explicitly. Each record also holds `text` and flat metadata fields. Use a new index name if your existing index has an incompatible schema. Cloud/region are configurable in `.env`; defaults are AWS/us-east-1. The code validates schema and embedding configuration. When switching from an existing 1,536-dimensional OpenAI index, choose a new `PINECONE_INDEX` (for example `enterprise-rag-claude`) before reingestion; old indexes are preserved.
5. **Retrieve:** Query dense and sparse fields separately, then fuse their ranked lists with weighted reciprocal rank fusion: `alpha/(60+dense_rank) + (1-alpha)/(60+sparse_rank)`. Default alpha=.5, 30 candidates per signal, 12 context chunks; score scales are never mixed directly. Alpha endpoints run a single signal. Optional `bge-reranker-v2-m3` reranks the union. Section expansion adds remaining chunks of selected sections. Document-type filters apply to both searches.
6. **Answer:** Claude returns native structured output for generation and verification. A structured answer must cite exact quotes using numbered evidence IDs. The code checks source ranges and verbatim quotes, then a second model call checks support and omitted conditions. Unsupported or empty answers become **“I don't know based on the supplied documents.”** Formal HR/compliance manuals take precedence over summaries; actual conflicts must cite both sources. Additional qualifications are not treated as contradictions. Evidence and questions are treated as untrusted data.
7. **Evaluate:** `evaluation.json` now contains fifteen editable gold cases, including the original cases: international remote work, breach deadlines, HTTP 429, onboarding training, L&D, retirement, webhook signatures, two absent topics, and an instruction-injection attempt. The UI and CLI compare dense, sparse, hybrid, and optional reranked retrieval with identical question sets and context budgets.

The source documents are dated 2024–2025. Answers report their contents; they do not verify current law or policy outside the corpus. Referenced but absent documents, such as the Travel Policy, cannot supply missing details. A person joining 95 days ago is not automatically eligible for international remote work: HR §3.1 and §3.5 supply additional conditions. The onboarding summary is compatible with those conditions, rather than a demonstrated conflict.

## Metadata contract

| Field | Purpose |
|---|---|
| `document_id` | Stable filename-derived document identity |
| `chunk_id` | Deterministic content/location-derived record identity |
| `filename`, `source_path`, `title` | Source navigation and readable citations |
| `document_type` | hr, compliance, product, onboarding; retrieval filter |
| `version` | Version printed in the source |
| `effective_date`, `updated_date` | Source dates where stated; month-only updates retain YYYY-MM precision |
| `owner` | Explicit source owner/reviewer; onboarding header identifies People Operations |
| `section_number`, `section_title` | Citation and complete-section expansion |
| `line_start`, `line_end` | Inclusive, one-based line locations in original TXT |
| `content_hash` | SHA-256 of the complete source file |
| `ingested_at` | UTC ingestion timestamp |

Unstated owner/date fields are omitted. Access groups and geographic applicability are not inferred. Version identifiers are source labels, not sortable evidence of authority across different documents.

## Updates and reproducibility

Each corpus + embedding configuration gets a deterministic namespace. Reingestion upserts identical IDs, waits for all IDs to be fetchable, and atomically updates `data/active.json`. A changed or removed document gets a different namespace; querying the new snapshot cannot see obsolete chunks. Failed ingestion leaves the previous active manifest unchanged. Pinecone is eventually consistent; fetch visibility does not guarantee immediate search convergence, so retry queries briefly after ingestion if necessary. Old namespaces are retained for rollback and must be cleaned up explicitly in Pinecone when no longer needed. This starter is designed for one local ingestion process at a time.

The local active manifest includes source excerpts and the BM25 encoder. Keep it with its index; rerun ingestion to recover if it is lost. Evaluation reports and downloadable question traces record the namespace and retrieval settings. `requirements.lock` captures the tested environment; `requirements.txt` defines compatible major-version ranges.

## Reading evaluation results

- **Candidate recall:** fraction of expected document/section pairs in the fused candidate pool.
- **Section recall:** fraction reaching the final answer context, including section expansion.
- **All required sections:** whether every expected section reaches context. Prioritize this for multi-condition questions.
- **Status matches:** answered/unknown classification matches the gold case (only with generation enabled).
- **Required-term coverage:** literal expected terms appear in the answer; diagnostic only, not semantic correctness.
- **Latency:** retrieval and total elapsed seconds. Runs are sequential across cases; provider usage may be billed separately.

Reports are saved to `data/evaluation-*.json`. Review the cited answers manually for factual completeness and conflicts. The model evidence verifier reduces unsupported claims but cannot guarantee factuality or perfect abstention. Gold cases are editable in `evaluation.json`; add contradictory document fixtures and expected behavior when actual conflicting versions become available. There is no calibrated relevance cutoff: unrelated results can reach generation, where the evidence checks must abstain.

## Validation and limits

```bash
python -m pytest -q
# Optional live check: uses Pinecone and Claude account credits
python tests/live_smoke.py
```

Tests check full source-line coverage, stable IDs, gold-section existence, sparse matching and persistence, fusion endpoints, Pinecone SDK wire shapes, retrieval filtering/section expansion, and rejection of fabricated quotes and unsupported answers. Streamlit smoke tests exercise the offline explorer and missing-key handling. Live validation completed with 86 indexed chunks, 12 passing unit tests, and successful cited answers for HTTP 429 and international remote work plus abstention on the absent Singapore per diem rate. The saved retrieval comparison achieved mean required-section recall of 1.000 for dense, 0.929 for sparse, and 1.000 for hybrid on the seven answerable gold cases; this small set does not establish broader accuracy. Live traces are saved in `data/claude-live-smoke.json`, and comparisons in `data/evaluation-*.json`. Embedding tests additionally check passage/query modes, batching, dimension validation, and operation without an OpenAI key.

This is a local starter, not a deployed multi-user service: no authentication, authorization, document upload, OCR, or background ingestion. Large corpora need token-aware limits, bounded section expansion, stronger operational retries, embedding caching, evaluation coverage, and concurrency controls.

## API references

- [Pinecone hybrid retrieval](https://docs.pinecone.io/guides/search/hybrid-search)
- [Pinecone SDK 10 and document schemas](https://sdk.pinecone.io/python/migration/v10-migration.html)
- [LangChain Claude structured output](https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic/with_structured_output)
- [Pinecone dense embedding model](https://docs.pinecone.io/models/llama-text-embed-v2)

## 15-question evaluation deliverable

The completed evaluation is in `docs/Enterprise_RAG_15_Question_Evaluation_Report.pdf`, with all 60 trial traces in `docs/evaluation_results.json`, a per-trial spreadsheet in `docs/evaluation_summary.csv`, and explicit source/rubric review notes in `docs/evaluation_review.json`.

It compares dense, sparse, hybrid and hybrid-reranked retrieval over the same frozen 15 questions, with Recall@12, Precision@12, MRR@12, nDCG@12, context completeness, generated-answer rubric grades and failure analysis. Semantic scores use a separate Claude call with the same configured model, not independent human certification. The report preserves frozen scores and separately documents a post-run question-scope correction; inspect the limitations before interpreting percentages.

```bash
.venv/bin/python scripts/run_deliverable_evaluation.py
# Offline PDF creation requires reportlab in the selected Python environment:
python scripts/build_evaluation_report.py
```

The live runner resumes completed trials for an identical snapshot, gold hash and model configuration. Archive the existing result JSON before a fresh run. It uses configured service credits. PDF generation reads saved results only and never reads keys or calls providers. Standard CLI/UI evaluation also exposes the new ranked metrics; semantic rubric grading and complete traces are provided by the deliverable runner. The old architecture guide documents its original 14 October implementation snapshot.
