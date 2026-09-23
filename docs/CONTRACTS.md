# Shared contracts

`src/contracts.py` defines Pydantic models and Python interfaces. Models reject
unknown fields. Examples in `tests/fixtures/contracts.json` use invented data.

| Model | Purpose |
| --- | --- |
| `Document`, `Chunk` | Source text, IDs, metadata, and character offsets |
| `UserContext` | Server-resolved identity and allowed document IDs |
| `QueryRequest` | Public input: question and baseline/agentic mode |
| `SearchRequest`, `Evidence` | Retrieval filters and ranked passages |
| `ToolRequest`, `ToolResult` | Structured lookups and counts |
| `GenerationRequest` | Question and authorized evidence for generation or critique |
| `CriticDecision` | Evidence sufficiency, confidence, and suggested searches |
| `Answer`, `QueryResponse` | Answer, citations, sources, and trace |
| `Usage`, `TraceStep`, `Trace` | Calls, tokens, cost, timings, and search history |

Retrievers implement `search`, structured tools `execute`, generators `generate`,
and critics `assess`. Their signatures are defined at the end of the module.

## Rules

- Missing identity or empty permissions denies access. Roles and departments do
  not grant access. Filter candidates before returning evidence; call
  `validate_evidence` before exposing results. `GenerationRequest` checks scope.
- Chunk offsets are half-open character indices into the original text. Unknown
  timestamps are null; supplied timestamps need a timezone.
- Citation IDs must be unique and resolve to supplied evidence. This validates
  references, not whether the evidence supports each claim.
- Tool adapters must validate supported filters and apply permissions before
  aggregation. The tool interfaces do not implement storage or authentication.
- Trace totals include generation and retries. Do not add totals to step usage
  again. Unknown cost is null. Keep credentials and benchmark answers out of
  runtime requests and traces.

`src/config.py` pins dataset/model revisions and defines defaults. Generation
settings and budget enforcement are in `src/generation/generator.py`.
