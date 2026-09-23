# Shared contracts (v1)

`src/contracts.py` defines validated Pydantic models and synchronous Python
protocols. Unknown fields are rejected; models are frozen. JSON fixtures in
`tests/fixtures/contracts.json` are invented test data, not benchmark evidence.

| Boundary | Contract |
| --- | --- |
| Ingestion | `Document`: original ID, source, title, content; optional timestamp |
| Chunking | `Chunk`: stable ID, parent document, original character offsets, text |
| Identity | `UserContext`: server-resolved identity, relevance context, allowed document IDs |
| Public input | `QueryRequest`: question and baseline/agentic mode only |
| Retrieval | `SearchRequest` → tuple of `Evidence` through `Retriever.search` |
| Structured lookup | `ToolRequest` → `ToolResult` through `StructuredTool.execute` |
| Critique | `GenerationRequest` → decision and usage through `SearchCritic.assess` |
| Generation | `GenerationRequest` → answer and usage through `Generator.generate` |
| Public output | `QueryResponse`: answer, cited evidence, trace |

Authorization is not inferred from role or department. Empty scope or missing
identity denies access. Adapters must filter candidates before returning evidence
and call `validate_evidence` before reranking, critique, generation, or exposure.
`GenerationRequest` enforces this check. Public responses still require a
server-side scope check; schema validation alone is not authentication.
Full benchmark evaluation uses an explicit synthetic benchmark identity with
all indexed document IDs allowed. Record this policy separately from access tests.

Chunk offsets are half-open Python character indices into the original content.
Create IDs from document ID, offsets, and chunking version. Citation IDs must be
unique within a response and resolve to supplied evidence. Validation checks
citation existence, not whether the cited text actually supports the claim.
Unknown timestamps remain null; supplied timestamps require timezone offsets. Never infer recency from source order.

Tool adapters must allowlist supported filters, scope rows before aggregation,
and return a validation error for unsupported fields. The request schema does
not implement a database or arbitrary query execution.

Trace steps record queries, tools, evidence IDs, critic decisions, elapsed time,
and per-step usage. Final trace usage includes generation and retries as well as
search/critic calls; do not add totals to step counters twice. Unknown costs are
null. Traces must not contain credentials and must respect the user's scope.

`src/config.py` provides dataset revision and bounded defaults. Paid calls default
to a zero-dollar budget. These are configuration contracts, not an implemented
budget-enforcement loop. Never send benchmark reference answers to these runtime
interfaces. Team review is still needed before downstream implementation.
