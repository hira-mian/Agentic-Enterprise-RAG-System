# Search agents

`EvidenceCritic` checks whether passages answer a question and suggests searches
for missing facts. `SearchAgent` runs search, critic, follow-up searches, and the
existing grounded generator. It only generates after the critic approves the
evidence above the configured confidence threshold.

`TextRouter` uses the question and critic's follow-up queries. `ModelRouter` can
choose text search, registered structured tools, or both. Register each tool with
its source, supported operations, and filter fields using `ToolSpec`. Unknown
tools and filters fail explicitly. Adapters must validate filter values and
apply user permissions **before** computing counts. Counts must come with citable
aggregate evidence; bare numbers are rejected. Actual database/API adapters are
not included yet.

## Connect the components

Supply a configured provider, a loaded retriever implementing `search`, and a
server-resolved user scope:

```python
from src.agents.routing import ModelRouter
from src.agents.search_agent import AgentConfig, SearchAgent
from src.agents.search_critic import EvidenceCritic
from src.contracts import SearchRequest
from src.generation.generator import GroundedGenerator

agent = SearchAgent(
    retriever=retriever,
    critic=EvidenceCritic(provider),
    generator=GroundedGenerator(provider),
    router=ModelRouter(provider),
    tools={},  # Add validated ToolSpec adapters when available.
    config=AgentConfig(max_rounds=3, max_cost_usd=0.10),
)
response = agent.run(SearchRequest(query="What is the upload limit?", user=user))
print(response.answer.text)
print(response.trace.model_dump_json(indent=2))
```

Here `provider`, `retriever`, and `user` are application-supplied objects.
`CorpusBM25.load(index_path, corpus_path=corpus_path)` supports `search` and returns
whole-document evidence. The dense retriever currently returns IDs/scores through `rank`
and needs an evidence adapter before it can be used here. A hybrid retriever can
replace BM25 once it implements the same interface.

Configure the provider's explicit paid-call opt-in, model, prices, and budget
before a live run. Share a provider instance across routing, critic, and generation
to share its cost reservation. No paid calls are made by the tests.

## Limits and traces

Stops: sufficient evidence, repeated/empty evidence, round limit, budget, or error.
Source/date filters and user scope persist across rounds; tool calls cannot widen
them. Citation and chunk IDs are deduplicated. New evidence gets context space
first. Unknown dates cannot satisfy date filters.

Time and token limits are checked between component calls; they cannot interrupt
an in-flight call and can be exceeded by that call. Configure provider timeouts
and output-token limits too. The existing provider reserves monetary cost before
calls; the agent separately checks reported run usage. Unknown cost after an API
call stops the run conservatively. Traces record routing reasons, queries, tool
filters, evidence IDs, critic decisions, time, tokens, calls, and known cost.

The default confidence threshold (0.8) is provisional, not calibrated. Scripted
fixtures cover evidence sufficiency, conflicts, missing dates, routing, scope,
budgets, and failures. They verify behavior, not real LLM judgment quality.
Live critic calibration and generated-answer evaluation remain separate work.

Run offline checks:

```sh
python -m pytest tests/test_agents.py -q
```

## Small development pilot

The pilot runs five development questions, spread across question categories,
using BM25, the critic, and the answer generator. It allows two search rounds.
It uses text routing; structured tools are not part of this first pilot.

First, check that your local data and index are ready. This makes **no LLM calls**:

```sh
python -m src.agents.pilot
```

It expects Hira's full-corpus files at:

- `.cache/corpus/documents.parquet`
- `.cache/corpus/questions.jsonl`
- `.cache/corpus/indexes/bm25/`

Obtain these matching artifacts from Hira, or build them with the README's dataset
and BM25 commands. Older reduced-corpus indexes use a different format and cannot
be substituted. Custom locations are accepted through `--corpus`, `--questions`,
and `--index`. The index's corpus hash is checked before running.

For a paid run, set `ANTHROPIC_API_KEY` in your terminal environment. Do not put
it in a tracked file. Replace the capitalized placeholders below with your
chosen model, total spending cap in dollars, and its current token prices:

```sh
python -m src.agents.pilot --live \
  --model MODEL_ID --budget-usd TOTAL_DOLLARS \
  --input-price INPUT_DOLLARS_PER_MILLION \
  --output-price OUTPUT_DOLLARS_PER_MILLION
```

One provider budget covers **all questions**, critics, and answers in this run.
Each new run has a new budget. Reservations are conservative, so a run can stop
before spending the entire cap. The default command never enables paid calls.
Use `--count 10` for a larger pilot after reviewing the first five.

Outputs stay local in `.cache/agent-pilot/<timestamp>/`:

| File | What to inspect |
| --- | --- |
| `case_01.json`, etc. | Question, answer, retrieved text, actual model inputs/outputs, critic decisions, follow-up searches, and usage |
| `review.json` | Separate reference answers and blank review fields for you to fill in |
| `summary.json` | Completed questions, stop reasons, tokens, reported cost, and latency |
| `manifest.json` | Model, prices, settings, prompts, index identity, and code fingerprints |

Each completed question is saved immediately. The pilot stops on errors or a
budget stop rather than repeatedly calling a failing service. Reference answers
are saved for human review but never passed to the agent. All indexed documents
are explicitly allowed for this synthetic benchmark run; this is not a production
permission policy. Retrieved text and prompts are stored locally for inspection.

Review whether the critic was right, follow-up searches helped, the answer was
correct and complete, and the citations supported it. Leave unreviewed fields
blank. These few cases are a debugging pilot, not an answer-quality benchmark.
