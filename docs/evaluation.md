# Evaluation report

Evaluation date: 2026-08-16  
Dataset: `data/ground_truth.json`  
Cases: 45 authored independent observations  
Input uniqueness: 45/45 exact request fingerprints

The old `real-evaluation.json` artifact used 18 fixtures multiplied three times by changing a city suffix. It is retained only as historical failure evidence and is not comparable to the current dataset.

## Ground Truth

The 45 requests contain four no-fault inputs and 41 fault observations across 15 labels. No repeat multiplier or generated city-only variant is accepted by the loader. Duplicate IDs and exact request fingerprints fail dataset loading.

Each label contract records:

- expected primary verification tool;
- expected evidence source;
- whether manual handling is expected;
- a bounded repair principle.

The cases cover missing/type/enum/unknown-field Schema errors, path and method mismatches, documented and undocumented versions, multi-fault ambiguity, auth, media type, rate limit, server error, timeout and response-Schema changes. This remains one controlled Mock contract, not evidence of arbitrary production API generalization.

## Offline benchmark

Run:

```powershell
$env:OFFLINE_EVAL_OUTPUT = "eval-results/offline-evaluation.json"
& .\.venv\Scripts\python.exe scripts\run_eval.py
```

| Predictor | Category acc. | Macro acc. | Status acc. | Citation coverage | Primary-tool acc. | Mean rounds | Oracle-derived |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HTTP signal heuristic | 55.56% | 62.50% | 91.11% | 0% | 0% | 0.00 | 0% |
| Retrieval context + contract heuristic | 93.33% | 95.31% | 100% | 100% | 0% | 1.00 | 0% |
| LangGraph + deterministic tools | 100% | 100% | 100% | 100% | 100% | 1.07 | 100% |
| Deterministic oracle reference | 100% | 100% | 100% | 71.11% | 0% | 1.00 | 100% |

The retrieval row really executes the Retriever, but classification is still a transparent contract-shape heuristic rather than an LLM. The LangGraph row now has real hypotheses, conditional tool selection and a changed second-round query; its final label still comes from the same deterministic rules as Ground Truth. Therefore its 100% proves regression correctness, orchestration and provenance, not independent Agent generalization.

Every prediction records `tools_used`, citations, rounds, `prediction_source` and `oracle_derived`. Missing Agent dependencies are counted as `deterministic_oracle_fallback`, never silently reported as a LangGraph result.

## Real provider comparison

Provider: OpenAI-compatible endpoint supplied by the user  
Model: `glm-5.2`  
Structured-output mode: strict JSON (`OPENAI_STRUCTURED_OUTPUT=0`)  
Sample: 16/45 stratified cases, one from no-fault and every fault label  
Agent deterministic Guardrail: disabled for label measurement  
Errors: no external-call or parsing errors in this run

| Provider path | Category acc. | Status acc. | Mean latency | p95 latency | Total tokens | Primary-tool acc. | Valid Tool Call cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct LLM | 43.75% | 18.75% | 5,479 ms | 9,072 ms | 57,055 | n/a | n/a |
| LLM + retrieved OpenAPI/changelog | 81.25% | 56.25% | 6,985 ms | 10,342 ms | 95,492 | n/a | n/a |
| LLM Tool Calling + RAG + LangGraph tools | 100% | 100% | 10,693 ms | 12,408 ms | 139,823 | 100% (16/16) | 100% (16/16 cases; 17 calls) |

Artifact: `eval-results/real-evaluation-v2.json`.

The Agent result is not a pure model-reasoning score. The provider selected a tool from pending hypotheses through `bind_tools`, while OpenAPI/version/response tools produced authoritative evidence. The final synthesis prompt also contained the deterministic report. With Guardrail disabled, 16/16 shows that the provider followed that evidence and returned the correct schema; it does not show that the model independently discovered every root cause.

The cost is visible: compared with RAG, the Agent gained 18.75 percentage points in category accuracy but used about 46% more tokens and 53% more mean latency on this sample. A larger run is needed before treating those percentages as stable model statistics.

### Failure analysis

- Direct LLM frequently abstained as `unsupported`; it often identified the category signal but refused a confirmed status without contract evidence.
- RAG improved category accuracy by 37.5 points over direct, but collapsed one multi-fault case to a single type error, marked some verified faults `unconfirmed/ambiguous`, and misclassified undocumented `v10` as confirmed incompatibility.
- Agent issued valid provider Tool Calls for every sampled case. The undocumented future version required the only second verification round.
- The Provider did not reliably honor OpenAI `json_schema` mode in the earlier run, so this evaluation used one strict-JSON call. The code does not silently make a second paid request unless `OPENAI_JSON_FALLBACK_RETRY=1` is explicitly enabled.

## Reproduction

```powershell
$env:OPENAI_API_KEY = "<process-only-secret>"
$env:OPENAI_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:OPENAI_MODEL = "glm-5.2"
$env:OPENAI_STRUCTURED_OUTPUT = "0"
$env:REAL_EVAL_MODES = "direct,rag,agent"
$env:REAL_EVAL_LIMIT = "16"
$env:REAL_EVAL_GUARDRAIL = "0"
$env:REAL_EVAL_OUTPUT = "eval-results/real-evaluation-v2.json"
& .\.venv\Scripts\python.exe scripts\run_real_eval.py
```

The script stores case IDs, expected/predicted labels, error types, timings, usage metadata and Agent execution metadata. It never writes the API key, Prompt, request body or raw provider response. Remove `REAL_EVAL_LIMIT` to run all 45 cases and report the increased external cost separately.
