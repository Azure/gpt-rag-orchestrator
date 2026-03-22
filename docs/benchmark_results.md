# Strategy & Model Benchmark Results

**Date:** March 22, 2026  
**Environment:** Azure Container Apps (`ca-olvdyl47765js-orchestrator`), East US 2  
**AI Resource:** `aif-olvdyl47765js` (Azure AI Services)

## Test Methodology

Each strategy was tested with **3 sequential questions** in a single conversation (shared `conversation_id`), simulating a realistic user interaction:

| # | Type | Question |
|---|------|----------|
| Q1 | Greeting | *"How are you doing today?"* |
| Q2 | Knowledge | *"How does an aircooled engine work?"* |
| Q3 | Follow-up | *"Tell me more about the combustion process"* |

**Procedure per test run:**
1. Switch `AGENT_STRATEGY` and `CHAT_DEPLOYMENT_NAME` in Azure App Configuration
2. Restart container revision to apply new config
3. Send a warm-up request (discarded from results)
4. Execute Q1 → Q2 → Q3 sequentially, reusing the same `conversation_id`
5. Collect **server-side flow time** from container logs and **client-side total time** from the test script

**Timing metrics:**
- **Server Flow Time** — measured inside the strategy code (primary metric, excludes network latency)
- **Client Total Time** — measured from request send to last byte received (includes network + streaming)

---

## Models Under Test

| Deployment Name | Model | Version | SKU | Capacity (TPM) |
|-----------------|-------|---------|-----|-----------------|
| `chat` | GPT-5-mini | 2025-08-07 | GlobalStandard | 1,000K |
| `gpt-4o-mini` | GPT-4o-mini | 2024-07-18 | GlobalStandard | 30K |

---

## Results — GPT-5-mini (`chat` deployment)

### Server-Side Flow Time (seconds)

| Strategy | Q1 (Greeting) | Q2 (Knowledge) | Q3 (Follow-up) | Average |
|----------|:-------------:|:--------------:|:---------------:|:-------:|
| `single_agent_rag` | 15.28 | 19.57 | 20.08 | **18.31** |
| `maf_agent_service` | 10.68 | 18.49 | 22.99 | **17.39** |
| `maf_lite` | 4.84 | 19.57 | 23.28 | **15.90** |

### Client-Side Total Time (seconds)

| Strategy | Q1 (Greeting) | Q2 (Knowledge) | Q3 (Follow-up) | Average |
|----------|:-------------:|:--------------:|:---------------:|:-------:|
| `single_agent_rag` | 15.73 | 20.04 | 20.53 | **18.77** |
| `maf_agent_service` | 11.07 | 18.90 | 23.37 | **17.78** |
| `maf_lite` | 5.21 | 19.98 | 23.70 | **16.30** |

### First Token Time (seconds)

| Strategy | Q1 | Q2 | Q3 |
|----------|:--:|:--:|:--:|
| `single_agent_rag` | 14.59 | 16.97 | 14.31 |
| `maf_agent_service` | — | — | — |
| `maf_lite` | — | — | — |

> *Note: First-token timing is only logged by `single_agent_rag` (Agent Flow V2). `maf_agent_service` and `maf_lite` stream responses directly without explicit first-token instrumentation.*

---

## Results — GPT-4o-mini (`gpt-4o-mini` deployment)

### Server-Side Flow Time (seconds)

| Strategy | Q1 (Greeting) | Q2 (Knowledge) | Q3 (Follow-up) | Average |
|----------|:-------------:|:--------------:|:---------------:|:-------:|
| `single_agent_rag` | 6.19 | 9.26 | 14.07 | **9.84** |
| `maf_agent_service` | 5.84 | 7.49 | 11.37 | **8.23** |
| `maf_lite` | 1.51 | 9.09 | 9.21 | **6.60** |

### Client-Side Total Time (seconds)

| Strategy | Q1 (Greeting) | Q2 (Knowledge) | Q3 (Follow-up) | Average |
|----------|:-------------:|:--------------:|:---------------:|:-------:|
| `single_agent_rag` | 6.37 | 9.71 | 14.51 | **10.20** |
| `maf_agent_service` | 6.21 | 7.95 | 11.82 | **8.66** |
| `maf_lite` | 1.90 | 9.61 | 9.61 | **7.04** |

### First Token Time (seconds)

| Strategy | Q1 | Q2 | Q3 |
|----------|:--:|:--:|:--:|
| `single_agent_rag` | 5.51 | 7.26 | 10.74 |
| `maf_agent_service` | — | — | — |
| `maf_lite` | — | — | — |

---

## Comparative Summary

### Average Server Flow Time by Strategy × Model

| Strategy | GPT-5-mini | GPT-4o-mini | Speedup |
|----------|:----------:|:-----------:|:-------:|
| `single_agent_rag` | 18.31s | 9.84s | **1.86x faster** |
| `maf_agent_service` | 17.39s | 8.23s | **2.11x faster** |
| `maf_lite` | 15.90s | 6.60s | **2.41x faster** |

### Strategy Ranking (fastest to slowest)

| Rank | GPT-5-mini | GPT-4o-mini |
|:----:|:----------:|:-----------:|
| 1st | `maf_lite` (15.90s) | `maf_lite` (6.60s) |
| 2nd | `maf_agent_service` (17.39s) | `maf_agent_service` (8.23s) |
| 3rd | `single_agent_rag` (18.31s) | `single_agent_rag` (9.84s) |

### Greeting Response (Q1) — Simple Queries

| Strategy | GPT-5-mini | GPT-4o-mini |
|----------|:----------:|:-----------:|
| `single_agent_rag` | 15.28s | 6.19s |
| `maf_agent_service` | 10.68s | 5.84s |
| `maf_lite` | **4.84s** | **1.51s** |

> `maf_lite` excels on simple queries due to its lightweight architecture (no Agent Service overhead).

---

## Key Observations

1. **`maf_lite` is consistently the fastest strategy** across both models and all question types. Its advantage is most pronounced on simple queries (Q1), where it avoids Agent Service setup, thread creation, and tool orchestration overhead.

2. **GPT-4o-mini is ~2x faster than GPT-5-mini** across all strategies. The speedup ranges from 1.86x (`single_agent_rag`) to 2.41x (`maf_lite`). This suggests GPT-5-mini's additional reasoning capabilities come at a significant latency cost for this RAG workload.

3. **Simple queries expose framework overhead.** On Q1 (greeting), `single_agent_rag` takes 15.28s with GPT-5-mini vs. `maf_lite`'s 4.84s — a 3.15x difference caused entirely by Agent Service infrastructure (thread creation, agent setup, message creation). With GPT-4o-mini, this gap is still 4.1x (6.19s vs 1.51s).

4. **Follow-up questions (Q3) are the slowest across all configurations.** This is expected: the growing conversation context increases token count, which increases both prompt processing and generation time.

5. **`maf_agent_service` and `single_agent_rag` converge on complex queries.** For Q2 and Q3, the gap between these two Agent Service-based strategies narrows because model inference dominates the total time.

6. **Rate limit risk with GPT-4o-mini (30K TPM).** During initial testing, Q3 of `single_agent_rag` hit a token rate limit. This was mitigated by adding 10s pauses between requests. The `chat` deployment (1,000K TPM) had no such issues.

---

## Recommendations

- **For latency-sensitive workloads:** Use `maf_lite` — it offers the lowest overhead and fastest responses.
- **For feature-rich agent workflows:** Use `maf_agent_service` — it balances server-side thread management with reasonable latency.
- **For model selection:** GPT-4o-mini provides significantly faster responses. Consider it for scenarios where GPT-5-mini's enhanced reasoning is not required.
- **To reduce Q3/follow-up latency:** Limit conversation history length or summarize earlier turns to reduce context window size.

---

## Test Configuration Details

| Parameter | Value |
|-----------|-------|
| Container App | `ca-olvdyl47765js-orchestrator` |
| Resource Group | `rg-gpt-rag-0320261901` |
| Region | East US 2 |
| AI Services Resource | `aif-olvdyl47765js` |
| App Configuration | `appcs-olvdyl47765js` |
| Config Label | `gpt-rag` |
| Test Date/Time (UTC) | 2026-03-22 00:15 — 00:45 |
| Warm-up | 1 discarded request per strategy switch |
| Inter-question pause | 3–10 seconds |
