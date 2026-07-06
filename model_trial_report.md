# Model Trial Report: Intent/Topic Extraction

## Task
Extract structured JSON (`intent`, `topics`, `concepts`, `tags`, `summary`) from 34 LLM conversation histories (~4000 chars each, with 10 Q&A pairs per conversation).

## Models Tested

| Model         | Size   | Speed (34 convs) | Intent Types | Topics Found | Concepts Found | "other" % |
|---------------|--------|-----------------|--------------|-------------|----------------|-----------|
| llama3.2:3b   | 2.0 GB | 68s             | 6           | 28/34       | 30/34          | 53% (18)  |
| qwen3.6:35b-mlx | 21.9 GB | 700s          | 7           | 14/34       | —              | 59% (20)  |
| **llama3.1:8b** | **4.7 GB** | **164s**   | **9**       | **32/34**   | **34/34**      | **6% (2)** |

## Detailed Results

### llama3.2:3b
- **Intents**: explain(9), other(18), build(3), compare(2), design(2)
- **Speed**: ~2-5s per call → 68s total (6 workers)
- **Issues**: Cannot follow JSON-only instruction for prompts >3000 chars. Frequently returns narrative text that fails JSON parsing, triggering the `"other"` fallback. Topics are keyword-level (e.g., `["difficulty", "QI"]`).
- **Verdict**: Too small for the task.

### qwen3.6:35b-mlx
- **Intents**: explain(8), other(20), build(2), design(1), analyze(1), explore(1), research(1)
- **Speed**: ~20-30s per call → 700s total (6 workers)
- **Issues**: Includes a `"thinking"` field with irrelevant reasoning tokens. Worse topic coverage (14/34) — frequently omits JSON fields entirely. 10x slower than 8B with no meaningful accuracy gain.
- **Verdict**: Overkill — quality doesn't justify the speed penalty.

### llama3.1:8b (Recommended)
- **Intents**: explain(17), build(6), compare(3), research(2), other(2), analyze(1), debug(1), design(1), explore(1)
- **Speed**: ~2-5s per call → 164s total (6 workers)
- **Strengths**: Perfect JSON compliance on all 34 conversations. Rich intent diversity with only 2/34 falling back to "other". Topics are conceptual and meaningful (e.g., `["Quai stability", "rising difficulty"]`). Fits in ~5GB VRAM on Apple Silicon.
- **Verdict**: Best balance of speed, accuracy, and reliability.

## Graph Statistics (llama3.1:8b)
- 56 unique topics, 137 unique concepts
- 243 nodes, 281 edges in the knowledge graph
- 66 clusters from 6616 Q&A pairs

## Conclusion
**llama3.1:8b** is the recommended model for intent/topic extraction from conversation data. The 3B model lacks the capacity for structured output at this prompt length, while 35B is unnecessarily large and slow. 8B hits the sweet spot for Apple Silicon: fast (<5s/call), accurate (6% fallback), and rich output.
