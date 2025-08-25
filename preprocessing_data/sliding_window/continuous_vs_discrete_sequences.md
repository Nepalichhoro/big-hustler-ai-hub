# Continuous Stream Sequences vs Discrete Sequences

## Introduction

When preparing data for sequence models like **RNNs** or **LSTMs**, it is important to distinguish between two kinds of data sources:

- **Continuous streams** (time series, sensor data, logs, etc.)
- **Discrete sequences** (NLP text: sentences, paragraphs, documents)

This distinction influences **how we create training samples**.

---

## Continuous Stream Sequences

A **continuous stream** is a long, uninterrupted series of values indexed by time.

Examples:

- Stock prices recorded every minute
- Heartbeat sensor data
- Application performance metrics (e.g., response times)

### Key Properties

- **No natural boundaries** → the series goes on indefinitely.
- **Sliding windows** are needed to create training samples.
- **Overlap is natural and necessary**, since every subsequence can carry useful patterns.

Example (series of values):

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

Sliding windows (`seq_len=3`):

```text
[1, 2, 3]
[2, 3, 4]
[3, 4, 5]
[4, 5, 6]
[5, 6, 7]
[6, 7, 8]
```

Here, tokens (time steps) are **reused** across multiple windows.

---

## Discrete Sequences

A **discrete sequence** has clear natural boundaries.

Examples:

- Sentences in text: _"I love cats."_
- Paragraphs in an article
- Conversations split by utterances

### Key Properties

- **Natural segmentation** → sentences or paragraphs are already finite training units.
- **No overlap needed** → each sequence is taken as-is.
- **Tokens are unique to a sequence** (a word appears once per sentence, not reused across overlapping windows).

Example (NLP sentence):

```text
"I love cats"
tokens = ["I", "love", "cats"]
```

The model processes this sequence **once**, without sliding windows.

---

## Why This Matters

- **Continuous streams**: models must be trained on _all possible subsequences_. Overlap ensures the model sees context around every time step.
- **Discrete sequences**: boundaries provide natural segmentation. Overlap would be redundant and distort the training distribution.

---

## Unified View

| Property         | Continuous Streams (Time Series)       | Discrete Sequences (NLP)            |
| ---------------- | -------------------------------------- | ----------------------------------- |
| Source           | Sensor data, metrics, logs             | Sentences, paragraphs, documents    |
| Boundaries       | None, infinite stream                  | Natural (sentence/document)         |
| Data preparation | Sliding windows (overlap)              | Direct tokenization (no overlap)    |
| Reuse of tokens  | Yes, tokens appear in multiple windows | No, tokens read once per sequence   |
| Example          | [1,2,3,4,5] → [1,2,3], [2,3,4]         | "I love cats" → ["I","love","cats"] |

---

## Summary

- **Continuous streams** (time series) require **sliding windows with overlap**, since they are infinite and lack natural boundaries.
- **Discrete sequences** (NLP) come with **built-in boundaries** (sentences, documents), so tokens are processed once per sequence.

Think of time series as **one never-ending book** where you must create your own “sentences,” while NLP already **gives you sentences** to work with.
