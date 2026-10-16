# EmbeddingGemma from Scratch: Train a Text Embedding Model in PyTorch

Google's EmbeddingGemma recipe, rebuilt end to end in pure PyTorch on a corpus you generate yourself. The corpus has twelve topics, sixty named entities and a twist: every topic word has a synonym that appears in queries but never in a document, so lexical search fails on exactly the queries a semantic model should get right. Pretrain a tiny Gemma-style decoder on a stream of documents, adapt it to read both ways with masked denoising, and turn it into an encoder with masked mean pooling and a projection head. Then build the loss stack the paper describes: query-document InfoNCE with in-batch negatives, mined hard negatives reweighted by exp(alpha * sim) and a mask for false negatives, the spread-out regularizer, the Matryoshka loss over nested dimensions, and geometric embedding distillation from a teacher. Finish with model soups, truncation, int8 and int4 weights, and an experiment that reproduces the paper's orderings at toy scale: random versus pretrained versus adapted initialization, each loss term's effect, what Matryoshka buys at half the dimensions, when averaging two fine-tunes fails, and what quantization costs.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_world
- [x] **2.** make_document
- [x] **3.** make_query
- [x] **4.** Vocab
- [x] **5.** bm25_rank
- [x] **6.** causal_allowed

---

Built on Deep-ML.
