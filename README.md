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
- [x] **7.** TransformerBlock
- [x] **8.** TinyLM
- [x] **9.** pretrain_lm
- [x] **10.** adapt_encoder
- [x] **11.** Embedder
- [x] **12.** contrastive_loss
- [x] **13.** spread_out_loss
- [x] **14.** mrl_loss
- [x] **15.** OracleTeacher
- [x] **16.** sample_batch
- [x] **17.** train_embedder
- [x] **18.** uniform_soup
- [x] **19.** retrieval_eval
- [x] **20.** fake_quantize_model
- [x] **21.** build_experiment
- [x] **22.** embedding_experiment

## Results

```
Corpus: 360 documents over 12 topics, vocabulary of 220 tokens.
Queries paraphrase the topic words with synonyms that never occur in any document.

1. Lexical search (BM25)
   lexical queries  R@1 0.91 nDCG@10 0.96
   synonym queries  R@1 0.08 nDCG@10 0.37   <- the words are not in the documents

2. Pretrain a decoder on a stream of documents, then adapt it to read both ways
   next-token loss 5.40 -> 3.96; masked-token loss after adaptation 4.88 -> 3.54
   cosine between a word and its synonym in the token embeddings: 0.80 (random init 0.05)

3. Initialization, trained on lexical pairs only (no synonym ever seen in a training pair)
   random   init: synonym queries R@1 0.08 nDCG@10 0.24 | lexical queries R@1 0.81 nDCG@10 0.90
   decoder  init: synonym queries R@1 0.42 nDCG@10 0.61 | lexical queries R@1 0.66 nDCG@10 0.82
   adapted  init: synonym queries R@1 0.59 nDCG@10 0.77 | lexical queries R@1 0.67 nDCG@10 0.84
   Pretraining lifts synonym recall by 0.34 with no synonym in any training pair: the words come from the stream, not from the pairs.
   The bidirectional adaptation adds to it, as in the paper's encoder-decoder ablation.
   Random init fits lexical queries fastest: the pretrained blocks must first be re-fit to the contrastive task.

4. The loss stack (adapted init, 20% of training queries use synonyms)
   in-batch only                   : synonym R@1 0.56 nDCG@10 0.76 | lexical R@1 0.68 nDCG@10 0.83 | final loss 0.190
   + hard negatives                : synonym R@1 0.57 nDCG@10 0.76 | lexical R@1 0.76 nDCG@10 0.88 | final loss 0.552
   + spread-out, MRL, distillation : synonym R@1 0.63 nDCG@10 0.79 | lexical R@1 0.80 nDCG@10 0.90 | final loss 1.823

5. Matryoshka truncation, recall@1 on synonym queries at 32 / 16 / 8 dimensions
   single-size loss  : 0.57 / 0.48 / 0.26
   Matryoshka loss   : 0.63 / 0.61 / 0.36

6. A soup of two fine-tunes of the same checkpoint: recall@1 on synonym queries along the segment
   alpha 0.00 0.57 | alpha 0.25 0.58 | alpha 0.50 0.58 | alpha 0.75 0.61 | alpha 1.00 0.63
   The midpoint holds up: the two runs share a basin and averaging is safe.

7. Weight quantization of the full model, recall@1 on synonym queries
   float 0.63 | int8 0.63 | int4 0.63

Done in 36s. Numbers move by a few points across seeds; the orderings are what the paper predicts at toy scale.
Corpus: 360 documents over 12 topics, vocabulary of 220 tokens.
Queries paraphrase the topic words with synonyms that never occur in any document.

1. Lexical search (BM25)
   lexical queries  R@1 0.91 nDCG@10 0.96
   synonym queries  R@1 0.08 nDCG@10 0.37   <- the words are not in the documents

2. Pretrain a decoder on a stream of documents, then adapt it to read both ways
   next-token loss 5.40 -> 3.96; masked-token loss after adaptation 4.88 -> 3.54
   cosine between a word and its synonym in the token embeddings: 0.80 (random init 0.05)

3. Initialization, trained on lexical pairs only (no synonym ever seen in a training pair)
   random   init: synonym queries R@1 0.08 nDCG@10 0.24 | lexical queries R@1 0.81 nDCG@10 0.90
   decoder  init: synonym queries R@1 0.42 nDCG@10 0.61 | lexical queries R@1 0.66 nDCG@10 0.82
   adapted  init: synonym queries R@1 0.59 nDCG@10 0.77 | lexical queries R@1 0.67 nDCG@10 0.84
   Pretraining lifts synonym recall by 0.34 with no synonym in any training pair: the words come from the stream, not from the pairs.
   The bidirectional adaptation adds to it, as in the paper's encoder-decoder ablation.
   Random init fits lexical queries fastest: the pretrained blocks must first be re-fit to the contrastive task.

4. The loss stack (adapted init, 20% of training queries use synonyms)
   in-batch only                   : synonym R@1 0.56 nDCG@10 0.76 | lexical R@1 0.68 nDCG@10 0.83 | final loss 0.190
   + hard negatives                : synonym R@1 0.57 nDCG@10 0.76 | lexical R@1 0.76 nDCG@10 0.88 | final loss 0.552
   + spread-out, MRL, distillation : synonym R@1 0.63 nDCG@10 0.79 | lexical R@1 0.80 nDCG@10 0.90 | final loss 1.823

5. Matryoshka truncation, recall@1 on synonym queries at 32 / 16 / 8 dimensions
   single-size loss  : 0.57 / 0.48 / 0.26
   Matryoshka loss   : 0.63 / 0.61 / 0.36

6. A soup of two fine-tunes of the same checkpoint: recall@1 on synonym queries along the segment
   alpha 0.00 0.57 | alpha 0.25 0.58 | alpha 0.50 0.58 | alpha 0.75 0.61 | alpha 1.00 0.63
   The midpoint holds up: the two runs share a basin and averaging is safe.

7. Weight quantization of the full model, recall@1 on synonym queries
   float 0.63 | int8 0.63 | int4 0.63

Done in 26s. Numbers move by a few points across seeds; the orderings are what the paper predicts at toy scale.
```
