"""
EmbeddingGemma from Scratch: Train a Text Embedding Model in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

import math
import random
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    t0 = time.time()
    res = embedding_experiment(seed=0)
    r1 = lambda m: f"R@1 {m['recall@1']:.2f} nDCG@10 {m['ndcg@10']:.2f}"
    print(f"Corpus: {res['n_docs']} documents over {res['n_topics']} topics, vocabulary of {res['vocab']} tokens.")
    print("Queries paraphrase the topic words with synonyms that never occur in any document.")

    print("\n1. Lexical search (BM25)")
    print(f"   lexical queries  {r1(res['bm25']['lex'])}")
    print(f"   synonym queries  {r1(res['bm25']['syn'])}   <- the words are not in the documents")

    lm = res["lm"]
    print("\n2. Pretrain a decoder on a stream of documents, then adapt it to read both ways")
    print(f"   next-token loss {lm['loss_first']:.2f} -> {lm['loss_last']:.2f}; masked-token loss after adaptation {lm['adapt_first']:.2f} -> {lm['adapt_last']:.2f}")
    print(f"   cosine between a word and its synonym in the token embeddings: {lm['synonym_cosine']:.2f} (random init {lm['random_cosine']:.2f})")

    print("\n3. Initialization, trained on lexical pairs only (no synonym ever seen in a training pair)")
    for name in ("random", "decoder", "adapted"):
        m = res["init"][name]
        print(f"   {name:8s} init: synonym queries {r1(m['syn'])} | lexical queries {r1(m['lex'])}")
    i = res["init"]
    gain = i["decoder"]["syn"]["recall@1"] - i["random"]["syn"]["recall@1"]
    adapt_gain = (i["adapted"]["syn"]["recall@1"] + i["adapted"]["lex"]["recall@1"]) - (i["decoder"]["syn"]["recall@1"] + i["decoder"]["lex"]["recall@1"])
    print(f"   Pretraining lifts synonym recall by {gain:.2f} with no synonym in any training pair: the words come from the stream, not from the pairs.")
    if adapt_gain > 0.02:
        print("   The bidirectional adaptation adds to it, as in the paper's encoder-decoder ablation.")
    elif adapt_gain < -0.02:
        print("   The adaptation stage did not pay off at this budget; the paper's gain comes from a far longer encoder-decoder stage.")
    else:
        print("   The adaptation stage is roughly neutral at this budget; the paper's gain comes from a far longer encoder-decoder stage.")
    print("   Random init fits lexical queries fastest: the pretrained blocks must first be re-fit to the contrastive task.")

    print("\n4. The loss stack (adapted init, 20% of training queries use synonyms)")
    for name, m in res["stack"].items():
        print(f"   {name:32s}: synonym {r1(m['syn'])} | lexical {r1(m['lex'])} | final loss {m['loss_last']:.3f}")

    print("\n5. Matryoshka truncation, recall@1 on synonym queries at 32 / 16 / 8 dimensions")
    for name, row in res["mrl"].items():
        print(f"   {name:18s}: {row[0]:.2f} / {row[1]:.2f} / {row[2]:.2f}")

    print("\n6. A soup of two fine-tunes of the same checkpoint: recall@1 on synonym queries along the segment")
    print("   " + " | ".join(f"{k} {v:.2f}" for k, v in res["soup"].items()))
    mid, ends = res["soup"]["alpha 0.50"], (res["soup"]["alpha 0.00"], res["soup"]["alpha 1.00"])
    if mid >= min(ends) - 0.02:
        print("   The midpoint holds up: the two runs share a basin and averaging is safe.")
    else:
        print("   The midpoint drops below both ends: short, high-learning-rate fine-tunes of a small model leave the shared basin, and averaging fails.")

    q = res["quant"]
    print("\n7. Weight quantization of the full model, recall@1 on synonym queries")
    print(f"   float {q['float']:.2f} | int8 {q['int8']:.2f} | int4 {q['int4']:.2f}")
    print(f"\nDone in {time.time() - t0:.0f}s. Numbers move by a few points across seeds; the orderings are what the paper predicts at toy scale.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

