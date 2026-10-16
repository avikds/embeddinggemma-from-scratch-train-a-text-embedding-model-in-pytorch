"""
EmbeddingGemma from Scratch: Train a Text Embedding Model in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_world
TOPIC_WORDS = {
    "cooking": [
        ("stove", "range"),
        ("recipe", "formula"),
        ("chef", "cook"),
        ("simmer", "stew"),
        ("spice", "seasoning"),
        ("skillet", "pan"),
    ],
    "astronomy": [
        ("telescope", "scope"),
        ("planet", "orb"),
        ("orbit", "trajectory"),
        ("lunar", "selenic"),
        ("stellar", "celestial"),
        ("eclipse", "occultation"),
    ],
    "gardening": [
        ("prune", "trim"),
        ("sow", "seed"),
        ("irrigate", "water"),
        ("seedling", "sprout"),
        ("cultivate", "till"),
        ("fertilize", "enrich"),
    ],
    "chess": [
        ("knight", "horse"),
        ("king", "monarch"),
        ("gambit", "sacrifice"),
        ("rook", "castle"),
        ("bishop", "prelate"),
        ("endgame", "finale"),
    ],
    "cycling": [
        ("bicycle", "bike"),
        ("cyclist", "rider"),
        ("sprint", "dash"),
        ("helmet", "headgear"),
        ("peloton", "pack"),
        ("route", "course"),
    ],
    "pottery": [
        ("kiln", "furnace"),
        ("clay", "loam"),
        ("glaze", "enamel"),
        ("potter", "ceramist"),
        ("vase", "urn"),
        ("wheel", "turntable"),
    ],
    "sailing": [
        ("vessel", "ship"),
        ("sailboat", "yacht"),
        ("helm", "tiller"),
        ("mast", "spar"),
        ("anchor", "mooring"),
        ("voyage", "passage"),
    ],
    "jazz": [
        ("improvise", "adlib"),
        ("saxophone", "horn"),
        ("swing", "groove"),
        ("ensemble", "band"),
        ("syncopation", "offbeat"),
        ("melody", "tune"),
    ],
    "geology": [
        ("stratum", "layer"),
        ("igneous", "volcanic"),
        ("fossil", "remnant"),
        ("erosion", "abrasion"),
        ("quartz", "crystal"),
        ("magma", "lava"),
    ],
    "baking": [
        ("dough", "paste"),
        ("yeast", "leavener"),
        ("proof", "rise"),
        ("batter", "mixture"),
        ("crumb", "interior"),
        ("icing", "frosting"),
    ],
    "photography": [
        ("shutter", "release"),
        ("aperture", "opening"),
        ("exposure", "brightness"),
        ("portrait", "likeness"),
        ("snapshot", "photo"),
        ("focus", "sharpness"),
    ],
    "climbing": [
        ("belay", "secure"),
        ("crag", "cliff"),
        ("summit", "peak"),
        ("ascent", "climb"),
        ("carabiner", "clip"),
        ("boulder", "rock"),
    ],
}

ENTITIES = [
    "alder", "birch", "cedar", "dogwood", "elm", "fern",
    "ginkgo", "hazel", "juniper", "kudzu", "lilac", "maple",
    "nettle", "oak", "petunia", "rose", "spruce", "tulip",
    "willow", "yarrow", "zinnia", "acacia", "bamboo", "clover",
    "dahlia", "gorse", "fir", "gardenia", "heather", "iris",
    "jasmine", "lotus", "magnolia", "narcissus", "orchid",
    "peony", "quince", "rosemary", "saffron", "thyme",
    "verbena", "wisteria", "yucca", "amethyst", "azure",
    "beige", "coral", "crimson", "indigo", "ivory", "jade",
    "khaki", "mauve", "ochre", "olive", "sienna", "teal",
    "amber", "vermilion", "ivy",
]

FILLER = [
    "the", "and", "or", "but", "if", "notes",
    "guide", "report", "into", "of", "near", "for",
]

SPECIAL = ["[PAD]", "[Q]", "[D]", "[MASK]"]


def make_world():
    topics = list(TOPIC_WORDS.keys())

    words = {
        topic: [word for word, synonym in pairs]
        for topic, pairs in TOPIC_WORDS.items()
    }

    synonyms = {
        word: synonym
        for pairs in TOPIC_WORDS.values()
        for word, synonym in pairs
    }

    partners = {
        entity: ENTITIES[(i + 1) % len(ENTITIES)]
        for i, entity in enumerate(ENTITIES)
    }

    topic_terms = [
        term
        for pairs in TOPIC_WORDS.values()
        for pair in pairs
        for term in pair
    ]

    all_tokens = topic_terms + ENTITIES + FILLER

    assert len(topics) == 12
    assert all(len(TOPIC_WORDS[t]) == 6 for t in topics)
    assert len(topic_terms) == 144
    assert len(synonyms) == 72
    assert len(ENTITIES) == 60
    assert len(FILLER) == 12
    assert len(all_tokens) == 216
    assert len(set(all_tokens)) == 216

    return {
        "topics": topics,
        "words": words,
        "synonyms": synonyms,
        "entities": ENTITIES,
        "filler": FILLER,
        "partners": partners,
    }

# Step 2 - make_document
import random

def make_document(world, rng, topic, entities, syn_rate=0.0):
    # Select two distinct topic words for the title.
    title = rng.sample(world["words"][topic], 2)

    # Construct the body from the title, three additional
    # topic-word draws, the entities, and two filler words.
    body = (
        title.copy()
        + [
            rng.choice(world["words"][topic])
            for _ in range(3)
        ]
        + list(entities)
        + rng.sample(world["filler"], 2)
    )

    # Shuffle only the body; the title stays at the beginning.
    rng.shuffle(body)

    # Combine title and body to create the token sequence.
    tokens = title + body

    # Replace topic words with synonyms probabilistically.
    # Consume a random draw for every topic-word occurrence.
    for i, token in enumerate(tokens):
        if token in world["synonyms"]:
            if rng.random() < syn_rate:
                tokens[i] = world["synonyms"][token]

    return {
        "topic": topic,
        "entities": list(entities),
        "title": tokens[:2],
        "tokens": tokens,
    }


def make_corpus(world, docs_per_topic, rng):
    documents = []

    for topic in world["topics"]:
        # Independently shuffle the entity list for each topic.
        ents = list(world["entities"])
        rng.shuffle(ents)

        for k in range(docs_per_topic):
            # Assign two distinct entities to each document.
            doc_entities = sorted(ents[2 * k:2 * k + 2])

            document = make_document(
                world,
                rng,
                topic,
                doc_entities,
                syn_rate=0.0,
            )

            # Give every document a consecutive integer ID.
            document["id"] = len(documents)
            documents.append(document)

    return documents


def pretraining_stream(
    world,
    rng,
    n,
    syn_rate=0.3,
    partner_rate=0.8,
):
    documents = []

    for _ in range(n):
        # Draw the topic and first entity.
        topic = rng.choice(world["topics"])
        entity = rng.choice(world["entities"])

        # Decide whether to use the first entity's partner.
        use_partner = rng.random() < partner_rate

        if use_partner:
            second_entity = world["partners"][entity]
        else:
            # Choose another entity, excluding the first entity.
            other_entities = [
                candidate
                for candidate in world["entities"]
                if candidate != entity
            ]
            second_entity = rng.choice(other_entities)

        document = make_document(
            world,
            rng,
            topic,
            [entity, second_entity],
            syn_rate=syn_rate,
        )

        documents.append(document)

    return documents

# Step 3 - make_query
def make_query(doc, world, rng, syn_rate):
    # Collect the sorted, unique topic words appearing in the document.
    topic_words = sorted(
        set(doc["tokens"]) & set(world["words"][doc["topic"]])
    )

    # Draw one or two topic words.
    k = rng.randint(1, 2)
    selected = rng.sample(topic_words, min(k, len(topic_words)))

    # Replace selected topic words with synonyms probabilistically.
    query_words = []
    for word in selected:
        if rng.random() < syn_rate:
            query_words.append(world["synonyms"][word])
        else:
            query_words.append(word)

    # Add one entity from the document.
    query_words.append(rng.choice(doc["entities"]))

    # With probability 0.5, append one filler word.
    if rng.random() < 0.5:
        query_words.append(rng.choice(world["filler"]))

    # Shuffle the final query.
    rng.shuffle(query_words)

    return query_words


def make_pairs(corpus, world, rng, per_doc, syn_rate):
    pairs = []

    # Generate queries for each document in corpus order.
    for doc in corpus:
        for _ in range(per_doc):
            query_words = make_query(
                doc,
                world,
                rng,
                syn_rate,
            )
            pairs.append((query_words, doc["id"]))

    return pairs


def make_title_pairs(corpus):
    # Use each document's title as a query paired with its ID.
    return [
        (doc["title"][:], doc["id"])
        for doc in corpus
    ]


def mix_pairs(lexical, semantic, w_semantic, rng, n):
    # Work on copies so the input lists remain unchanged.
    lexical_pool = list(lexical)
    semantic_pool = list(semantic)

    # Shuffle both pools independently.
    rng.shuffle(lexical_pool)
    rng.shuffle(semantic_pool)

    # Determine the number of semantic pairs.
    n_semantic = round(w_semantic * n)
    n_lexical = n - n_semantic

    # Select semantic pairs first, then lexical pairs.
    selected_semantic = semantic_pool[:n_semantic]
    selected_lexical = lexical_pool[:n_lexical]

    return selected_semantic + selected_lexical

# Step 4 - Vocab
class Vocab:
    def __init__(self, world):
        # Special tokens must occupy the first four positions.
        self.itos = list(SPECIAL)

        # Add each topic word followed immediately by its synonym.
        for topic in world["topics"]:
            for word in world["words"][topic]:
                self.itos.append(word)
                self.itos.append(world["synonyms"][word])

        # Add entities and filler words in their defined order.
        self.itos.extend(world["entities"])
        self.itos.extend(world["filler"])

        # Build the token-to-index lookup table.
        self.stoi = {
            token: index
            for index, token in enumerate(self.itos)
        }

        # Store the special-token IDs.
        self.pad = self.stoi["[PAD]"]
        self.q = self.stoi["[Q]"]
        self.d = self.stoi["[D]"]
        self.mask_id = self.stoi["[MASK]"]

    def __len__(self):
        return len(self.itos)

    def encode(self, words, prefix):
        # Query sequences receive [Q]; all other sequences receive [D].
        prefix_id = self.q if prefix == "query" else self.d

        # Return the prefix ID followed by the word IDs.
        return [
            prefix_id,
            *[self.stoi[word] for word in words],
        ]

    def collate(self, seqs, max_len=16):
        import torch

        # T is the longest sequence, capped at max_len.
        T = min(max((len(seq) for seq in seqs), default=0), max_len)

        # Initialize padded token IDs and attention masks.
        ids = torch.full(
            (len(seqs), T),
            self.pad,
            dtype=torch.long,
        )
        mask = torch.zeros(
            (len(seqs), T),
            dtype=torch.long,
        )

        # Truncate each sequence and mark real tokens with 1.
        for i, seq in enumerate(seqs):
            seq = seq[:T]
            if seq:
                ids[i, :len(seq)] = torch.tensor(
                    seq,
                    dtype=torch.long,
                )
                mask[i, :len(seq)] = 1

        return ids, mask

# Step 5 - bm25_rank
import math
from collections import Counter


def bm25_index(corpus):
    # Number of documents.
    n = len(corpus)

    # Token counts for each document.
    tf = [Counter(doc["tokens"]) for doc in corpus]

    # Document frequency: number of documents containing each token.
    df = Counter()
    for doc_tf in tf:
        for token in doc_tf:
            df[token] += 1

    # Average document length.
    avg_len = (
        sum(sum(doc_tf.values()) for doc_tf in tf) / n
        if n else 0.0
    )

    return {
        "n": n,
        "df": dict(df),
        "avg_len": avg_len,
        "tf": tf,
    }


def bm25_rank(query_words, corpus, index=None, k1=1.5, b=0.75):
    # Build the index if one was not supplied.
    if index is None:
        index = bm25_index(corpus)

    n = index["n"]
    df = index["df"]
    avg_len = index["avg_len"]
    tf = index["tf"]

    scores = []

    for doc_index, doc in enumerate(corpus):
        doc_tf = tf[doc_index]
        doc_len = sum(doc_tf.values())
        score = 0.0

        for token in query_words:
            frequency = doc_tf.get(token, 0)

            # Only score query terms occurring in this document.
            if frequency == 0:
                continue

            document_frequency = df.get(token, 0)

            # BM25 inverse document frequency.
            idf = math.log(
                1.0
                + (n - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            # BM25 term-frequency normalization.
            denominator = frequency + k1 * (
                1.0 - b
                + b * doc_len / avg_len
                if avg_len > 0
                else 1.0
            )

            score += idf * (
                frequency * (k1 + 1.0) / denominator
            )

        scores.append((score, doc_index))

    # Higher scores first; lower document indices break ties.
    scores.sort(key=lambda item: (-item[0], item[1]))

    return [doc_index for score, doc_index in scores]


def ranking_metrics(ranked_lists, truths, k=10):
    # Calculate recall@1, recall@5, and nDCG@10.
    recall1 = 0.0
    recall5 = 0.0
    ndcg10 = 0.0

    count = len(truths)

    if count == 0:
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "ndcg@10": 0.0,
        }

    for ranked, truth in zip(ranked_lists, truths):
        # Recall at rank 1.
        if ranked and ranked[0] == truth:
            recall1 += 1.0

        # Recall within the first five results.
        if truth in ranked[:5]:
            recall5 += 1.0

        # nDCG for one relevant document.
        if truth in ranked[:k]:
            rank = ranked.index(truth) + 1
            ndcg10 += 1.0 / math.log2(rank + 1)

    return {
        "recall@1": round(recall1 / count, 4),
        "recall@5": round(recall5 / count, 4),
        "ndcg@10": round(ndcg10 / count, 4),
    }


def evaluate_ranker(rank_fn, pairs):
    ranked_lists = []
    truths = []

    # Each pair contains (query_words, document_id).
    for query_words, doc_id in pairs:
        ranked_lists.append(rank_fn(query_words))
        truths.append(doc_id)

    return ranking_metrics(ranked_lists, truths)

# Step 6 - causal_allowed
def causal_allowed(T):
    # Create a lower-triangular boolean attention mask.
    # A query at position i may attend to keys at positions j <= i.
    return torch.tril(
        torch.ones((T, T), dtype=torch.bool)
    )


def padding_allowed(mask):
    # Input mask has shape (B, T), with 1 for real tokens
    # and 0 for padding tokens.
    # Reshape to (B, 1, 1, T) for broadcasting across
    # attention heads and query positions.
    return mask[:, None, None, :].bool()

# Step 7 - TransformerBlock
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()

        # Learnable scale initialized to zero.
        self.scale = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        # Calculate the mean square in float32 for numerical stability.
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)

        # RMS normalization, followed by the learnable scale.
        normalized = x_float * torch.rsqrt(mean_square + self.eps)
        normalized = normalized * (1.0 + self.scale)

        # Preserve the input tensor's dtype.
        return normalized.to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()

        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")

        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.d_ff = d_ff

        # Pre-normalization for attention.
        self.norm1 = RMSNorm(d)

        # Combined query, key, and value projection.
        self.qkv = nn.Linear(d, 3 * d, bias=False)

        # Attention output projection.
        self.out = nn.Linear(d, d, bias=False)

        # Pre-normalization for the feed-forward network.
        self.norm2 = RMSNorm(d)

        # Gated feed-forward projections.
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, allowed):
        B, T, D = x.shape

        # -----------------------------------------------
        # 1. Multi-head self-attention with pre-norm.
        # -----------------------------------------------
        h = self.norm1(x)

        # Project to Q, K, V and separate the attention heads.
        qkv = self.qkv(h)
        qkv = qkv.reshape(
            B, T, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        # Scaled dot-product attention scores.
        scores = (
            q @ k.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        # Prevent attention to disallowed key positions.
        scores = scores.masked_fill(
            ~allowed.bool(),
            float("-inf"),
        )

        # Normalize attention scores and combine value vectors.
        attn = torch.softmax(scores, dim=-1)
        h = attn @ v

        # Merge attention heads back into the model dimension.
        h = h.transpose(1, 2).contiguous().reshape(B, T, D)

        # Attention output projection and residual connection.
        x = x + self.out(h)

        # -----------------------------------------------
        # 2. Gated GELU feed-forward network with pre-norm.
        # -----------------------------------------------
        h = self.norm2(x)

        # Apply GELU to the gate branch and multiply elementwise
        # by the up-projection branch.
        gated = torch.nn.functional.gelu(self.gate(h))
        up = self.up(h)

        # Project back to the model dimension and add residual.
        x = x + self.down(gated * up)

        return x

# Step 8 - TinyLM
class TinyLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_len=16,
    ):
        super().__init__()

        self.d = d
        self.max_len = max_len

        # Token embeddings, initialized with a small normal distribution.
        self.tok = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)

        # Learned positional embeddings.
        self.pos = nn.Embedding(max_len, d)

        # Stack of Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Final RMS normalization.
        self.norm = RMSNorm(d)

    def hidden(self, ids, allowed):
        # ids has shape (B, T).
        B, T = ids.shape

        # Embed tokens and scale by sqrt(d).
        x = self.tok(ids) * (self.d ** 0.5)

        # Add positional embeddings.
        positions = torch.arange(T, device=ids.device)
        x = x + self.pos(positions).unsqueeze(0)

        # Apply every Transformer block with the supplied mask.
        for block in self.blocks:
            x = block(x, allowed)

        # Return normalized hidden states of shape (B, T, d).
        return self.norm(x)

    def forward(self, ids):
        # Construct the causal attention mask for the sequence.
        T = ids.shape[1]
        allowed = causal_allowed(T).to(device=ids.device)

        # Compute causal hidden states.
        h = self.hidden(ids, allowed)

        # Tie the output projection weights to the token embedding.
        # Output shape: (B, T, vocab_size).
        logits = h @ self.tok.weight.T

        return logits

    def lm_loss(self, ids, mask):
        # Predict each token from the preceding tokens.
        # Input: ids[:, :-1]; targets: ids[:, 1:].
        inputs = ids[:, :-1]
        targets = ids[:, 1:]

        logits = self.forward(inputs)

        # Only positions whose target tokens are real tokens
        # contribute to the next-token prediction loss.
        valid = mask[:, 1:].to(
            device=logits.device,
            dtype=torch.bool,
        )

        # Compute per-token cross-entropy without reduction.
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)

        # Average over valid target positions only.
        valid_float = valid.to(dtype=losses.dtype)
        loss_sum = (losses * valid_float).sum()
        num_valid = valid_float.sum().clamp_min(1.0)

        return loss_sum / num_valid

# Step 9 - pretrain_lm
def pretrain_lm(
    model,
    world,
    vocab,
    steps,
    rng,
    batch_size=64,
    lr=5e-4,
    weight_decay=0.1,
    syn_rate=0.3,
):
    # AdamW optimizer with the specified learning rate and weight decay.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Keep track of the training loss at every optimization step.
    losses = []

    # Determine the device on which the model resides.
    device = next(model.parameters()).device

    model.train()

    for _ in range(steps):
        # Generate a fresh batch of synthetic pretraining documents.
        documents = pretraining_stream(
            world,
            rng,
            batch_size,
            syn_rate=syn_rate,
        )

        # Encode documents using the document task prefix.
        sequences = [
            vocab.encode(doc["tokens"], "doc")
            for doc in documents
        ]

        # Pad the sequences and construct the attention mask.
        ids, mask = vocab.collate(sequences)

        ids = ids.to(device)
        mask = mask.to(device)

        # Compute the next-token prediction loss.
        optimizer.zero_grad()
        loss = model.lm_loss(ids, mask)

        # Backpropagate and update model parameters.
        loss.backward()
        optimizer.step()

        # Store each loss as a Python float.
        losses.append(float(loss.detach().cpu().item()))

    return losses


def synonym_cosine(model, vocab, world):
    # Compare each topic word's token embedding with its synonym.
    similarities = []

    # Embeddings do not need gradients for this evaluation.
    with torch.no_grad():
        embeddings = model.tok.weight

        for topic in world["topics"]:
            for word in world["words"][topic]:
                synonym = world["synonyms"][word]

                # Retrieve the vocabulary indices.
                word_id = vocab.stoi[word]
                synonym_id = vocab.stoi[synonym]

                # Calculate cosine similarity between the two embeddings.
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[word_id].unsqueeze(0),
                    embeddings[synonym_id].unsqueeze(0),
                    dim=-1,
                )

                similarities.append(similarity.item())

    # Return the mean similarity rounded to four decimal places.
    if not similarities:
        return 0.0

    return round(sum(similarities) / len(similarities), 4)

# Step 10 - adapt_encoder
def adapt_encoder(
    model,
    world,
    vocab,
    steps,
    rng,
    batch_size=64,
    lr=5e-4,
    mask_rate=0.3,
    syn_rate=0.3,
):
    # AdamW optimizer with the required weight decay.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
    )

    # Use a separate Torch generator for deterministic masking.
    generator = torch.Generator()
    generator.manual_seed(rng.randrange(1 << 30))

    # Store one floating-point loss per training step.
    losses = []

    # Use the same device as the model parameters.
    device = next(model.parameters()).device

    model.train()

    for _ in range(steps):
        # Generate a fresh batch of pretraining-stream documents.
        documents = pretraining_stream(
            world,
            rng,
            batch_size,
            syn_rate=syn_rate,
        )

        # Encode each document with the [D] prefix.
        sequences = [
            vocab.encode(doc["tokens"], "doc")
            for doc in documents
        ]

        # Pad sequences and obtain a mask for real tokens.
        ids, mask = vocab.collate(sequences)

        ids = ids.to(device)
        mask = mask.to(device)

        # Preserve original token IDs as reconstruction targets.
        targets = ids.clone()

        # Randomly select positions for masked denoising.
        # The generator uses a fixed seed derived from rng.
        random_values = torch.rand(
            ids.shape,
            generator=generator,
        ).to(device)

        corrupt = random_values < mask_rate

        # Only corrupt real tokens, never padding or the [D] prefix.
        corrupt = (
            corrupt
            & mask.bool()
            & (ids != vocab.d)
        )

        # Replace selected input tokens with [MASK].
        corrupted_ids = ids.clone()
        corrupted_ids[corrupt] = vocab.mask_id

        # Full bidirectional attention: keys are restricted only
        # by padding, rather than by a causal mask.
        allowed = padding_allowed(mask)

        # Compute hidden states for the corrupted sequences.
        optimizer.zero_grad()

        hidden = model.hidden(corrupted_ids, allowed)

        # Use the tied token embedding matrix for reconstruction.
        logits = hidden @ model.tok.weight.T

        # Calculate cross-entropy only at corrupted positions.
        if corrupt.any():
            loss = torch.nn.functional.cross_entropy(
                logits[corrupt],
                targets[corrupt],
            )

            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
        else:
            # A batch can exceptionally contain no selected tokens.
            # Record a finite loss and leave the model unchanged.
            losses.append(0.0)

    return losses

# Step 11 - Embedder
import copy

class Embedder(nn.Module):
    def __init__(self, lm, d_out=32):
        super().__init__()

        # Store the TinyLM backbone.
        self.lm = lm

        # Project pooled hidden states into the embedding space.
        self.proj = nn.Linear(lm.d, d_out, bias=False)

    @staticmethod
    def from_lm(lm, d_out=32):
        # Deep-copy the language model so the original remains unchanged.
        copied_lm = copy.deepcopy(lm)
        return Embedder(copied_lm, d_out=d_out)

    @staticmethod
    def fresh(vocab_size, d_out=32, **kw):
        # Create a new TinyLM from scratch and wrap it as an embedder.
        lm = TinyLM(vocab_size, **kw)
        return Embedder(lm, d_out=d_out)

    def forward(self, ids, mask):
        # ids and mask have shape (B, T).
        # Ensure the mask is on the same device as the input IDs.
        mask = mask.to(device=ids.device)

        # Use bidirectional attention, masking only padding positions.
        allowed = padding_allowed(mask)
        hidden = self.lm.hidden(ids, allowed)

        # Masked mean pooling over real tokens.
        # hidden: (B, T, d)
        # mask:   (B, T)
        weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)

        pooled = (hidden * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled / counts

        # Project pooled representations into the output space.
        embeddings = self.proj(pooled)

        # L2-normalize embeddings along the feature dimension.
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=-1,
        )

        return embeddings

    def embed_words(self, vocab, word_lists, prefix):
        # Encode each word list with the requested task prefix.
        sequences = [
            vocab.encode(words, prefix)
            for words in word_lists
        ]

        # Pad sequences and construct their attention masks.
        ids, mask = vocab.collate(sequences)

        # Place inputs on the same device as the model.
        device = next(self.parameters()).device
        ids = ids.to(device)
        mask = mask.to(device)

        # Return normalized embeddings with shape (B, d_out).
        return self.forward(ids, mask)

