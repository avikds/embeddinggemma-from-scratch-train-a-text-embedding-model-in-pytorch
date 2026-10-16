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

