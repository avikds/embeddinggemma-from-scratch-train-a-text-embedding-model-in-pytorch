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

