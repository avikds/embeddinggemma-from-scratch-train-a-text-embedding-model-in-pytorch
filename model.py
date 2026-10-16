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
    # Select two topic words for the document title.
    title = rng.sample(world["words"][topic], 2)

    # Build the body from the title, three additional topic words,
    # the supplied entities, and two distinct filler words.
    body = (
        title.copy()
        + [rng.choice(world["words"][topic]) for _ in range(3)]
        + list(entities)
        + rng.sample(world["filler"], 2)
    )

    # Shuffle the body while keeping the title prefix unchanged.
    rng.shuffle(body)

    # The title is repeated at the beginning of the body.
    # With two entities, the final token list contains 11 tokens.
    tokens = title + body

    # Replace topic words with their synonyms probabilistically.
    # Consume one rng.random() draw per topic-word token, in order.
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

    # Generate documents in the original topic order.
    for topic in world["topics"]:
        # Shuffle a fresh copy of the entity list for this topic.
        ents = list(world["entities"])
        rng.shuffle(ents)

        # Assign two distinct entities to each document.
        for k in range(docs_per_topic):
            doc_entities = sorted(ents[2 * k:2 * k + 2])

            # Corpus documents use the original document vocabulary.
            document = make_document(
                world,
                rng,
                topic,
                doc_entities,
                syn_rate=0.0,
            )

            # Assign sequential IDs starting from zero.
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
        # Required random draw order: topic, entity, partner decision.
        topic = rng.choice(world["topics"])
        entity = rng.choice(world["entities"])

        # Make the partner decision even when partner_rate is 0 or 1.
        use_partner = rng.random() < partner_rate

        if use_partner:
            second_entity = world["partners"][entity]
        else:
            # Select a different entity from the first one.
            other_entities = [
                candidate
                for candidate in world["entities"]
                if candidate != entity
            ]
            second_entity = rng.choice(other_entities)

        # Allow synonym replacement during pretraining.
        document = make_document(
            world,
            rng,
            topic,
            [entity, second_entity],
            syn_rate=syn_rate,
        )

        documents.append(document)

    return documents

