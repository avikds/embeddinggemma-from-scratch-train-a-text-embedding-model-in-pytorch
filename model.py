"""
EmbeddingGemma from Scratch: Train a Text Embedding Model in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_world
TOPIC_WORDS = {
    "cooking": [
        ("simmer", "stew"),
        ("saute", "fry"),
        ("whisk", "beat"),
        ("grate", "shred"),
        ("mince", "chop"),
        ("sear", "brown"),
    ],
    "astronomy": [
        ("telescope", "scope"),
        ("planet", "world"),
        ("orbit", "circuit"),
        ("lunar", "moonlike"),
        ("stellar", "cosmic"),
        ("eclipse", "occultation"),
    ],
    "gardening": [
        ("prune", "trim"),
        ("sow", "plant"),
        ("irrigate", "water"),
        ("seedling", "sprout"),
        ("cultivate", "tend"),
        ("fertilize", "enrich"),
    ],
    "chess": [
        ("blunder", "mistake"),
        ("checkmate", "mate"),
        ("stalemate", "draw"),
        ("gambit", "sacrifice"),
        ("knight", "horse"),
        ("rook", "castle"),
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
        ("kiln", "oven"),
        ("clay", "earth"),
        ("glaze", "coating"),
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
        ("erosion", "wear"),
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

# Exactly 60 unique entities.
# "alder" is first and "ivy" is last for cyclic pairing.
ENTITIES = [
    "alder",
    "birch",
    "cedar",
    "dogwood",
    "elm",
    "fir",
    "ginkgo",
    "hazel",
    "juniper",
    "kudzu",
    "lilac",
    "maple",
    "nettle",
    "oak",
    "petunia",
    "rose",
    "spruce",
    "tulip",
    "willow",
    "yarrow",
    "zinnia",
    "acacia",
    "bamboo",
    "clover",
    "dahlia",
    "elder",
    "fern",
    "gardenia",
    "heather",
    "iris",
    "jasmine",
    "lotus",
    "magnolia",
    "narcissus",
    "orchid",
    "peony",
    "quince",
    "rosemary",
    "saffron",
    "thyme",
    "verbena",
    "wisteria",
    "yucca",
    "amethyst",
    "azure",
    "beige",
    "coral",
    "crimson",
    "indigo",
    "ivory",
    "jade",
    "khaki",
    "mauve",
    "ochre",
    "olive",
    "sienna",
    "teal",
    "umber",
    "vermilion",
    "ivy",
]

FILLER = [
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "with",
    "from",
    "into",
    "near",
    "about",
    "for",
]

SPECIAL = ["[PAD]", "[Q]", "[D]", "[MASK]"]


def make_world():
    # Preserve the topic order defined in TOPIC_WORDS.
    topics = list(TOPIC_WORDS.keys())

    # Extract the six document words for each topic.
    words = {
        topic: [word for word, synonym in pairs]
        for topic, pairs in TOPIC_WORDS.items()
    }

    # Map each document word to its unique synonym.
    synonyms = {
        word: synonym
        for pairs in TOPIC_WORDS.values()
        for word, synonym in pairs
    }

    # Pair every entity with the next entity cyclically.
    # The final entity, ivy, is paired with the first, alder.
    partners = {
        entity: ENTITIES[(i + 1) % len(ENTITIES)]
        for i, entity in enumerate(ENTITIES)
    }

    # Validate topic and vocabulary counts.
    assert len(topics) == 12
    assert all(len(TOPIC_WORDS[t]) == 6 for t in topics)
    assert len(words) == 12
    assert len(synonyms) == 72
    assert len(ENTITIES) == 60
    assert len(FILLER) == 12
    assert len(SPECIAL) == 4

    # Validate the number of topic words and synonyms.
    topic_terms = [
        term
        for pairs in TOPIC_WORDS.values()
        for pair in pairs
        for term in pair
    ]
    assert len(topic_terms) == 144

    # All document words, synonyms, entities, and filler
    # words must be pairwise distinct.
    all_tokens = topic_terms + ENTITIES + FILLER
    assert len(all_tokens) == 216
    assert len(set(all_tokens)) == 216

    # Confirm the required cyclic entity relationships.
    assert partners["alder"] == "birch"
    assert partners["ivy"] == "alder"

    return {
        "topics": topics,
        "words": words,
        "synonyms": synonyms,
        "entities": ENTITIES,
        "filler": FILLER,
        "partners": partners,
    }

