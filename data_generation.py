import json
import random
import os

# -----------------------------------------
# CONFIG
# -----------------------------------------
TRAIN_SIZE = 1000
DEV_SIZE = 200
OUTPUT_DIR = "./data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------
# UTILITIES
# -----------------------------------------

FILLERS = ["uh", "um", "you know", "like", "hmm", "err", "ah"]
ASR_DIGIT_NOISE = {
    "2": ["to", "too", "two"],
    "4": ["for", "four"],
    "8": ["ate", "eight"]
}

EMAIL_DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "protonmail"]
CITY_LIST = ["chennai", "bangalore", "delhi", "mumbai", "new york", "san francisco", "london"]
LOCATION_LIST = ["fifth avenue", "main street", "church street", "mg road", "brigade road"]
NAMES = ["john doe", "alex kumar", "sarah thomas", "deepa sharma", "raj patel", "maria fernandes"]
MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

# -----------------------------------------
# HELPERS
# -----------------------------------------

def add_filler_noise(text):
    if random.random() < 0.3:
        filler = random.choice(FILLERS)
        positions = ["start", "middle", "end"]
        pos = random.choice(positions)

        if pos == "start":
            return f"{filler} {text}"
        elif pos == "middle":
            words = text.split()
            idx = random.randint(0, len(words)-1)
            words.insert(idx, filler)
            return " ".join(words)
        else:
            return f"{text} {filler}"

    return text


def add_digit_asr_noise(word):
    return ASR_DIGIT_NOISE.get(word, [word])[0]


def spelled_out_digits(n_digits=16):
    digits = [str(random.randint(0,9)) for _ in range(n_digits)]
    noisy_digits = [add_digit_asr_noise(d) for d in digits]
    return " ".join(noisy_digits)


def random_name():
    return random.choice(NAMES)


def random_city():
    return random.choice(CITY_LIST)


def random_location():
    return random.choice(LOCATION_LIST)


def random_date():
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    year = random.choice(["two thousand nineteen", "twenty twenty", "twenty twenty one", "twenty eighteen"])

    day_words = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
        7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
        12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
        20: "twentieth", 21: "twenty first", 22: "twenty second",
        23: "twenty third", 24: "twenty fourth", 25: "twenty fifth",
        26: "twenty sixth", 27: "twenty seventh", 28: "twenty eighth"
    }

    return f"{day_words[day]} {month} {year}"


def random_email():
    name = random.choice(["john", "alex", "sarah", "deepa", "raj", "maria"])
    lname = random.choice(["doe", "kumar", "patel", "sharma", "thomas", "fernandes"])
    domain = random.choice(EMAIL_DOMAINS)
    tld = random.choice(["com", "co", "co dot in", "org"])

    email = f"{name} dot {lname} at {domain} dot {tld}"
    return email


# -----------------------------------------
# TEMPLATES
# -----------------------------------------

TEMPLATES = [
    "my email is {EMAIL}",
    "my phone number is {PHONE}",
    "my credit card number is {CREDIT_CARD}",
    "i spoke to {PERSON_NAME} yesterday",
    "the meeting is on {DATE}",
    "please deliver to {CITY}",
    "the location is {LOCATION}",
    "{PERSON_NAME} contacted me on {DATE}",
    "my phone is {PHONE} and email is {EMAIL}",
    "ship the item to {LOCATION} in {CITY}",
    "i met {PERSON_NAME} in {CITY} on {DATE}"
]

# -----------------------------------------
# ENTITY GENERATION
# -----------------------------------------

def generate_entities():
    return {
        "CREDIT_CARD": spelled_out_digits(random.choice([14, 15, 16])),
        "PHONE": spelled_out_digits(10),
        "EMAIL": random_email(),
        "PERSON_NAME": random_name(),
        "DATE": random_date(),
        "CITY": random_city(),
        "LOCATION": random_location()
    }


# -----------------------------------------
# MAIN EXAMPLE GENERATOR
# -----------------------------------------

def make_example(idx):
    template = random.choice(TEMPLATES)
    entity_values = generate_entities()

    text = template
    entities = []

    # Insert entity strings and track offsets
    for ent_type, ent_value in entity_values.items():
        if "{" + ent_type + "}" in text:
            start = text.index("{" + ent_type + "}")
            text = text.replace("{" + ent_type + "}", ent_value, 1)
            end = start + len(ent_value)

            entities.append({
                "start": start,
                "end": end,
                "label": ent_type
            })

    # Add STT noise
    text = add_filler_noise(text)
    text = text.lower().strip()

    return {
        "id": f"utt_{idx:04d}",
        "text": text,
        "entities": entities
    }


# -----------------------------------------
# GENERATE DATASETS
# -----------------------------------------

def generate_dataset(size, filename):
    with open(filename, "w") as f:
        for i in range(size):
            ex = make_example(i)
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {size} examples → {filename}")


# -----------------------------------------
# RUN
# -----------------------------------------

generate_dataset(TRAIN_SIZE, os.path.join(OUTPUT_DIR, "train.jsonl"))
generate_dataset(DEV_SIZE, os.path.join(OUTPUT_DIR, "dev.jsonl"))
