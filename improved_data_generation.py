import json
import random
import os
import string

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
TRAIN_SIZE = 1000
DEV_SIZE = 200
TEST_SIZE = 150       # you can change this freely
OUTPUT_DIR = "./data_advanced"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# STATIC DICTIONARIES
# ----------------------------------------------------
FILLERS = ["uh", "um", "you know", "like", "hmm", "err", "ah", "basically", "actually"]
HOMOPHONES = {
    "to": ["two", "too"],
    "for": ["four"],
    "eight": ["ate"],
    "won": ["one"],
    "there": ["their", "theyre"],
    "your": ["you're"],
}
ASR_MERGE_SPLIT = [
    ("credit card", "creditcard"),
    ("phone number", "phonenumber"),
    ("email id", "emailid"),
    ("at gmail dot com", "atgmaildotcom"),
]

MISSPELLINGS = {
    "gmail": ["g male", "gee mail", "g mail"],
    "outlook": ["out look", "out luk"],
    "san francisco": ["sanfransisco", "san fransisco"],
    "chennai": ["chen nai", "chenn i"],
}

EMAIL_DOMAINS = ["gmail", "yahoo", "outlook", "hotmail"]
CITY_LIST = ["chennai", "bangalore", "delhi", "mumbai", "new york", "san francisco", "london"]
LOCATION_LIST = ["fifth avenue", "main street", "church street", "mg road", "brigade road"]
NAMES = ["john doe", "alex kumar", "sarah thomas", "deepa sharma", "raj patel", "maria fernandes"]

MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

DIGIT_WORDS = {
    "0": ["zero", "oh"],
    "1": ["one", "won"],
    "2": ["two", "to", "too"],
    "3": ["three"],
    "4": ["four", "for"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight", "ate"],
    "9": ["nine"],
}

# ----------------------------------------------------
# ADVANCED NOISE FUNCTIONS
# ----------------------------------------------------

def random_letter_noise(word):
    """Randomly delete/insert/substitute characters."""
    if len(word) <= 3:
        return word

    if random.random() < 0.1:  # deletion
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx+1:]

    if random.random() < 0.1:  # substitution
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + random.choice(string.ascii_lowercase) + word[idx+1:]

    if random.random() < 0.05:  # insertion
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + random.choice(string.ascii_lowercase) + word[idx:]

    return word


def apply_homophones(text):
    words = text.split()
    out = []
    for w in words:
        if w in HOMOPHONES and random.random() < 0.3:
            out.append(random.choice(HOMOPHONES[w]))
        else:
            out.append(w)
    return " ".join(out)


def apply_merge_split_noise(text):
    if random.random() < 0.2:
        src, tgt = random.choice(ASR_MERGE_SPLIT)
        return text.replace(src, tgt)
    return text


def apply_misspellings(text):
    for k, vlist in MISSPELLINGS.items():
        if k in text and random.random() < 0.2:
            text = text.replace(k, random.choice(vlist))
    return text


def random_filler_noise(text):
    if random.random() < 0.3:
        filler = random.choice(FILLERS)
        insert_pos = random.choice(["start", "end", "middle"])
        words = text.split()

        if insert_pos == "start":
            return filler + " " + text
        elif insert_pos == "end":
            return text + " " + filler
        else:
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, filler)
            return " ".join(words)

    return text


def advanced_noise_pipeline(text):
    text = apply_homophones(text)
    text = apply_merge_split_noise(text)
    text = apply_misspellings(text)

    noisy_words = []
    for w in text.split():
        if random.random() < 0.15:
            noisy_words.append(random_letter_noise(w))
        else:
            noisy_words.append(w)
    text = " ".join(noisy_words)

    text = random_filler_noise(text)
    return text

# ----------------------------------------------------
# ENTITY GENERATORS
# ----------------------------------------------------

def spelled_out_digits(n_digits=16):
    digits = [str(random.randint(0, 9)) for _ in range(n_digits)]
    return " ".join(random.choice(DIGIT_WORDS[d]) for d in digits)


def random_email():
    name = random.choice(["john", "alex", "sarah", "deepa", "raj", "maria"])
    lname = random.choice(["doe", "kumar", "patel", "sharma", "thomas"])
    domain = random.choice(EMAIL_DOMAINS)
    tld = random.choice(["com", "co", "org", "in"])
    return f"{name} dot {lname} at {domain} dot {tld}"


def random_date():
    day = random.randint(1, 28)
    day_map = {
        1:"first",2:"second",3:"third",4:"fourth",5:"fifth",6:"sixth",7:"seventh",8:"eighth",9:"ninth",
        10:"tenth",11:"eleventh",12:"twelfth",13:"thirteenth",14:"fourteenth",15:"fifteenth",
        16:"sixteenth",17:"seventeenth",18:"eighteenth",19:"nineteenth",20:"twentieth",
        21:"twenty first",22:"twenty second",23:"twenty third",24:"twenty fourth",
        25:"twenty fifth",26:"twenty sixth",27:"twenty seventh",28:"twenty eighth"
    }
    month = random.choice(MONTHS)
    year = random.choice(["twenty nineteen", "twenty twenty", "twenty twenty one"])
    return f"{day_map[day]} {month} {year}"


def generate_entities():
    return {
        "CREDIT_CARD": spelled_out_digits(random.choice([14, 15, 16])),
        "PHONE": spelled_out_digits(10),
        "EMAIL": random_email(),
        "PERSON_NAME": random.choice(NAMES),
        "DATE": random_date(),
        "CITY": random.choice(CITY_LIST),
        "LOCATION": random.choice(LOCATION_LIST)
    }

# ----------------------------------------------------
# TEMPLATES
# ----------------------------------------------------

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
    "i met {PERSON_NAME} in {CITY} on {DATE}",
    "reach me at {PHONE} or email me at {EMAIL}",
    "i need to update my card it is {CREDIT_CARD}",
]

# ----------------------------------------------------
# EXAMPLE GENERATORS (LABELED + UNLABELED)
# ----------------------------------------------------

def make_labeled_example(idx):
    template = random.choice(TEMPLATES)
    ent_vals = generate_entities()

    # Insert entities & compute offsets BEFORE noise
    text = template
    entities = []

    for ent_type, ent in ent_vals.items():
        placeholder = "{" + ent_type + "}"
        if placeholder in text:
            start = text.index(placeholder)
            text = text.replace(placeholder, ent, 1)
            end = start + len(ent)
            entities.append({"start": start, "end": end, "label": ent_type})

    # Apply noise AFTER offsets
    noisy_text = advanced_noise_pipeline(text).lower().strip()

    return {
        "id": f"utt_{idx:04d}",
        "text": noisy_text,
        "entities": entities
    }


def make_unlabeled_example(idx):
    """Same distribution, but remove labels."""
    ex = make_labeled_example(idx)
    return {
        "id": ex["id"],
        "text": ex["text"],
        "entities": []   # empty list for test set
    }

# ----------------------------------------------------
# WRITE DATASETS
# ----------------------------------------------------

def write_jsonl(size, filename, labeled=True):
    with open(filename, "w") as f:
        for i in range(size):
            if labeled:
                ex = make_labeled_example(i)
            else:
                ex = make_unlabeled_example(i)
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {size} → {filename}")


# TRAIN + DEV + TEST
write_jsonl(TRAIN_SIZE, os.path.join(OUTPUT_DIR, "train.jsonl"), labeled=True)
write_jsonl(DEV_SIZE, os.path.join(OUTPUT_DIR, "dev.jsonl"), labeled=True)
write_jsonl(TEST_SIZE, os.path.join(OUTPUT_DIR, "test.jsonl"), labeled=False)
