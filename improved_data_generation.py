import json
import random
import os
from faker import Faker
from tqdm import tqdm

# Initialize Faker
fake = Faker()

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
OUTPUT_DIR = "./data"
TRAIN_SIZE = 1000
DEV_SIZE = 200
TEST_SIZE = 200

# ----------------------------------------------------
# NOISE FUNCTIONS (Simulating STT Errors)
# ----------------------------------------------------
def noisy_digits(digits_str):
    """
    Converts '123' -> 'one two three', '1 2 3', or 'one 2 three'.
    STT often spells out digits or puts spaces between them.
    """
    digit_map = {
        '0': ['zero', 'oh', '0'],
        '1': ['one', '1'],
        '2': ['two', '2'],
        '3': ['three', '3'],
        '4': ['four', '4'],
        '5': ['five', '5'],
        '6': ['six', '6'],
        '7': ['seven', '7'],
        '8': ['eight', '8'],
        '9': ['nine', '9']
    }
    
    out = []
    for d in digits_str:
        if d in digit_map:
            # 80% chance of being a word, 10% digit char, 10% kept as is
            val = random.choices(digit_map[d], weights=[0.8, 0.1, 0.1], k=1)[0]
            out.append(val)
        else:
            out.append(d) 
    return " ".join(out)

def noisy_email(email):
    """
    john@gmail.com -> john at gmail dot com
    STT often misinterprets symbols.
    """
    # 90% chance to replace @ and . with words
    if random.random() < 0.9:
        return email.replace("@", " at ").replace(".", " dot ")
    return email

def noisy_date(date_obj):
    """
    2023-01-01 -> january first twenty twenty three
    """
    try:
        # Base format: January 01 2023
        d = date_obj.strftime("%B %d %Y") 
        
        # Add ordinal suffixes (st, nd, rd, th) roughly
        day_num = int(date_obj.strftime("%d"))
        suffix = "th"
        if day_num in [1, 21, 31]: suffix = "st"
        elif day_num in [2, 22]: suffix = "nd"
        elif day_num in [3, 23]: suffix = "rd"
        
        # Replace the number in the string
        d = d.replace(str(day_num).zfill(2), f"{day_num}{suffix}")
        return d.lower()
    except:
        return str(date_obj)

# ----------------------------------------------------
# GENERATOR LOGIC
# ----------------------------------------------------
def generate_sample(uid):
    """
    Builds the sentence chunk-by-chunk to ensure character offsets 
    are perfectly aligned with the generated text.
    """
    
    # --- 1. Entity Generators ---
    def get_credit_card():
        return noisy_digits(fake.credit_card_number()), "CREDIT_CARD"
    
    def get_phone():
        return noisy_digits(fake.msisdn()), "PHONE"
    
    def get_email():
        return noisy_email(fake.email()), "EMAIL"
    
    def get_person():
        return fake.name().lower(), "PERSON_NAME"
    
    def get_city():
        return fake.city().lower(), "CITY"
    
    def get_location():
        # Clean newlines from addresses
        return fake.address().replace("\n", " ").lower(), "LOCATION"
    
    def get_date():
        return noisy_date(fake.date_object()), "DATE"

    # --- 2. Templates ---
    # Lists of [Context String, Generator Function, Context String...]
    templates = [
        # Credit Card scenarios
        ["my credit card number is ", get_credit_card, " thank you"],
        ["charge the ", get_credit_card, " for the payment"],
        ["i am paying with card ", get_credit_card],
        
        # Phone scenarios
        ["call me at ", get_phone],
        ["my number is ", get_phone, " or try ", get_phone],
        ["reach ", get_person, " at ", get_phone],
        
        # Email scenarios
        ["email me at ", get_email],
        ["my address is ", get_email, " and i live in ", get_city],
        ["contact ", get_email, " regarding the issue"],
        
        # Person/Location scenarios
        ["my name is ", get_person, " and i am from ", get_city],
        ["is ", get_person, " located at ", get_location, "?"],
        ["i visited ", get_location, " on ", get_date],
        
        # Mixed scenarios
        ["confirming details for ", get_person, " phone ", get_phone, " date ", get_date],
        ["date of birth is ", get_date],
        ["send the package to ", get_city, " specifically ", get_location],
    ]

    selected_template = random.choice(templates)
    
    final_text_parts = []
    final_entities = []
    cursor = 0 # Tracks the current character character index
    
    # Optional: Add filler word at start ("um", "uh")
    if random.random() < 0.3:
        f = random.choice(["um", "uh", "ok", "so"])
        final_text_parts.append(f)
        cursor += len(f) + 1 # +1 for the space we'll add during join
    
    # --- 3. Construction Loop ---
    for item in selected_template:
        chunk_text = ""
        label = "O"
        
        if callable(item):
            # It's an entity generator
            chunk_text, label = item()
        else:
            # It's static context text
            chunk_text = item
            # Randomly insert filler words inside context
            if random.random() < 0.05: 
                 chunk_text += " uh "

        # Clean text
        chunk_text = chunk_text.strip()
        if not chunk_text: continue

        # Add to list
        final_text_parts.append(chunk_text)
        
        # Record entity if it's not O
        if label != "O":
            final_entities.append({
                "start": cursor,
                "end": cursor + len(chunk_text),
                "label": label
            })
            
        # Update cursor: length of text + 1 for the space
        cursor += len(chunk_text) + 1
        
    final_string = " ".join(final_text_parts)
    
    return {
        "id": uid,
        "text": final_string,
        "entities": final_entities
    }

def write_jsonl(filename, num_samples):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for i in tqdm(range(num_samples), desc=f"Writing {filename}"):
            data = generate_sample(f"utt_{i:04d}")
            f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    print(f"Generating Data in '{OUTPUT_DIR}'...")
    write_jsonl(f"{OUTPUT_DIR}/train.jsonl", TRAIN_SIZE)
    write_jsonl(f"{OUTPUT_DIR}/dev.jsonl", DEV_SIZE)
    write_jsonl(f"{OUTPUT_DIR}/test.jsonl", TEST_SIZE)
    print("Data generation complete.")