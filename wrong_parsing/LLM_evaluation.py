import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import os

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY"  # insert you OpenAI API key here
INPUT_FILE = "friends_lines_with_colon.csv"
OUTPUT_FILE = "friends_lines_evaluated.csv"
MODEL = "gpt-4o-mini"

# --- INIT ---
client = OpenAI(api_key=API_KEY)

# --- Load input ---
df = pd.read_csv(INPUT_FILE)

# --- Load existing progress if available ---
if os.path.exists(OUTPUT_FILE):
    results_df = pd.read_csv(OUTPUT_FILE)
    done_ids = set(results_df["line_id"])
    print(f"✅ Loaded {len(done_ids)} already processed lines.")
else:
    results_df = pd.DataFrame(columns=["line_id", "evaluation"])
    done_ids = set()

# --- Define the evaluator function ---
def evaluate_line(text):
    prompt = f"""
You are helping clean a TV script dataset from the show *Friends*.
Some lines have parsing errors (they accidentally include the speaker label before a colon or other scene directions).

Examples of problematic lines:
- "No, he was this creepy guy from high school who had this huge crush on her since like the ninth grade. Ross with a look of wondering how long this is going to go on on his face: Still me."  → NOT OK
- "No, I'm not talking about you. It was your fat friends brother with that bad afro, do you remember? Ross starts talking over her 'do you remember' line: Amy. I'm going to save you some time, ok. All me. Monica and Chandler's Apartment."       → NOT OK
- "Oh I was just thinking. You know what would be incredible? If you guys died. Ross first has a look of 'huh' then changes it to sarcastic happy: Thank you Amy."   → NOT OK
- ": sweetie it's ok, I still love you, let me be a part of this." → OK 
- "Okay, Monica: Right foot red."            → OK (colon used naturally)

Task: Decide if the line looks correctly parsed or not.

Return only one of these labels:
- ok → looks fine
- not ok → probably misparsed
- unsure → unclear, needs manual review

Now classify this line:

"{text}"
"""

    for _ in range(3):  # retry up to 3 times
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()

            if "not ok" in answer:
                return "not ok"
            elif "ok" in answer:
                return "ok"
            elif "unsure" in answer:
                return "unsure"
            else:
                return "unsure"
        except Exception as e:
            print("⚠️ Error:", e)
            time.sleep(5)
    return "unsure"

# --- Main loop (resumable) ---
new_results = []
remaining_df = df[~df["line_id"].isin(done_ids)]

print(f"Processing {len(remaining_df)} new lines...")

for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):
    result = evaluate_line(row["text"])
    new_results.append({"line_id": row["line_id"], "evaluation": result})

    # Save every 20 lines for safety
    if len(new_results) % 20 == 0:
        temp = pd.DataFrame(new_results)
        results_df = pd.concat([results_df, temp], ignore_index=True)
        results_df.to_csv(OUTPUT_FILE, index=False)
        new_results = []

# Save any remaining lines
if new_results:
    temp = pd.DataFrame(new_results)
    results_df = pd.concat([results_df, temp], ignore_index=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

print("🎉 Done! Results saved to:", OUTPUT_FILE)