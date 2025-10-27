import pandas as pd
import json
from openai import OpenAI

print("🚀 Script started")

# Initialize OpenAI client (use env var or replace with your key)
client = OpenAI(api_key="YOUR_API_KEY")

# Load your full dataset
df = pd.read_csv("Data/FRIENDS_SCRIPT.csv")

# Filter for the problematic episode
df_ep = df[df["EPISODE_ID"] == "S9E8"].copy()

# Join all utterances into one text block
episode_text = "\n".join(
    f"{row['speaker']}: {row['text']}" for _, row in df_ep.iterrows()
)

print("📡 Sending request to OpenAI API... (this may take up to 1 minute)")

# Send to API
response = client.chat.completions.create(
    model="gpt-4-turbo",
    temperature=0,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": (
                "You are a meticulous data-cleaning assistant for TV scripts. "
                "Your ONLY job is to output valid JSON that can be parsed with json.loads(). "
                "Do not include explanations, notes, or text outside JSON braces. "
                "Do not restart or repeat the output. "
                "Every string must be properly quoted with double quotes."
            ),
        },
        {
            "role": "user",
            "content": (
                "You will be given a messy episode transcript that contains spoken lines and scene directions. "
                "Your task:\n"
                "1. Separate embedded scene directions (e.g. 'Later that day.', 'Ross and Rachel's Apartment.') "
                "   into standalone entries labeled with speaker = 'Scene Directions'.\n"
                "2. Keep all spoken lines as-is with the correct speaker names.\n"
                "3. Increment the scene number whenever a new scene direction appears.\n"
                "4. Preserve the order of all utterances and number them sequentially starting at 1.\n"
                "5. Output JSON in this exact format:\n\n"
                "{\n"
                '  "transcript": [\n'
                '    {"scene": 1, "speaker": "Scene Directions", "text": "[Scene: Monica and Chandler\'s apartment.]", "utterance": 1},\n'
                '    {"scene": 1, "speaker": "Monica Geller", "text": "Hey Hon, could you help me get the plates down?", "utterance": 2}\n'
                "  ]\n"
                "}\n\n"
                "Make sure the JSON is valid, closed properly, and contains NO text outside braces.\n\n"
                f"Here is the episode text:\n{episode_text}"
            ),
        },
    ],
)

print("✅ API response received, parsing JSON...")

# --- Safe JSON parsing ---
raw_output = response.choices[0].message.content.strip()

start = raw_output.find("{")
end = raw_output.rfind("}")
if start != -1 and end != -1:
    raw_output = raw_output[start:end+1]

try:
    cleaned_json = json.loads(raw_output)
except json.JSONDecodeError as e:
    print("⚠️ JSON parsing failed:", e)
    with open("debug_response.txt", "w", encoding="utf-8") as f:
        f.write(raw_output)
    raise SystemExit("❌ Saved raw model output to debug_response.txt for manual fix.")

# --- Convert to DataFrame ---
df_cleaned = pd.DataFrame(cleaned_json["transcript"])
df_cleaned["EPISODE_ID"] = "S9E8"
df_cleaned["season"] = 9
df_cleaned["episode"] = 8
df_cleaned["source"] = "api_cleaned"

df_cleaned.to_csv("S9E8_cleaned.csv", index=False)
print("💾 Cleaned episode saved as S9E8_cleaned.csv")

# --- Merge back with the main dataset ---
df_others = df[df["EPISODE_ID"] != "S9E8"]
df_final = pd.concat([df_others, df_cleaned], ignore_index=True)
df_final = df_final.sort_values(by=["season", "episode", "scene", "utterance"]).reset_index(drop=True)

df_final.to_csv("friends_fixed.csv", index=False)
print("🎉 Merged dataset saved as friends_fixed.csv")