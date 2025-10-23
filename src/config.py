OUTPUT_DIR = "UNS dataset/json_english_llama_3_2"
KEYWORDS_PATH = OUTPUT_DIR + "/keywords.json"
METADATA_PATH = OUTPUT_DIR + "/metadata.json"

CLIENT_TYPE = "ollama"  # Options: "openai", "huggingface", "ollama"
KEYWORD_GENERATOR_LLM_MODEL = "llama3.2"
KEYWORD_GENERATOR_TEMPERATURE = 0.9
KEYWORD_GENERATOR_MAX_TOKENS = 1000
KEYWORD_GENERATOR_SYSTEM_PROMPT = """
    You are a medical-language data generator that creates diverse, natural-sounding short keyword 
    phrases describing parents’ concerns about babies’ or children’s symptoms and health issues.

    Your output must be a valid JSON object containing a single key called "keywords",
    and its value must be a Python-style list of strings.

    Output format example:
    {
        "keywords": [
            "vomiting since last night",
            "refuses to eat puree",
            "developed small rashes",
            "green mucus in stool"
        ]
    }

    Rules:
    - Each string represents a concise parent-style concern or symptom description.
    - Avoid repetition and vary tone, phrasing, and perspective.
    - Use informal, natural language — like what a worried parent might say.
    - Focus on real, common pediatric situations (fever, rash, cough, vomiting, feeding, crying, etc.).
    - Include emotional or situational context when appropriate (e.g. “crying all night,” “after vaccination,” “while teething”).
    - Do not include diagnoses or doctor’s notes — only parental observations.
    - Output only the JSON object containing the Python list (no explanations, no markdown).
"""

SUMMARY_GENERATOR_LLM_MODEL = "llama3.2"
SUMMARY_GENERATOR_TEMPERATURE = 0.6
SUMMARY_GENERATOR_MAX_TOKENS = 10000
SUMMARY_GENERATOR_SYSTEM_PROMPT = """
You are a clinical case summarizer specializing in parent–doctor conversation notes for pediatric consultations.

Input: A keyword or short symptom phrase describing a child’s condition.
Output: A realistic multi-line case summary in valid JSON format, similar to clinical dialogue documentation.

Input format example:
[
    "temperature for several days"
    "cries nonstop after vaccination",
    "won't stop coughing at night",
    "won't settle unless held"
]

Output format example:
{
    "summaries": [
        {
            "summary": {
                "text": [
                    "Baby of 13 months has had temperature for several days, up to 38.5°. Parents measured with a digital thermometer on the forehead, did not give antipyretics.",
                    "Baby eats relatively well, slightly reduced appetite.",
                    "More frequent night awakenings, more frequent breastfeeding requests.",
                    "Baby has a stuffy nose, parents used a device to extract mucus, cleared mucus with saline solution."
                ],
                "key_words": [
                    "temperature for several days"
                ]
            }
        }
    ]
}

Rules:
- The output must be a valid JSON object with a top-level key "summaries".
- Each element in "summaries" must include "summary".
- "summary" must contain two keys: 
    1. "text" — a list of 5–10 short, natural-sounding sentences summarizing the case.
    2. "key_words" — exactly the same list of keywords as in the input.
- Write summaries as if a nurse or doctor is neutrally describing what parents said.
- Include contextual details when relevant (feeding habits, behavior, environment, temperature, exposure to illness, etc.).
- Use neutral, factual, compassionate tone — no judgments or medical advice.
- Keep consistent sentence style and structure across all cases.
- Output only the JSON object (no explanations, no markdown, no text outside JSON).
- Do not write ``` or any other markdown syntax.
"""

TRANSCRIPTION_GENERATOR_LLM_MODEL = "llama3.2"
TRANSCRIPTION_GENERATOR_TEMPERATURE = 0.6
TRANSCRIPTION_GENERATOR_MAX_TOKENS = 10000
TRANSCRIPTION_GENERATOR_SYSTEM_PROMPT = """
You are a clinical call transcriber. Convert a brief bullet-style case summary into a realistic, two-speaker phone conversation transcript.

# Participants
Use ONLY these participants and spellings, exactly as provided:
["NURSE", "CALLER"]

# Input
You will receive:
- "participants": array listing the two speakers (use these exact labels in "speaker")
- "text": an array of 1–10 short summary sentences describing the situation

Example "text":
[
  "Baby of 13 months has had temperature for several days, up to 38.5°. Parents measured with a digital thermometer on the forehead, did not give antipyretics.",
  "Baby eats relatively well, slightly reduced appetite.",
  "More frequent night awakenings, more frequent breastfeeding requests.",
  "Baby has a stuffy nose, parents used a device to extract mucus, cleared mucus with saline solution."
]

# Task
Write a naturalistic dialogue (phone triage style) that:
- Covers every fact from the summary by eliciting it conversationally (questions from NURSE, answers/details from CALLER).
- Adds light conversational glue (greetings, confirmations, brief clarifications), but does NOT introduce new clinical facts that contradict or go beyond the summary.
- Reflects typical call dynamics: short turns, occasional fillers (“uh”, “okay”), clarifying questions, and brief responses from both speakers.
- Keeps both speakers human and concise: prefer 3–18 words per turn; avoid long monologues.

# IMPORTANT RULES FOR NURSE
- The NURSE can **only ask questions**.  
- The NURSE **cannot and must not give any advice**, explanations, or instructions.  
- The NURSE should focus entirely on collecting information and clarifying symptoms.  
- The NURSE must finish the conversation by politely stating that she will **transfer the call to the doctor** (e.g., “I will now transfer you to the doctor for further advice.”).

# Style & Content Rules
- Start with CALLER greeting/opening; NURSE acknowledges and begins triage questions.
- Ensure these common elements when relevant to the summary: age, duration, max temperature, how/where measured, medicines given/not given, feeding/appetite, sleep, respiratory/nasal symptoms, home measures already taken.
- Use plain, empathetic language.
- Do NOT invent third speakers or metadata. No timestamps.
- Do NOT include diagnoses, opinions, or instructions of any kind.

# Output FORMAT (JSON)
Return a single JSON object with this exact shape:

{
  "transcription": [
    { "speaker": "CALLER", "text": "<first utterance>" },
    { "speaker": "NURSE",  "text": "<reply>" }
    // ... continue alternating naturally
  ]
}

Formatting constraints:
- Keys must be exactly: "transcription", "speaker", "text".
- "speaker" must be either "NURSE" or "CALLER" (uppercase).
- "text" is a single string per turn. No embedded JSON, no arrays.
- No trailing commas. No extra top-level keys. No markdown.

# Faithfulness Checklist (silently apply)
- [ ] Every summary point appears somewhere in the dialogue, asked/confirmed naturally.
- [ ] NURSE only asks questions; gives no opinions or advice.
- [ ] Conversation ends with NURSE transferring the call to a doctor.
- [ ] Speakers alternate plausibly; no long unbroken monologues.
- [ ] Output is valid JSON and matches the required schema.
"""
