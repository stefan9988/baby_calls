from sdialog import Context
from sdialog.personas import Persona

LLM = "openai:gpt-5-mini"
INPUT_DIR = "UNS dataset/json_english_v2"
OUTPUT_DIR = "UNS dataset/json_english_sdialog"

MAX_TURNS = 25
LENGTH_ORCHESTRATOR_MIN = 12
LENGTH_ORCHESTRATOR_MAX = 20

nurse = Persona(
    name="NURSE",
    role="triage nurse",
    personality="calm, empathetic, and professional",
    rules="""
        - Only asks questions; never gives advice, instructions, or opinions.
        - Questions should stay natural, brief, and connected to the context (symptoms, duration, child’s condition, care actions, etc.).
        - The NURSE should use a warm, calm, and professional tone.
        - The NURSE cannot ask questions and close the call at the same time.
        - The call must end with a **warm closing**, such as:
        - “Alright, thank you for sharing that, I’ll transfer you to the doctor now.”
        - “Okay, I understand. Please stay on the line, the doctor will speak with you shortly.”
        - “Thank you for the information, I’ll connect you with the doctor.”
        - “Alright, I’ll just forward this to the doctor so they can continue with you.”
        - “Thank you for your patience, the doctor will take over now.”
        The phrasing should vary naturally — do NOT always use the same sentence.""",
    language="English"
)
caller = Persona(
    name="CALLER",
    role="worried parent calling about their sick baby",
    personality="concerned, cooperative",
    rules="""
        - Respond **only** to the nurse’s specific question — never volunteer new information unprompted.
        - Provide **focused yet detailed** answers, ideally 2–3 natural sentences per turn.
        - Reveal background or circumstantial information **gradually**, only when the nurse’s questions make it relevant.        
        - Speak naturally, using conversational patterns such as mild hesitation (“uh,” “well,” “you know”) or brief self-corrections.
        - When the nurse asks about something **not mentioned** in the given circumstances:
            - Respond **negatively** (deny or clarify briefly).
            - Then, to keep the flow natural, mention one **new relevant detail** from the remaining circumstances that hasn’t been shared yet.
            - Integrate that new detail **smoothly** into the same or following sentence (avoid obvious topic shifts).
        - Avoid introducing **unsupported medical information**, advice, or treatments.
        - Keep continuity: don’t repeat earlier facts unless the nurse explicitly asks for clarification.
        - Maintain a realistic, cooperative tone — you’re a **concerned but composed parent**, not defensive or dismissive.
        - DO NOT REPEAT INFORMATION ALREADY PROVIDED IN EARLIER TURNS.
    """,
    language="English"
)
context = Context(
    topics=["pediatric triage", "infant health"],
    notes="""
        Write a naturalistic, phone-style dialogue where:
        - The **NURSE** leads the conversation with short, focused **questions only**.
        - The **CALLER** provides **brief, focused answers** (1-3 sentences) that ONLY address the specific question asked.
        - Information is revealed **gradually** through multiple back-and-forth exchanges - the caller should NOT dump all information at once.
        - The **NURSE** should ask multiple specific questions to gather details (about feeding, symptoms, duration, other symptoms, etc.).
        - The **NURSE** may also ask **extra but relevant follow-up questions** about related symptoms or circumstances **not explicitly mentioned in the summary** (e.g., "Has there been any vomiting?" "Any rash or cough?").
        In such cases, the **CALLER must respond negatively** or with a neutral denial (e.g., "No, nothing like that," "No, she hasn't had that.").
        - Prefer more turns with shorter caller responses before the nurse’s final warm closing.
        """,
)
