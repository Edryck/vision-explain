from src.extract_evidence import evidence_to_text

EVIDENCE_HEADER = "Automated inspection report (from an anomaly detection model):"


def build_question_block(question):
    return f"Question: {question}"


def build_prompt_a(question):
    return build_question_block(question)


def build_prompt_b(question, evidence):
    report = evidence_to_text(evidence)
    return f"{EVIDENCE_HEADER}\n{report}\n\n{build_question_block(question)}"


def build_messages(prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
