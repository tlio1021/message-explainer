import re
from functools import lru_cache

import gradio as gr

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
HEADINGS = [
    "WHAT DOES THIS MEAN?",
    "WHAT MATTERS?",
    "WHAT DO I NEED TO DO?",
    "WHAT SHOULD I CHECK?",
]

HIGH_RISK_PATTERNS = [
    r"\bmedical\b", r"\bdiagnos", r"\bprescription\b",
    r"\binvest(?:ment|ing)?\b", r"\bstock\b", r"\bcrypto\b",
    r"\blegal\b", r"\blawsuit\b", r"\battorney\b",
    r"\bpassword\b", r"\bcredential\b", r"\bprivate key\b",
    r"\bseed phrase\b", r"\bone[- ]time password\b", r"\botp\b",
    r"의료", r"진단", r"투자", r"법률", r"비밀번호", r"인증번호",
]

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)


def contains_high_risk_content(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.I) for pattern in HIGH_RISK_PATTERNS)


def extract_source_facts(text: str) -> list[str]:
    patterns = [
        rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,\s*\d{{4}})?\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
        r"\b\d{1,2}\s*(?:AM|PM|am|pm)\b",
        r"[$€£]\s?\d+(?:\.\d{1,2})?",
        r"\b\d+(?:\.\d+)?%",
        r"https?://[^\s)>\]]+",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.I):
            value = match if isinstance(match, str) else "".join(match)
            value = value.strip()
            if value and value not in found:
                found.append(value)
    return found


def _appointment_fallback(text: str) -> str | None:
    m = re.search(
        rf"scheduled\s+for\s+(.+?)\s+has\s+been\s+moved\s+to\s+(.+?)\.\s*"
        rf"Please\s+confirm(?:\s+the\s+new\s+time)?\s+by\s+(.+?)(?:\.|$)",
        text,
        flags=re.I | re.S,
    )
    if not m:
        return None
    old_time, new_time, confirm_by = [x.strip() for x in m.groups()]
    return (
        "WHAT DOES THIS MEAN?\n"
        "예약 일정이 변경되었다는 뜻입니다.\n\n"
        "WHAT MATTERS?\n"
        f"기존 일정은 {old_time}이고, 새 일정은 {new_time}입니다.\n\n"
        "WHAT DO I NEED TO DO?\n"
        f"{confirm_by}까지 변경된 일정을 확인해야 합니다.\n\n"
        "WHAT SHOULD I CHECK?\n"
        f"기존 일정: {old_time}\n새 일정: {new_time}\n확인 마감: {confirm_by}"
    )


def _price_change_fallback(text: str) -> str | None:
    money = re.findall(r"[$€£]\s?\d+(?:\.\d{1,2})?", text)
    date = re.search(rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,\s*\d{{4}})?\b", text, flags=re.I)
    if len(money) < 2 or not re.search(r"price|pricing|cost|subscription", text, flags=re.I):
        return None
    when = date.group(0) if date else "원문에 표시된 적용일"
    return (
        "WHAT DOES THIS MEAN?\n"
        "서비스 가격이 변경된다는 안내입니다.\n\n"
        "WHAT MATTERS?\n"
        f"원문에 표시된 가격이 {money[0]}에서 {money[1]}로 변경됩니다.\n\n"
        "WHAT DO I NEED TO DO?\n"
        "변경된 가격을 확인하고, 계속 이용할지 판단하세요. 원문에 별도 회신이나 취소 절차가 있는지도 확인하세요.\n\n"
        "WHAT SHOULD I CHECK?\n"
        f"기존 가격: {money[0]}\n변경 가격: {money[1]}\n적용 시점: {when}"
    )


def _deadline_fallback(text: str) -> str | None:
    if not re.search(r"deadline|submit|due|complete .* by|respond .* by", text, flags=re.I):
        return None
    facts = extract_source_facts(text)
    check = "\n".join(f"- {x}" for x in facts) if facts else "원문에서 마감 날짜와 시간을 다시 확인하세요."
    return (
        "WHAT DOES THIS MEAN?\n"
        "제출이나 회신 마감이 있는 안내입니다.\n\n"
        "WHAT MATTERS?\n"
        "정해진 기한 안에 필요한 행동을 완료해야 합니다.\n\n"
        "WHAT DO I NEED TO DO?\n"
        "원문에서 요구하는 제출 또는 회신 행동을 확인하고 마감 전에 완료하세요.\n\n"
        "WHAT SHOULD I CHECK?\n"
        f"{check}"
    )


def deterministic_fallback(text: str) -> str | None:
    for fn in (_appointment_fallback, _price_change_fallback, _deadline_fallback):
        result = fn(text)
        if result:
            return result
    return None


@lru_cache(maxsize=1)
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def build_prompt(message: str) -> str:
    return f"""You are Message Explainer, a cautious assistant for ordinary, non-sensitive business and service messages.

Read only the source message below. Do not invent facts. When possible, explain in simple Korean.
Return exactly these four headings and nothing else:
WHAT DOES THIS MEAN?
WHAT MATTERS?
WHAT DO I NEED TO DO?
WHAT SHOULD I CHECK?

For dates, times, prices, URLs, email addresses, and deadlines, copy only facts that actually appear in the source.
If an action or fact cannot be confirmed, say that it could not be confirmed from the message.

SOURCE MESSAGE:
{message}
"""


def run_local_model(message: str) -> str:
    tokenizer, model = load_model()
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=240,
        do_sample=False,
        repetition_penalty=1.08,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def normalize_sections(text: str, source: str) -> str:
    cleaned = text.strip()
    sections: dict[str, str] = {}
    upper = cleaned.upper()

    for i, heading in enumerate(HEADINGS):
        start = upper.find(heading)
        if start == -1:
            sections[heading] = "I couldn't confirm this from the message."
            continue
        content_start = start + len(heading)
        next_positions = []
        for later in HEADINGS[i + 1:]:
            pos = upper.find(later, content_start)
            if pos != -1:
                next_positions.append(pos)
        end = min(next_positions) if next_positions else len(cleaned)
        content = cleaned[content_start:end].strip(" \n:-")
        sections[heading] = content or "I couldn't confirm this from the message."

    facts = extract_source_facts(source)
    if facts:
        grounded = "\n".join(f"- {fact}" for fact in facts)
        check = sections["WHAT SHOULD I CHECK?"]
        if not all(fact in check for fact in facts):
            check = f"{check}\n\nSource facts to verify:\n{grounded}"
        sections["WHAT SHOULD I CHECK?"] = check

    return "\n\n".join(f"{heading}\n{sections[heading]}" for heading in HEADINGS)


def explain_message(message: str) -> str:
    message = (message or "").strip()
    if not message:
        return "Paste a public or non-sensitive message first."

    if contains_high_risk_content(message):
        return (
            "This prototype does not process medical, financial, legal, security, "
            "credential-bearing, private, or other high-risk messages. "
            "Please use a public or non-sensitive example."
        )

    fallback = deterministic_fallback(message)
    if fallback:
        return fallback

    try:
        raw = run_local_model(message)
        return normalize_sections(raw, message)
    except Exception as exc:
        facts = extract_source_facts(message)
        fact_text = "\n".join(f"- {x}" for x in facts) if facts else "No date, time, amount, URL, or email could be confirmed."
        return (
            "WHAT DOES THIS MEAN?\n"
            "로컬 모델을 실행하지 못해 의미를 확정하지 못했습니다.\n\n"
            "WHAT MATTERS?\n"
            "원문을 직접 확인하세요.\n\n"
            "WHAT DO I NEED TO DO?\n"
            "모델이 준비된 뒤 다시 실행하세요.\n\n"
            "WHAT SHOULD I CHECK?\n"
            f"{fact_text}\n\n"
            f"Local model error: {type(exc).__name__}"
        )


EXAMPLE = (
    "Your appointment scheduled for October 10 at 3 PM has been moved to "
    "October 12 at 2 PM. Please confirm the new time by October 8."
)

with gr.Blocks(title="Message Explainer") as demo:
    gr.Markdown(
        "# Message Explainer\n"
        "Turn confusing English messages into clear next steps. "
        "**Use only public or non-sensitive text.**"
    )
    message = gr.Textbox(
        label="Paste a message",
        lines=8,
        value=EXAMPLE,
        placeholder="Paste a public or non-sensitive English message...",
    )
    run = gr.Button("Explain")
    output = gr.Textbox(label="Clear next steps", lines=18)
    run.click(explain_message, inputs=message, outputs=output)
    gr.Markdown(
        "Prototype limitation: not for medical, financial, legal, security, "
        "credential-bearing, private, or other high-risk messages."
    )

if __name__ == "__main__":
    demo.launch()
