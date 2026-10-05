# Message Explainer

Turn confusing English business messages into clear next steps.

Message Explainer is a small local-first AI utility built for the Hacktoberfest Weekend Challenge 2026 — **Build for a Friend**.

## Why I built it

I often work with overseas customers and partners in English. Not everyone I work with is equally comfortable reading business messages in English, so colleagues sometimes ask for help understanding what a message means, what matters, and what action they need to take.

Message Explainer turns a pasted public or non-sensitive English message into four predictable sections:

1. **WHAT DOES THIS MEAN?**
2. **WHAT MATTERS?**
3. **WHAT DO I NEED TO DO?**
4. **WHAT SHOULD I CHECK?**

The goal is not literal translation. The goal is:

**MESSAGE → MEANING → PRIORITY → ACTION**

## Why open-weight AI

The app runs `HuggingFaceTB/SmolLM2-360M-Instruct` locally with Transformers and PyTorch.

- No paid AI API
- No external account required by the app
- No database
- No message history service
- CPU-friendly prototype

A small deterministic post-processing layer keeps the four-section contract stable and preserves source-grounded facts such as dates, times, amounts, URLs, and email addresses.

## Safety and privacy

Use only public or non-sensitive text.

The app is intentionally not designed for:

- medical advice
- financial or investment decisions
- legal advice
- security credentials
- private customer/company information
- other high-risk messages

If something cannot be confirmed from the source message, the app should say so instead of inventing it.

## Demo

Synthetic example:

> Your appointment scheduled for October 10 at 3 PM has been moved to October 12 at 2 PM. Please confirm the new time by October 8.

Expected structure:

- **WHAT DOES THIS MEAN?** The appointment changed.
- **WHAT MATTERS?** The new appointment is October 12 at 2 PM.
- **WHAT DO I NEED TO DO?** Confirm the new time by October 8.
- **WHAT SHOULD I CHECK?** New appointment: Oct 12, 2 PM / Confirmation deadline: Oct 8.

See the synthetic demo image: [demo/message-explainer-demo.svg](demo/message-explainer-demo.svg)

## Tech stack

- Python 3.12
- HuggingFaceTB/SmolLM2-360M-Instruct
- PyTorch
- Transformers
- Gradio
- pytest

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL shown in the terminal.

The first run downloads the open-weight model from Hugging Face, so an internet connection is required only for that model download.

## Tests

```bash
pytest -q
```

The test suite covers the four-section contract, fact extraction, safety blocking, and deterministic grounding helpers.

## Limitations

The 360M model is intentionally small.

- It is not a full translation system.
- It can still be repetitive or incomplete.
- Users should compare the output with the original message.
- The app avoids unsupported certainty and high-risk use cases.

## Project files

- `app.py` — local Gradio app
- `requirements.txt` — dependencies
- `tests/test_app.py` — deterministic tests
- `demo/message-explainer-demo.svg` — synthetic demo image
- `DEMO_CAPTURE.md` — demo notes

## Challenge note

This repository contains no real company, customer, partner, contract, contact, or other confidential information. All examples are synthetic.
