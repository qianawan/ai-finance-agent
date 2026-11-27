# AI Finance Agent

A Streamlit web app that turns a personal accounting CSV into an AI-generated financial diagnosis.

Upload a transaction CSV (a MoneyThings export or any standard accounting CSV), and the app cleans and categorises the data, splits income from expense, and uses a DeepSeek LLM to produce a structured financial report plus an interactive Q&A advisor grounded in the same data.

## What it does

- **CSV parsing & cleaning** — strips thousands separators, coerces amounts, and filters out non-consumption rows (transfers, top-ups, reimbursements).
- **Income/expense analysis** — computes total income, total expense and net cash flow, with per-scene and per-tag breakdown tables.
- **AI diagnosis report** — prompts a DeepSeek LLM to output a five-part review: spending health, impulse-spending traces, hidden recurring costs, waste to cut, and three actionable next-month suggestions.
- **AI financial advisor chat** — a follow-up conversation grounded in the same data, with guardrails against fabricating figures the data cannot support.

## Stack

Python · Streamlit · pandas · OpenAI SDK (pointed at the DeepSeek API, `deepseek-chat`)

## Run it

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...   # from platform.deepseek.com
streamlit run app.py
```

## Notes

- The AI steps require a DeepSeek API key with a funded account.
- `.env` (API key) is git-ignored — supply your own key.
- The UI and prompts are written in Chinese (the app was built for personal use with Chinese bank/accounting CSV exports).
