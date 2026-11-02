# Annotation guidelines (paper 48 revision)

Two annotators (A and B) work **independently**: do not discuss items or look at each other's sheet until both are finished. Fill only your own file (`*_sheet_A.csv` or `*_sheet_B.csv`). Open the CSV in Excel or LibreOffice and save it as CSV (UTF-8).

## Task 1: intent labels (`intent_sheet_X.csv`, 400 rows, about 40 minutes)

Each row is one message a user sent to AutoStream's chat assistant. If `prev_bot` is filled, it is the assistant's previous message; use it to interpret short replies. Write exactly one label in the last column:

| Label | Use when the message... | Examples |
|---|---|---|
| `greeting` | is only a salutation or small talk, with no request | "hi", "good morning!", "hello again" |
| `product_inquiry` | asks anything about AutoStream: plans, prices, features, limits, policies, billing, apps, support, **including troubleshooting and "how do I" questions** | "how much is Pro?", "export is stuck, any fix?", "does it support .mov?" |
| `high_intent` | wants to buy, subscribe, sign up, start a trial, upgrade, or talk to sales, **or agrees to an offer to sign up** | "I want Pro", "yes please" (after "Would you like to sign up?") |
| `general` | anything else: thanks, goodbyes, off-topic requests, chit-chat about unrelated things, unclear messages, declining an offer | "thanks!", "what's the weather", "maybe later" |

Tie-breaking rules:
1. Greeting + request: label the request ("hi, how much is Pro?" is `product_inquiry`).
2. Purchase wish + question: `high_intent` ("I want Pro, does it have 4K?").
3. Asking *how a plan works* or *what it costs* without saying they want it: `product_inquiry`.
4. Compliments or feedback without a request ("loving the editing tools!"): `general`.

## Task 2: answer quality (`qa_sheet_X.csv`, 105 rows, about 60 minutes)

Each row shows a customer question, whether it is answerable from the knowledge base, a reference answer, the relevant knowledge-base text, and an assistant's answer. You do not know which system wrote it. Fill two columns:

**correctness**
- `2` correct and complete: every fact the reference requires is present and accurate.
- `1` partially correct: some required facts are missing or vague, but nothing stated is wrong.
- `0` incorrect: a required fact is wrong, the answer contradicts the reference, or the question is not answered.
- For rows with `answerable = no`: `2` if the assistant says it does not have the information (pointing to support is fine) without inventing facts; otherwise `0`.

**hallucination**
- `1` if the answer states any specific fact (price, limit, date, feature, policy) that is **not supported by** or **contradicts** the knowledge-base text shown; otherwise `0`.
- Judge only against the knowledge-base text shown, not your own knowledge.

When finished, send the CSV back unchanged apart from the filled columns.
