# Retail Data Workbench

**Track: AI Engineering.** Profile and clean messy retail CSVs without losing data, then chat with them.
Every number in every answer comes from a calculation that actually ran, and every answer shows where it came from.

The one rule: **the model may plan. The application validates, executes and verifies.**
A persuasive answer with no computation behind it is a wrong answer.

Walkthrough video: https://drive.google.com/file/d/1vNWuiH5N8pwwjP1fx1_vmaYVUfvgw0tI/view?usp=sharing

---

## What you can check in five minutes

These are claims you do not have to take on trust. Each one is a command.

| Claim | Command | What you will see |
|---|---|---|
| The cleaner is graded against the *truth*, cell by cell | `python -m app.scorecard` | 14,662 cells compared. Of 1,529 deliberately damaged cells, **1,496 repaired exactly (97.8%)**, **0 wrong fixes, 0 damaged, 0 rows lost, 0 rows invented**. The 33 cells left alone are all flagged in the reports. |
| Expected answers do not come from the code under test | `python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output` | 28 of 28 gold questions pass. Their expected values are computed from `data/sample_truth` with plain pandas. |
| Every answer shows its receipt | `python -m app.chat.cli data/retail --ask "Revenue by region"` | A receipt block: data version, rows used, filters, joins (with unmatched rows), steps, planner. |
| Answers can be reproduced | `python -m app.chat.cli data/retail --verify SESSION_ID` | Re-runs a saved session and says `REPRODUCED` or exactly what changed (new data version, different numbers). |
| The same input is never processed twice | `python -m app.prepare data/sample` (twice) | Second time: `reused, nothing recomputed`. |
| Failures are survivable | `python -m app.evaluate --input eval/cases_edge.json --output edge.json --artifacts-dir ./edge_out` | 20 awkward cases (empty file, malformed CSV, join fan-out, bad case entries...). None crashes the batch. |

---

## Quick start (clean clone)

Needs Python 3.10 or newer. No paid service, no API key and no internet are needed for anything below.

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q                       # 270+ tests, about 25 seconds
```

**Prepare the sample data and chat with it:**

```bash
python -m app.prepare data/sample         # clean + normalize, saved as a run
python -m app.chat.cli data/retail        # interactive chat (type :quit to leave)
```

Try these in order:

1. `only stores in the West region`
2. `among those, which store has the highest return rate?`
3. `what about East?`
4. `show sales` (it asks which measure you mean), then `revenue`
5. `Which products need attention?` then `Why is P010 flagged?`
6. `What is the profit by category?` (refused, and it says what is missing)

Inside the chat: `:plan`, `:evidence` and `:trace` show the plan, the evidence and the decision trail of the last answer.

**More commands**

```bash
python -m app.chat.cli data/retail --ask "Revenue by region"      # one question
python -m app.chat.cli data/retail --trace SESSION_ID             # print a saved session
python -m app.runs list                                           # saved runs
python -m app.cleaning.plan data/sample/orders.csv                # the cleaning plan, before anything changes
python -m app.retail.report data/retail                           # joins, metrics, product health score
python -m app.chat.engine data/retail --demo                      # 4 good plans and 8 rejected ones
python scripts/generate_messy_data.py                             # regenerate the sample data (deterministic)
python scripts/build_gold.py                                      # rebuild the gold questions from the truth data
python scripts/make_edge_cases.py --big                           # rebuild the edge pack, plus a 120,000-row file
```

---

## Headless evaluation

```bash
python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output
```

Add `--rules-only` to guarantee that no model is ever called.

**Input.** A JSON list of cases, or `{"cases": [...]}`. A case:

```json
{"id": "shop_1",
 "datasets": {"orders": "path/orders.csv", "products": "path/products.csv"},
 "questions": ["Revenue by region",
               {"question": "among those, which has the highest return rate?", "session": "main",
                "expect": {"status": "ok", "first_row": {"region": "West"}}}]}
```

A case may use `datasets` (name to path), `input` (one file) or `folder`. A question may carry `expect` with
`status`, `first_row`, `rows_include`, `row_count`, `contains_text`, `refusal_code`, `plan_filters_include`,
`evidence_dataset`, `rows_after_filters` or `clarification_options_include`.
The exact input shape was left open by the assessment, so this is our design.

**Output.** `{"summary": {...}, "cases": [...]}`. For one dataset the case keys are flat (`dataset`, `profile_before`, `issues`,
`cleaning_plan`, `profile_after`, `validation`, `chat_evaluation`, `artifacts`). For several datasets each of those keys maps
`dataset name -> value`, while `issues` and `cleaning_plan` stay flat lists because every id already contains the dataset name.
Extras: `retail_decisions`, `versions`, `errors`, `status` (`ok`, `partial`, `error`).
Each case folder keeps the **original** files, the **cleaned** files and the reports, with sha256 hashes in `artifacts`.

**Guarantees.**

- One failed case never aborts the batch. A bad file, a bad dataset inside a case, a crashing question or a malformed case becomes a structured error and the next one runs.
- The output has no timestamps or timings. Two runs in different folders produce byte-identical JSON (tested).
- Each case writes only inside its own folder. A case id such as `../../x` cannot escape it. Dataset names are restricted to `[a-z0-9_]`.
- The process exits 0 when the result file was written, and 2 when the input is unreadable or has the wrong top-level shape.

**Independent expected answers.** The gold questions were not written by looking at the pipeline's output. The sample data was generated
from a clean version (`data/sample_truth`) and then damaged; the expected values are computed from the clean version with plain pandas.
The gold set deliberately avoids category questions (the system keeps `Denim` separate until a person confirms it) and stock or cost
questions (the generator destroyed that information, so no honest expected value exists).

---

## What leaves your machine

**With the default setting (`LLM_PROVIDER=none`): nothing.** The app makes no network call of any kind. Cleaning, joins, metrics, scoring,
chat, replay and evaluation all run locally on the rules planner.

**With `LLM_PROVIDER=gemini`**, one request per question goes to Google's Gemini API. It contains:

| Sent | Detail |
|---|---|
| The question text | As an escaped JSON string inside a block marked untrusted |
| Dataset, column names and column types | For example `orders_wide`, `revenue: numeric` |
| A few category labels | Up to 12 per low-cardinality text column (for example `West`, `T-Shirt`, `Naukr Mart Pune`), each matching a plain-label pattern and not instruction-like |
| The order date range | Two dates |
| The structure of the previous plan's filters | Field, operator, value of a filter the user already typed. Never results. |

| Never sent | Why |
|---|---|
| Rows, preview rows, cell values (other than the labels above) | The model plans; it does not need data |
| Identifier values (order, customer, product, store ids) | Never listed, even in a tiny dataset where a column would otherwise qualify as a "label" column |
| Names, emails, phones, addresses | Personal columns are hidden from the catalog before any prompt is built |
| Any number from a result, any answer text | The model never sees what it caused |
| File paths, file names, session ids, the API key (except in the request header) | Not part of any prompt |

Questions that try to override the rules or ask for personal data are refused by a deterministic check **before** any request is made.

**Things to know before you turn it on.**

- Column names, category labels and the question itself can contain sensitive text (a store name, a customer name typed into a question). Leave hosted mode off for such data.
- Google's own data terms apply to what is sent. We have not negotiated anything.
- The key lives only in the environment or in a git-ignored `.env`. It goes in an HTTP header, never in a URL, and it is not written to any log, report, run or saved chat (tested).
- Runs and the chat database stay on local disk with no automatic retention policy.

---

## When the AI is down

The planner order is: **deterministic safety check -> model (if enabled) -> validator -> rules planner as the fallback.**
The model is optional at every step. Nothing that computes a number depends on it.

| What happens | What the system does |
|---|---|
| No key, or `LLM_PROVIDER=none` | The rules planner answers. The model is never called. |
| Timeout, network error, rate limit (429), server error (5xx) | Retries after 0.5 s and 1 s, then the rules planner answers. The answer's trace says which error and how many attempts. |
| Key rejected (401 / 403) or bad request | No retry. The rules planner answers. |
| Reply is not valid JSON, is too long, or has the wrong shape | One repair attempt (the repair prompt holds the problem list and no data), then the rules planner. |
| Reply is valid JSON but the **validator rejects the plan** (unknown field, `sql`, limit 10,000,000, personal column, huge grouping) | The plan never runs. One repair attempt, then the rules planner. |
| Three failures in a row | A circuit breaker pauses model calls for 60 seconds, so users are not made to wait for a model that is down. One trial call is allowed afterwards. |

**Still works with the AI down:** profiling, cleaning, runs, joins, metrics, the health score, the evaluator, replay, saved sessions,
refusals, clarifying questions, product-health explanations, and every question the rules planner's grammar understands
(group and rank, filters, dates, comparisons, distinct values, summaries, counts, follow-ups, stock questions).

**Gets worse with the AI down:** unusual wording. The rules planner does not guess; it asks "I did not understand that, which of these would you like?"

**Never changes:** the numbers, the safety checks, the validator, the evidence.

**When the AI is up it can still be wrong.** The validator proves a plan is *legal and safe*; it cannot prove the plan matches what you
meant. That is why every answer shows its plan, its scope and its receipt, and why the gold tests assert exact values.

---

## What this system intentionally does not do

- **It never executes model-written SQL, Python or formulas.** Plans are data (JSON) checked against an allow-list. Joins cannot be written in a plan at all; a plan can only choose a pre-checked view.
- **It never sends rows or personal data to a model.** See above.
- **It never guesses an ambiguous date.** `05/01/2026` is converted only when the same file proves its order (a value like `27/09/2026` can only be day-first). With no proof, or conflicting proof, nothing changes and the step is marked `needs_review`.
- **It never merges a category it is unsure about.** `Tee -> T-Shirt` (0.95) is applied. `Denim -> Jeans` (0.75) is left alone, because denim might mean denim jackets. Misspellings are only suggested.
- **It never invents missing values, deletes rows that disagree, drops columns or renames columns.** Empty stays empty. Two rows with the same id but different content are flagged, not removed. Negative stock is flagged, not changed.
- **It never lets a pile of unmatched rows pose as a category in a ranking.** Those rows are left out of rankings, and the answer says so. They stay in totals and are counted in the receipt.
- **It does not answer what the data cannot support.** Profit, forecasts, causes ("why did sales drop"), discounts, competitors, weather, "last month" with no calendar: refused, with the reason and a suggestion.
- **It does not infer protected traits or profile customers.** Scores use explicit retail signals only.
- **It does not authenticate users.** The terminal chat is single-user. Sessions are kept apart by id, which is bookkeeping, not a security boundary.
- **It has no web UI and no HTTP API.** The terminal chat and the evaluator are the interfaces.
- **It has no background job queue and no streaming.** Everything is synchronous and in memory (files up to 50 MB, tables up to 1,000,000 rows). 120,000 rows clean in about 7 seconds.
- **It cannot interrupt one long pandas call.** The time budget is checked between steps.
- **It does not claim live-model accuracy.** No live Gemini call was run while building this (see Known limitations).

---

## How it works

```
 raw CSVs
    v  RUN        fingerprint the inputs -> run id; copy the originals; build in a temp folder; publish in one step
    v  PROFILE    read everything as text; counts, nulls, duplicates, formats, parse failures
    v  DETECT     issues with stable ids (ISS:orders:revenue:NUMBERS_AS_TEXT)
    v  PLAN       a to-do list BEFORE any change: id, reason, columns, risk, source, status, rows it may remove
    v  CLEAN      plain deterministic Python, one step at a time, each on a copy
    v  VALIDATE   7 checks, including: cleaning the output again changes nothing
    v  NORMALIZE  retail aliases with confidence; uncertain ones go to review
    v  CATALOG    datasets, columns, types, version fingerprints; personal columns hidden; safe joined views
 question
    v  SCREEN     deterministic: injection, personal data, profit / forecast / "why", out-of-scope
    v  PLANNER    rules planner, or Gemini planner with fallback
    v  VALIDATOR  allow-list: fields, operators, types, limits, group size
    v  EXECUTOR   plain pandas on a typed copy; evidence recorded at every step
    v  ANSWER     text built ONLY from the executed result, plus a receipt
    v  STORE      question -> plan -> evidence -> answer -> trace, per session, in SQLite
```

| Folder | Contents |
|---|---|
| `app/cleaning/` | `profiler`, `issues`, `plan`, `cleaner`, `validate`, `pipeline`, `rules` (the allow-list of cleaning actions) |
| `app/retail/` | `normalize` (aliases + confidence), `joins` (fan-out checks), `metrics`, `scoring` (product health) |
| `app/chat/` | `plan_schema`, `catalog`, `validator`, `executor`, `engine`, `rules_planner`, `ai_planner`, `llm_client`, `answers`, `insights`, `store`, `verify`, `assistant`, `cli` |
| `app/` | `runs`, `prepare`, `evaluate`, `scorecard`, `config`, `errors` |
| `eval/`, `cases.json` | gold questions and the edge-case pack |
| `data/sample/`, `data/sample_truth/` | messy sample data and the clean data it was made from |

### Runs

Every preparation is saved as `data/runs/<run-id>/` with `original/` (byte-for-byte copies), `cleaned/`, `retail/` and `run.json`
(status, inputs with sha256, per-dataset counts and validation, artifact hashes). The run id is a fingerprint of the input files and
the pipeline version, so the same input maps to the same run and is **reused**; one changed byte makes a new run. A run is built in a
hidden temp folder and moved into place in one step, so a crash cannot leave a half-written run that looks finished. Failed and partial
runs are saved too, with the structured error. Statuses: `succeeded`, `partial`, `failed`.

### Data pipeline

- **Read as text.** pandas never guesses types, so `007` cannot become `7` and `Rs. 1,299` cannot hide.
- **Profile before and after:** rows, columns, nulls, exact and near duplicates, key conflicts, distinct counts, date formats, numbers
  stored as text, parse failures, spelling variants, mixed code formats. Personal columns never show example values; instruction-like
  text is counted but never copied.
- **Plan before mutation.** Every step has a stable id (`STEP:orders:order_date:convert_date_to_iso:nn-slash`), a reason, columns, a risk,
  a source (`code`), a status (`planned`, `needs_review`, `skipped`; the cleaner adds `applied` or `failed`) and the exact range of rows it may remove.
- **Validation after cleaning:** same columns in the same order; rows removed within the planned range; every output row came from an
  input row; columns no step touched are identical; no step failed; no unexpected issue left; and cleaning the output again changes nothing
  (checked with a real second pass and by file hash).
- **Lineage.** Each file gets a version label such as `orders_clean@<12 hex>`, a fingerprint of its content. Reports store file names, not machine paths, so the same data gives the same report anywhere.

### Retail intelligence

- **Normalization with uncertainty.** Applied at 0.85 confidence or more; otherwise left alone and listed for review. Rules follow column names, not file names.
- **Safe joins.** Discovered from shared `*_id` columns and classified (`many_to_one`, `one_to_one`, `many_to_many`). The "one" side must be
  unique or the join is refused (`FAN_OUT_RISK`). After a join the row count must equal the left table's or the result is discarded.
  Left joins keep every row; unmatched rows (for example 5 orders for product `P999`) and blank keys are counted in the receipt.
- **Metrics:** revenue, orders, units, average order value, return rate, returned revenue, stock-outs, sell-through, stock cover. Each comes with its formula, the rows read and anything excluded.
- **Product health score (0 to 100).** Margin 30%, sales speed 30%, return rate 20%, stock cover 20%, each mapped to 0-1 by a fixed rule.
  A missing signal is left out, the remaining weights are rescaled, and the gap is listed as missing evidence. The answer shows every signal,
  weight and point. The model never touches a score.

### Chat

- **Plan format.** `intent` (aggregate, list, distinct, describe), `dataset`, `filters`, `group_by`, `metrics`, `sort`, `limit`, optional `time_bucket`. Unknown keys such as `sql` or `join` are rejected by the schema.
- **Validator limits.** Limit at most 100; 8 filters; 3 group columns; 6 metrics; estimated groups at most 10,000; the right operator for each column type; personal columns unavailable; every problem reported, with "did you mean" hints.
- **Multi-turn.** Filters carry over after cues such as "among those" or "what about East?" and are replaced when the same column is mentioned again. A fresh question does not inherit old filters, and says so.
- **Ambiguity.** "show sales" asks: revenue, units or orders. "Delhi" in two columns asks: customer city or store city. A misspelled value ("Wset region") asks "did you mean West?" instead of being ignored. Pending questions are saved, so they survive a restart.
- **Answers carry evidence and a receipt:** dataset and version, columns used, filters, joins behind the dataset, ordered operations with row counts, rows considered, whether the preview was cut off, formulas and notes.

| Challenge case from the assessment | What happens |
|---|---|
| Ambiguous question | Targeted clarification with options |
| Unanswerable question | Refused with the reason and what you could ask instead |
| Prompt injection in a cell | Stays plain data. As a filter value it matches nothing. It is never shown to a model. |
| Prompt injection in a question | Refused by the deterministic check before any model call |
| Malicious or expensive plan | Rejected by the validator, with every problem listed |
| LLM outage or invalid JSON | See "When the AI is down" |
| Raw vs cleaned conflict | Every answer names the dataset version; a test shows `_clean` and `_retail` give different return counts |

---

## AI versus code

| Done by code (always) | Done by the model (optional, never trusted) |
|---|---|
| Counting, summing, averaging, grouping, filtering | Reading free-form wording |
| Joins and the fan-out check | Choosing which valid plan matches the question |
| Duplicate detection, schema checks, parsing dates and numbers | Short clarification or refusal wording |
| Scoring arithmetic and the health score | |
| Validating every plan | |
| Safety checks | |
| Building the answer text from executed results | |

The model cannot change a number, read a row, read personal data, write a join, raise a limit or run anything. Its reply must be one JSON object
of a fixed shape; free text it writes is length-limited and stripped of links, and is never used for a number.

**Provider and model:** Google Gemini over plain HTTPS (`generateContent`, key in the `x-goog-api-key` header, temperature 0, JSON output).
Default model `gemini-3.5-flash`, changeable with `LLM_MODEL`. Turn it on in `.env` (copy `.env.example`):

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...        # never commit this file; it is in .gitignore
LLM_MODEL=gemini-3.5-flash
LLM_MODE=ai_first         # or rules_first: ask the model only when the rules do not understand
```

## Security

- Filenames, cell values, model output and plans are all untrusted input.
- Only `.csv` files up to 50 MB. Run and case folders are confined to their parent. Dataset names are restricted to `[a-z0-9_]`.
- No user input is ever executed: no `eval`, no SQL, no generated Python.
- Questions are capped at 500 characters; results at 100 rows.
- Personal columns (names such as `name`, `email`, `phone`, `address`) are hidden from questions, never shown as profile examples and left out of joined views.
- Secrets: see "What leaves your machine".

## Tests

`python -m pytest -q` runs 270+ tests in about 25 seconds. The ones that matter most:

- **Scorecard against the truth:** 0 wrong fixes, 0 damage, 0 lost rows; and a test that sabotages the cleaner and confirms the scorecard fails.
- **No silent row loss:** a validator test removes a row on purpose and must fail.
- **Idempotency:** every sample file cleaned twice gives an identical file hash; the same input reuses the same run.
- **Safe joins:** fan-out refused, unmatched rows counted, personal columns excluded, many-to-many refused.
- **Plan validation:** 22 bad plans, each rejected with the right code; garbage input never crashes the engine.
- **Grounding:** a test swaps the executed result for a fake number and the answer text follows the fake number, proving the text layer cannot invent numbers.
- **Replay:** a saved session reproduces exactly, and a changed data version or tampered numbers are detected.
- **Model failure paths:** timeout, rate limit, rejected key, invalid JSON, oversized reply, dangerous plans, circuit breaker; the prompt never contains rows, personal data or identifiers; the key never leaks.
- **Evaluator:** one failure never stops the batch; a wrong expectation fails (so a pass means something); output is repeatable.
- **Docs:** every command in this README must exist.

## Trade-offs

- **Rules planner plus optional AI, not AI only.** Rules are free, instant and repeatable, so the project always runs and the evaluator is deterministic. The model adds flexibility for unusual wording. Cost: the rules understand a fixed set of question shapes.
- **Plans, not code.** A plan cannot express anything the executor does not support. Cost: less expressive than SQL.
- **Pre-built joined views.** A fan-out join is impossible to request. Cost: only joins the catalog discovered are available.
- **Uncertainty stays visible.** A 0.75-confidence alias is not applied. Cost: `Denim` stays separate until a person confirms it.
- **Text until the end.** Safer for cleaning; typed copies are built for queries. Fine at this scale.

## Known limitations

- **No live Gemini call was run during development** (the build environment had no internet). The request format follows Google's documentation and is tested against a fake HTTP layer and a scripted model. Check the first live call once. Model names and free-tier limits change; set `LLM_MODEL` accordingly.
- Personal-data detection is by column name, so an oddly named personal column could be missed.
- The alias tables are small and hand-written. A model could suggest more, but suggestions would go through the same confidence rule.
- The rules planner covers a fixed set of question shapes. "Worst" always means "lowest". Relative times are refused.
- The health score needs at least 5 orders for a return rate. With the sample data most products look overstocked, because the generator gave every store plenty of stock.
- The scorecard is only as good as the truth it is given. It cannot grade cells the generator destroyed (missing costs, stock), so it reports those as left-as-is and checks that each is flagged.

## Requirement map

| Assessment requirement | Where |
|---|---|
| Persist a run with status, metadata, original and cleaned artifact | `app/runs.py`, `data/runs/<run-id>/` |
| Profile before and after | `app/cleaning/profiler.py`; both profiles in every report |
| Issues, plan with id / reason / fields / risk / source / status | `issues.py`, `plan.py` |
| Deterministic apply; row and schema deltas; idempotency | `cleaner.py`, `validate.py`, tests, `app/scorecard.py` |
| Two or more retail capabilities | All four: normalization, safe joins, metrics, health score |
| Chat: filter, group, top/bottom, distinct, describe, multi-turn, joins, evidence | `app/chat/` |
| Allow-listed plan execution; no model code | `plan_schema.py`, `validator.py`, `executor.py` |
| Challenge cases | Table above |
| Batch evaluator; one failure does not abort | `app/evaluate.py` |
| Security and privacy documented | "What leaves your machine", "Security" |
| Tests incl. one refusal and one model-failure path | `tests/` |
| README, `.env.example`, sample data | This file, `.env.example`, `data/sample/` |

## Sample data and schema

The sample data is **synthetic**, created by `scripts/generate_messy_data.py` (seed 42, fully reproducible). It plants about 1,100 documented
problems (counts in `data/sample/planted_issues.json`): four date formats, ambiguous dates, `Rs. 1,299` numbers, number words, duplicates,
spelling variants, mixed SKU formats, orphan product ids, missing values, negative stock and a prompt-injection string inside a product name.
`data/sample_truth/` holds the same data before the damage.

| Table | Columns | Links |
|---|---|---|
| `stores` | store_id, store_name, city, region | |
| `products` | product_id, sku, product_name, category, brand, price, cost | |
| `customers` | customer_id, name, email, city, signup_date | personal columns: name, email |
| `orders` | order_id, order_date, customer_id, product_id, store_id, quantity, revenue, returned | to customers, products, stores |
| `inventory` | product_id, store_id, stock_on_hand, reorder_level, last_restock_date | to products, stores |

The system is not tied to this data. Rules follow column names, joins are discovered from shared `*_id` columns, and a dataset without a date column or without a join key still works (tested).
