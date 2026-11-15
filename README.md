# Retail Data & Conversational Intelligence Workbench

Take messy retail CSVs, clean them without losing data, and chat with them. Every number in
every answer comes from a calculation that actually ran, and every answer carries its evidence.

> **Track: AI Engineering.** Validated structured outputs, a retry / fallback / circuit-breaker layer,
> prompt-injection defenses, and an evaluation harness with gold questions.

**The one rule the whole project is built on:** the model may *plan*. The application must
*validate, execute and verify*. A persuasive answer with no computation behind it is a wrong answer.

---

## 1. Quick start (clean clone)

Needs Python 3.10 or newer. No paid service, no API key and no internet are needed for anything below.

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q                       # 250+ tests, about 25 seconds
```

**Evaluate (the command reviewers run, no UI):**

```bash
python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output
```

`cases.json` holds 28 gold questions on the included messy sample data. Expected result:
`28/28 passed`. Add `--rules-only` to guarantee no model is ever called.
Cleaned files appear in `evaluation_output/sample_retail_gold/clean/` and `.../retail/`.

**Chat with the data:**

```bash
python -m app.prepare data/sample         # clean + normalize every CSV in the folder
python -m app.chat.cli data/retail        # interactive chat (type :quit to leave)
python -m app.chat.cli data/retail --ask "Revenue by region"      # one question
python -m app.chat.cli data/retail --trace SESSION_ID             # replay a saved session
```

Inside the chat: `:plan`, `:evidence` and `:trace` show the plan, evidence and decision trail of the last answer.

Questions to try, in this order:

1. `only stores in the West region`
2. `among those, which store has the highest return rate?`
3. `what about East?`
4. `show sales` (it asks which measure you mean), then `revenue`
5. `Which products need attention?` then `Why is P010 flagged?`
6. `What is the profit by category?` (refused, and it says what is missing)

**Other commands**

```bash
python -m app.cleaning.profiler data/sample/orders.csv      # profile one file
python -m app.cleaning.plan data/sample/orders.csv          # see the cleaning plan before anything changes
python -m app.cleaning.pipeline data/sample --out data/cleaned
python -m app.retail.report data/retail                     # joins, metrics, product health score
python -m app.chat.engine data/retail --demo                # 4 good plans and 8 rejected ones
python scripts/generate_messy_data.py                       # regenerate the sample data (deterministic)
python scripts/build_gold.py                                # rebuild the gold questions from the truth data
python scripts/make_edge_cases.py --big                     # rebuild the edge pack (+ a 120,000-row file)
python -m app.evaluate --input eval/cases_edge.json --output edge.json --artifacts-dir ./edge_out
```

---

## 2. Architecture and data flow

```
 raw CSVs
    |
    v   PROFILE      read everything as text; counts, nulls, duplicates, formats, parse failures
    v   DETECT       issues with stable ids (ISS:orders:revenue:NUMBERS_AS_TEXT)
    v   PLAN         to-do list BEFORE any change: id, reason, columns, risk, source, status
    v   CLEAN        plain deterministic Python, one step at a time, each on a copy
    v   VALIDATE     schema same? rows as planned? nothing invented? untouched columns identical? cleaning twice changes nothing?
    v   NORMALIZE    retail aliases with confidence (Tee -> T-Shirt, Yes -> Y); uncertain ones go to review
    |
    v   CATALOG      datasets + columns + types + version fingerprints; personal columns hidden;
                     ready-made SAFE joined views: orders_wide, inventory_wide
    |
 question
    |
    v   SCREEN       deterministic: injection, personal data, profit / forecast / "why", out-of-scope  -> refuse
    v   PLANNER      rules planner (always available)  or  Gemini planner (optional, with fallback)
    v   VALIDATOR    plan must pass an allow-list: fields, operators, types, limits, group size
    v   EXECUTOR     plain pandas on a typed copy; evidence recorded at each step
    v   ANSWER       text built ONLY from the executed result
    v   STORE        question -> plan -> evidence -> answer -> trace, saved in SQLite per session
```

| Folder | What is inside |
|---|---|
| `app/cleaning/` | `profiler`, `issues`, `plan`, `cleaner`, `validate`, `pipeline`, `rules` (allow-list of cleaning actions) |
| `app/retail/` | `normalize` (aliases + confidence), `joins` (fan-out checks), `metrics`, `scoring` (product health) |
| `app/chat/` | `plan_schema`, `catalog`, `validator`, `executor`, `engine`, `rules_planner`, `ai_planner`, `llm_client`, `answers`, `insights`, `store`, `assistant`, `cli` |
| `app/evaluate.py`, `app/prepare.py` | the batch evaluator and the one-command setup |
| `eval/`, `cases.json` | gold questions and the edge-case pack |
| `data/sample/`, `data/sample_truth/` | messy sample data and the clean "truth" it was made from |
| `tests/` | invariant tests (section 9) |

---

## 3. The AI-versus-code boundary

| Done by **code** (always) | Done by the **model** (optional, never trusted) |
|---|---|
| Counting, summing, averaging, grouping, filtering | Reading free-form wording ("which category brings in the most money") |
| Joins and the fan-out check | Choosing which *valid plan* matches the question |
| Duplicate detection, schema checks, parsing dates and numbers | Short clarification or refusal wording |
| Scoring arithmetic and the product health score | |
| Validating every plan against the allow-list | |
| Safety screening (injection, personal data) | |
| Building the answer text from executed results | |

What the model can never do: change a number, read a row, read personal data, write a join, bypass a limit, or run anything.
Its JSON is parsed against a strict schema and then validated by the same gate as every other plan. If the validator rejects it,
it gets **one** repair attempt (the repair prompt contains the problem list and no data). If that fails, the rules planner answers.

**Provider and model:** Google Gemini over plain HTTPS (`generateContent`, key in the `x-goog-api-key` header, temperature 0,
JSON output). Default model `gemini-3.5-flash`, changeable with `LLM_MODEL`.
**The app is fully usable without it:** with no key it runs the rules planner.

**Data that leaves the machine (only when `LLM_PROVIDER=gemini`):**

- the question text (as an escaped JSON string, marked untrusted)
- dataset and column names and types
- up to 12 harmless category labels per low-cardinality column (for example `West`, `T-Shirt`), each sanitized
- the order date range
- the structure (not the results) of the previous plan's filters, for follow-up questions

**Never sent:** rows, order or customer ids (identifier values are never listed, even in tiny datasets), names, emails, phone
numbers, addresses, any result number, product names, or text that looks like an instruction. Questions that are injections or ask
for personal data are refused *before* any model call.

Turn the model on in `.env` (copy `.env.example`):

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...        # never commit this file; it is in .gitignore
LLM_MODEL=gemini-3.5-flash
LLM_MODE=ai_first         # or rules_first: ask the model only when the rules do not understand
```

---

## 4. Data pipeline

- **Read as text.** pandas never guesses types, so `007` cannot become `7` and `Rs. 1,299` cannot hide.
- **Profile before and after:** rows, columns, nulls, exact and near duplicates, key conflicts, distinct counts, date formats,
  numbers stored as text, parse failures, spelling variants, mixed code formats. Personal columns never show example values,
  and instruction-like text is counted but never copied.
- **Plan before mutation.** Every step has a stable id (`STEP:orders:order_date:convert_date_to_iso:nn-slash`), reason, columns,
  risk, source (`code`), status (`planned`, `needs_review`, `skipped`; the cleaner adds `applied` or `failed`) and the exact number of
  rows it may remove. Only actions on an allow-list can run.
- **Safety rules built in:** missing values are never invented; columns are never dropped or renamed; rows with the same id but
  different content are flagged, never deleted; negative stock is flagged, never changed.
- **The ambiguous-date decision.** `05/01/2026` can be 5 January or 1 May. The planner does not guess. It looks for *proof* in the
  same file: a value like `27/09/2026` can only be day-first. If a format has proof for one order and none for the other, it applies
  that order and lists every ambiguous row in the change log. With no proof, or conflicting proof, the step becomes `needs_review`
  and nothing changes.
- **Validation after cleaning** (7 checks): same columns in the same order, rows removed within the planned range, every output row
  came from an input row, columns no step touched are identical, no step failed, no unexpected issue left, and **cleaning the
  output again changes nothing** (verified by an actual second pass and by file hash).
- **Lineage:** each file gets a version label such as `orders_clean@c8b71733fa73`, a fingerprint of its content. Reports store file
  names, not machine paths, so the same data gives the same report anywhere.

## 5. Retail intelligence (all four capabilities)

- **Normalization with uncertainty.** `Tee -> T-Shirt` (0.95), `Bangalore -> Bengaluru` (0.98), `Yes` and `1` to `Y` are applied.
  Anything under 0.85 confidence is left alone and marked `needs_review`: `Denim -> Jeans` is only 0.75 because denim could also
  mean denim jackets. Misspellings are only *suggested*, never applied. Rules follow column names, not file names.
- **Safe joins.** Joins are discovered from shared `*_id` columns and classified (`many_to_one`, `one_to_one`, `many_to_many`).
  The "one" side must be unique or the join is refused (`FAN_OUT_RISK`). After a join the row count must equal the left table or
  the result is discarded. Left joins keep every row: unmatched rows (for example 5 orders for product `P999`) are counted and
  reported, never dropped, so revenue is never lost. A plan cannot write a join at all; it can only pick a pre-checked view.
- **Metrics:** revenue, orders, units, average order value, return rate, returned revenue, stock-outs, sell-through, stock cover.
  Each comes back with its formula, the rows read and anything excluded.
- **Product health score (0 to 100, explainable).** Margin 30%, sales speed 30%, return rate 20%, stock cover 20%, each mapped to
  0 to 1 by a fixed rule. A missing signal is left out, the remaining weights are rescaled, and the gap is listed as missing evidence.
  The answer shows every signal, weight and points, so the total can be checked by hand. The model never touches a score.

## 6. Chat assistant

- **Plan format.** `intent` (aggregate, list, distinct, describe), `dataset`, `filters`, `group_by`, `metrics`, `sort`, `limit`,
  optional `time_bucket`. Unknown keys such as `sql` or `join` are rejected by the schema.
- **Validator limits:** limit at most 100, 8 filters, 3 group columns, 6 metrics, estimated groups at most 10,000, the right operator
  for each column type, personal columns unavailable, every problem reported (not only the first), with "did you mean" hints.
  A soft time budget is checked between steps.
- **Evidence on every answer:** dataset and version, columns used, filters, the joins behind the dataset (rows before and after,
  unmatched rows), the ordered operations with row counts, rows considered, whether the preview was cut off, formulas, and notes.
- **Multi-turn.** Filters carry over after cues such as "among those" or "what about East?" and are replaced when the same column
  is mentioned again. A fresh question does not inherit old filters (and says so). Sessions are separate.
- **Ambiguity.** "show sales" asks: revenue, units or orders. "Delhi" in two columns asks: customer city or store city.
  A misspelled value ("Wset region") asks "did you mean West?" and is never silently ignored. Pending questions are saved, so they
  survive a restart.
- **Rankings** leave out rows with no match in the group column (a pile of unmatched rows is not a real category) and say so.
- **Saved sessions** (SQLite): question, plan, evidence, answer and trace for every turn. No prompts or keys are stored.

### The assessment's challenge cases

| Case | What happens |
|---|---|
| Ambiguous question | Targeted clarification with options; answerable by word or number |
| Unanswerable question | Refused with the reason and what you could ask instead (profit, forecasts, "why", discounts, weather...) |
| Prompt injection in a cell | Stays plain data. As a filter value it matches nothing. It is never shown to the model. |
| Prompt injection in a question | Refused by the deterministic screen before any model call |
| Malicious / expensive plan | Rejected by the validator: unknown fields or operators, limit 10,000,000, a grouping that would make millions of groups, SQL or join keys |
| LLM outage / invalid JSON | Retries with backoff (0.5 s, 1 s), one repair attempt, a circuit breaker, then the rules planner. Data operations never depend on the model. |
| Raw vs cleaned conflict | Every answer names the dataset version, e.g. `orders_wide_retail@91930d1a7526`; a test shows `_clean` and `_retail` give different return counts |

---

## 7. Evaluation

`python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output`

- A case is `{"id", "datasets" | "input" | "folder", "questions": [...]}`. A question may carry `expect`
  (`status`, `first_row`, `rows_include`, `row_count`, `contains_text`, `refusal_code`, `plan_filters_include`, ...).
- The output holds a `summary` and a `cases` list. For one dataset the keys are flat (`dataset`, `profile_before`, `issues`,
  `cleaning_plan`, `profile_after`, `validation`, `chat_evaluation`, `artifacts`); for several datasets each key maps
  `dataset name -> value`. Extras: `retail_decisions`, `versions`, `errors`, `status` (`ok`, `partial`, `error`).
- **One failed case never aborts the batch.** A bad file, a bad dataset inside a case, a crashing question, or a malformed case each
  become a structured error and the batch continues.
- **Repeatable:** no timestamps or timings in the output; two runs in different folders produce byte-identical JSON.
- **Independent answers.** The gold questions' expected values are computed from `data/sample_truth` (the clean data before the mess
  was added) with plain pandas, *not* by the pipeline under test. The gold set deliberately avoids category questions (the system
  keeps `Denim` separate until a person confirms it) and stock or cost questions (the generator destroyed that information).
- **Edge pack** (`eval/cases_edge.json`, 20 cases; `--big` adds a 21st): header-only, empty, all-null column, malformed CSV, non-UTF-8, duplicate ids,
  ambiguous dates, single row, wrong file type, missing file, injection in a cell, join fan-out, missing join key, one bad file in a
  multi-file case, a case id that tries to escape the folder, and malformed case entries. With `--big`, also a 120,000-row file (about 10 seconds).

## 8. Security and privacy decisions

- Filenames, cell values, model output and plans are all treated as untrusted input.
- Only `.csv` files up to 50 MB; case output folders are confined to `--artifacts-dir`; dataset names are restricted to
  `[a-z0-9_]`; case ids cannot escape the folder.
- The API key lives only in the environment or `.env` (git-ignored). It goes in a header, never a URL, and it is not in any error,
  log, report or saved chat (tested).
- Personal columns (names such as `name`, `email`, `phone`, `address`) are hidden from questions, never shown as profile examples,
  and left out of joined views.
- Questions are capped at 500 characters; results at 100 rows.
- No user input is ever executed: no `eval`, no SQL, no generated Python. Plans are data checked against an allow-list.
- Customer analytics use only explicit retail signals; nothing infers protected traits.

## 9. Tests

`python -m pytest -q` runs 250+ tests (about 25 seconds). The ones that matter most:

- **No silent row loss:** a validator test removes a row on purpose and must fail; row counts match the answer key.
- **Idempotency:** every sample file cleaned twice gives an identical file hash.
- **Independent truth:** cleaned totals (revenue, returned orders) equal the generator's answer key.
- **Safe joins:** fan-out refused, orphans counted, personal columns excluded, many-to-many refused.
- **Plan validation:** 22 bad plans, each rejected with the right code; garbage input never crashes the engine.
- **Grounding:** a test swaps the executed result for a fake number and the answer text follows the fake number, proving the text
  layer cannot invent numbers.
- **Multi-turn, clarification, 10 refusal paths, persistence across a restart, session isolation.**
- **Model failure paths:** timeout, rate limit, rejected key, invalid JSON, oversized reply, dangerous plans, circuit breaker,
  prompt contents (no rows, no personal data, no identifiers), and the key never leaking.
- **Evaluator:** one failure never stops the batch; a wrong expectation fails (so a pass means something); repeatable output.

## 10. Design decisions and trade-offs

- **Rules planner plus optional AI, not AI only.** Rules are free, instant, repeatable and need no key, so the project always runs and
  the evaluator is deterministic. The AI adds flexibility for unusual wording. Trade-off: the rules understand a fixed set of question
  shapes and ask for clarification otherwise.
- **Plans, not code.** Strict JSON plans cannot express anything the executor does not support. Trade-off: less expressive than SQL.
- **Pre-built joined views.** A plan cannot specify a join, so fan-out is impossible by construction. Trade-off: only the joins the
  catalog discovered are available.
- **Uncertainty is kept visible.** A 0.75-confidence alias is not applied. Trade-off: "Denim" stays a separate category until a person
  confirms it.
- **Text everywhere until the end.** Safer for cleaning, slower to type-convert later. Fine at this scale.

## 11. Known limitations

- **No live Gemini call was run during development** (the build environment had no internet). The request format follows Google's
  documentation and is tested against a fake HTTP layer and a scripted model. The first live call should be checked once.
  Model names and free-tier limits change; set `LLM_MODEL` accordingly.
- Everything is synchronous and in memory (files up to 50 MB, tables up to 1,000,000 rows). There is no background job queue or
  chunked reading; a production version would add both. 120,000 rows take about 7 seconds to clean.
- The time budget is checked between steps; one very long pandas call cannot be interrupted.
- Personal-data detection is by column name, so an oddly named personal column could be missed.
- Sessions are separated by id, but there is no login. The CLI is single-user.
- The rules planner is limited to the question shapes above. "Worst" always means "lowest". Relative times ("last month") are refused.
- Alias tables are small and hand-written. A model could suggest more later, but they would go through the same confidence rule.
- The product health score needs at least 5 orders for a return rate. With the sample data most products look overstocked because the
  generator gave every store plenty of stock.
- There is no web UI; the terminal chat is the interface.

## 12. Sample data and schema

The sample data is **synthetic**, created by `scripts/generate_messy_data.py` (seed 42, fully reproducible). It plants 1,100+
documented problems (listed with counts in `data/sample/planted_issues.json`): four date formats, ambiguous dates, `Rs. 1,299`
numbers, number words, duplicates, spelling variants, mixed SKU formats, orphan product ids, missing values, negative stock,
and a prompt-injection string inside a product name. `data/sample_truth/` holds the same data before the mess was added.

| Table | Columns | Links |
|---|---|---|
| `stores` | store_id, store_name, city, region | |
| `products` | product_id, sku, product_name, category, brand, price, cost | |
| `customers` | customer_id, name, email, city, signup_date | personal columns: name, email |
| `orders` | order_id, order_date, customer_id, product_id, store_id, quantity, revenue, returned | to customers, products, stores |
| `inventory` | product_id, store_id, stock_on_hand, reorder_level, last_restock_date | to products, stores |

The system is not tied to this data: rules follow column names, joins are discovered from shared `*_id` columns, and a dataset
without a date column or without a join key still works (tested).
