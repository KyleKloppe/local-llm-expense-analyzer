# Offline Expense Analyzer

This project provides a simple, private and offline financial analyzer for
business expense reports.  It reads expense data from a CSV file, performs
basic aggregations and anomaly detection, and uses a locally installed
language model via [Ollama](https://github.com/ollama/ollama) to generate
natural‑language summaries.  All data processing happens on your local
machine—no information is sent to the cloud.

## Features

* **CSV Parsing** – Supports common column names for dates, vendors,
  categories and amounts.  Unknown fields are ignored gracefully.
* **Analysis** – Calculates total spend by category and vendor, flags
  duplicate transactions and unusually high‑value expenses, and computes
  month‑over‑month spending changes.
* **AI Summary** – Invokes a local LLM via the `ollama run` CLI to
  summarise the report in plain English.  If no model is available, a
  fallback message is returned.
* **Reporting** – Writes both the raw analysis (in JSON) and the AI
  summary (in text) to a time‑stamped file in the `Reports/` directory.
  Optionally plots a bar chart of the top categories.
* **Privacy‑safe** – The script never makes any network requests; it
  operates fully offline provided an Ollama model is installed locally.

## Requirements

* Python 3.9 or newer
* A locally installed LLM accessible via `ollama` (e.g. `phi`, `llama2`,
  `mistral`) – optional, the script will still run without one but will
  produce a stub summary.
* (Optional) `matplotlib` for generating charts.

## Usage

```bash
python expense_bot.py --file path/to/your_expenses.csv --model phi --threshold 10000
