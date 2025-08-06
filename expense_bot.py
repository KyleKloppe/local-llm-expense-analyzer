#!/usr/bin/env python3
"""
expense_bot.py
----------------

A private, local‑only AI‑powered financial analyzer tool. This script reads
CSV files containing expense report data, performs basic analysis to compute
spending by vendor and category, detects duplicate or unusually large
transactions, and uses a local language model via Ollama to generate
human‑readable summaries. All processing occurs locally; no internet
connections or cloud APIs are used.

Usage:
    python expense_bot.py --file path/to/data.csv [--model phi] [--threshold 10000]

Features:
    • CSV parsing for common expense fields (Date, Vendor, Category, Amount, Notes).
    • Automatic grouping and total calculations by category and vendor.
    • Detection of duplicate transactions and high‑value anomalies.
    • Generation of natural language summary via a local LLM through Ollama.
    • Export of summaries and analysis metadata to the Reports/ directory.
    • Optional simple bar chart of top categories saved as PNG.

This script is designed to run on Windows (Python 3.9+) without any
internet connectivity. If `ollama` is not installed, the summary is
generated using a stub fallback.
"""

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # use non‑interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Charting will be disabled if matplotlib isn't available


def parse_csv(file_path: str) -> List[Dict[str, str]]:
    """Read a CSV file and return a list of rows as dictionaries.

    The parser is forgiving about column names. It maps common variants of
    expected fields (Date, Vendor, Category, Amount, Notes) to normalized
    keys. Unknown columns are preserved but ignored by analysis.
    """
    required_mappings = {
        'date': {'date', 'payment_date', 'transaction_date'},
        'vendor': {'vendor', 'creditor_name', 'payee'},
        'category': {'category', 'subjective_group', 'subjective_subgroup', 'subjective_detail'},
        'amount': {'amount', 'net_amount', 'gross_amount', 'value'},
        'notes': {'notes', 'memo', 'description'},
    }
    # Read header and map columns
    # Attempt to read using UTF‑8; if there are invalid bytes,
    # ignore them to avoid crashing on non‑UTF‑8 input (e.g., latin‑1).  This
    # preserves as much data as possible without raising a UnicodeDecodeError.
    with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        columns = [c.strip().lower() for c in reader.fieldnames or []]
        # Build mapping from normalized key -> actual column name
        column_map: Dict[str, Optional[str]] = {key: None for key in required_mappings}
        for idx, col in enumerate(columns):
            for norm_key, variants in required_mappings.items():
                if col in variants:
                    column_map[norm_key] = reader.fieldnames[idx]
        rows = []
        for raw_row in reader:
            row = {}
            for norm_key, col_name in column_map.items():
                if col_name and col_name in raw_row and raw_row[col_name] != '':
                    row[norm_key] = raw_row[col_name].strip()
                else:
                    row[norm_key] = ''
            # also keep unknown columns
            rows.append(row)
        return rows


def cast_amount(value: str) -> float:
    """Convert a string to a float, handling commas and parentheses."""
    if not value:
        return 0.0
    # Remove commas and currency symbols
    v = value.replace(',', '').replace('$', '').strip()
    # Handle parentheses for negatives
    if v.startswith('(') and v.endswith(')'):
        v = '-' + v[1:-1]
    try:
        return float(v)
    except ValueError:
        return 0.0


def analyze_expenses(rows: List[Dict[str, str]], high_threshold: float) -> Dict:
    """Perform analysis on expense data.

    Returns a dictionary with totals per category and vendor, duplicates,
    high‑value anomalies, and month‑over‑month spending change.
    """
    totals_by_category: Dict[str, float] = defaultdict(float)
    totals_by_vendor: Dict[str, float] = defaultdict(float)
    all_amounts: List[float] = []
    duplicates: List[Dict[str, str]] = []
    high_values: List[Dict[str, str]] = []

    seen_transactions: Counter = Counter()

    dates = []
    for row in rows:
        amount = cast_amount(row['amount'])
        category = row['category'] or 'Uncategorized'
        vendor = row['vendor'] or 'Unknown'
        date = row['date']
        totals_by_category[category] += amount
        totals_by_vendor[vendor] += amount
        all_amounts.append(abs(amount))
        if date:
            dates.append(date)
        # Duplicate detection: track by (vendor, amount, date)
        key = (vendor.lower(), amount, date)
        if seen_transactions[key] > 0:
            duplicates.append(row)
        seen_transactions[key] += 1
        # High value
        if abs(amount) >= high_threshold:
            high_values.append(row)

    # Determine month‑over‑month change if possible
    month_totals: Dict[str, float] = defaultdict(float)
    for row in rows:
        date_str = row['date']
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            try:
                dt = datetime.strptime(date_str, '%d/%m/%Y')
            except ValueError:
                continue
        month_key = dt.strftime('%Y-%m')
        month_totals[month_key] += cast_amount(row['amount'])

    month_keys = sorted(month_totals.keys())
    mom_change = None
    if len(month_keys) >= 2:
        last_month, prev_month = month_keys[-1], month_keys[-2]
        prev_total = month_totals[prev_month]
        last_total = month_totals[last_month]
        if prev_total != 0:
            mom_change = ((last_total - prev_total) / abs(prev_total)) * 100

    # Determine average and threshold suggestions
    average_amount = mean(all_amounts) if all_amounts else 0

    return {
        'totals_by_category': dict(totals_by_category),
        'totals_by_vendor': dict(totals_by_vendor),
        'duplicates': duplicates,
        'high_values': high_values,
        'mom_change': mom_change,
        'average_amount': average_amount,
    }


def generate_prompt(analysis: Dict, threshold: float) -> str:
    """Build a prompt for the LLM summarizing the expense report analysis."""
    lines = []
    lines.append("You are a financial assistant. Analyze the following data summary and identify major spending categories, flag anomalies or duplicate expenses, and generate a 3-bullet business summary of the report. Return only the response, no explanations.\n")
    lines.append("=== Expense Summary ===")
    # Top categories
    top_categories = sorted(analysis['totals_by_category'].items(), key=lambda x: -abs(x[1]))[:5]
    lines.append("Top categories by spend:")
    for category, total in top_categories:
        lines.append(f"- {category}: {total:.2f}")
    # Duplicates
    if analysis['duplicates']:
        lines.append(f"\nDetected {len(analysis['duplicates'])} potential duplicate transactions.")
        for dup in analysis['duplicates'][:3]:
            lines.append(f"Duplicate: {dup['vendor']} on {dup['date']} for {dup['amount']}")
    else:
        lines.append("\nNo duplicate transactions detected.")
    # High values
    if analysis['high_values']:
        lines.append(f"\nDetected {len(analysis['high_values'])} high-value transactions (≥{threshold}).")
        for hv in analysis['high_values'][:3]:
            lines.append(f"High: {hv['vendor']} on {hv['date']} for {hv['amount']}")
    else:
        lines.append(f"\nNo high-value transactions (≥{threshold}) detected.")
    # Month over month
    if analysis['mom_change'] is not None:
        lines.append(f"\nMonth-over-month change: {analysis['mom_change']:+.2f}%")
    else:
        lines.append("\nNot enough data for month-over-month analysis.")
    return '\n'.join(lines)


def call_ollama(prompt: str, model: str) -> str:
    """Call a local language model via ollama and return its response.

    If `ollama` is not installed or an error occurs, return a fallback
    summary instructing the user to review the analysis manually.
    """
    try:
        result = subprocess.run([
            'ollama', 'run', model
        ], input=prompt, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            response = result.stdout.strip()
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            return response or "No response generated."
        else:
            return ("Unable to generate summary because the local LLM encountered an error. "
                    "Please review the analysis manually.")
    except FileNotFoundError:
        return ("Local model not installed. Please install an Ollama model such as 'phi' "
                "and try again, or review the analysis manually.")
    except Exception as exc:
        return ("An unexpected error occurred while invoking the local model. "
                "Please check your environment and try again.")


def save_report(summary: str, analysis: Dict, output_dir: Path, base_name: str) -> Tuple[Path, Path]:
    """Save the summary to a text file and the analysis metadata to JSON.

    Returns the paths to the text and JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = output_dir / f"{base_name}_{timestamp}.txt"
    json_path = output_dir / f"{base_name}_{timestamp}.json"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary.strip() + "\n")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    return txt_path, json_path


def plot_top_categories(analysis: Dict, output_dir: Path, base_name: str) -> Optional[Path]:
    """Plot a simple bar chart of top categories and save to a PNG file."""
    if plt is None:
        return None
    categories = sorted(analysis['totals_by_category'].items(), key=lambda x: -abs(x[1]))[:5]
    names = [c[0] for c in categories]
    values = [abs(c[1]) for c in categories]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, values, color='#0A66C2')
    ax.set_xlabel('Total Spend')
    ax.set_title('Top Spending Categories')
    ax.invert_yaxis()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}_chart.png"
    plt.savefig(png_path)
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(description='Local AI‑powered expense analyzer')
    parser.add_argument('--file', '-f', required=False, help='Path to CSV file with expense data')
    parser.add_argument('--model', '-m', default='phi', help='Name of the local model to use (ollama)')
    parser.add_argument('--threshold', '-t', type=float, default=None, help='High value threshold for anomalies')
    args = parser.parse_args()

    csv_path = args.file or 'sample_data.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found. Please provide a valid file.")
        return
    rows = parse_csv(csv_path)
    # Determine threshold: user provided or 3x average amount
    if args.threshold is not None:
        high_threshold = args.threshold
    else:
        # compute a quick average to set threshold
        amounts = [abs(cast_amount(r['amount'])) for r in rows if r['amount']]
        avg = mean(amounts) if amounts else 0
        high_threshold = avg * 3 if avg > 0 else 10000

    analysis = analyze_expenses(rows, high_threshold)
    prompt = generate_prompt(analysis, high_threshold)
    print("\n--- Prompt to model ---\n")
    print(prompt)
    print("\n--- Generating summary... ---\n")
    summary = call_ollama(prompt, args.model)
    print("\n--- Summary ---\n")
    print(summary)
    # Save report
    reports_dir = Path('Reports')
    txt_path, json_path = save_report(summary, analysis, reports_dir, base_name='expense_report')
    print(f"\nSummary saved to {txt_path}\nAnalysis saved to {json_path}")
    # Plot chart
    chart_path = plot_top_categories(analysis, reports_dir, base_name='expense_report')
    if chart_path:
        print(f"Chart saved to {chart_path}")


if __name__ == '__main__':
    main()