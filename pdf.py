from flask import Flask, request, render_template, redirect, url_for
import pdfplumber
from openai import OpenAI
import os
from collections import defaultdict
app = Flask(__name__)
from datetime import datetime, timedelta
import re

from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)

import re
# ===== Target-day mapping by bank =====
BANK_TO_DAYS = {
    "TATA":       [1, 5, 15, 25],
    "IDFC":       [1, 10, 20],
    "BAJAJ":      [2, 10, 20, 30],
    "YES BANK":   [5, 15, 25],
    "PIRAMAL":    [5, 15, 25, 30],
    "AXIS BANK":  [5, 15, 25],
    "HDFC":       [2, 5, 7, 15, 20],
    "HERO":       [],  # N/A
    "POONAWALA":  [5, 10, 15, 20, 25, 30],
    "ICICI":      [5, 12, 15],
    "AU":         "DAILY",
    "CHOLA":      [5, 10, 15, 20, 25],
}
DEFAULT_DAYS = [5]  # fallback if bank not in map
BANK_MONTH_POLICY = {
    "TATA":      {"months": 3,  "include_current": False},  # LATEST 3 MONTH + CURRENT MONTH IS NOT COUNTED AS MONTH
    "IDFC":      {"months": 7,  "include_current": True},   # LATEST 6 MONTH + CURRENT MONTH
    "BAJAJ":     {"months": 4,  "include_current": True},   # LATEST 3 MONTH + CURRENT MONTH
    "YES BANK":  {"months": 7,  "include_current": True},   # LATEST 6 MONTH + CURRENT MONTH
    "PIRAMAL":   {"months": 6,  "include_current": False},  # LATEST 6 MONTH - CURRENT MONTH IS NOT COUNTED AS MONTH
    "AXIS BANK": {"months": 4,  "include_current": True},   # LATEST 3 MONTH + CURRENT MONTH
    "HDFC":      {"months": 7,  "include_current": True},   # LATEST 6 MONTH + CURRENT MONTH
    "HERO":      {"months": 0,  "include_current": False},  # N/A
    "POONAWALA": {"months": 7,  "include_current": True},   # LATEST 6 MONTH + CURRENT MONTH
    "ICICI":     {"months": 13, "include_current": True},   # LATEST 12 MONTH + CURRENT MONTH
    "AU":        {"months": 7,  "include_current": True},   # LATEST 6 MONTH + CURRENT MONTH
    "CHOLA":     {"months": 7,  "include_current": True},   # LATEST 6 MONTH + CURRENT MONTH
}
def ym(dt: datetime) -> tuple[int, int]:
    return (dt.year, dt.month)
def months_back_list(today: datetime, n: int, include_current: bool) -> list[tuple[int, int]]:
    """
    Return list of (year, month) tuples.
      - If include_current = True and n=6 on Aug 2025 => [Aug25, Jul25, Jun25, May25, Apr25, Mar25]
      - If include_current = False and n=3 on Aug 2025 => [Jul25, Jun25, May25]
    """
    out = []
    year, month = today.year, today.month
    offset = 0 if include_current else 1
    for i in range(offset, offset + n):
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1
        out.append((y, m))
    return out

date_pattern = re.compile(
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'  # 01/01/2024 or 01-01-2024
        r'|\d{1,2}[.]\d{1,2}[.]\d{2,4}'  # 01.01.2024
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,]?\s+\d{2,4}'  # 01 Jan 2024
        r'|\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-]\d{2,4}'  # 01-Jan-24
        r'|\d{4}[-/]\d{2}[-/]\d{2}'  # 2024-01-01
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'  # 2 Jan 2025
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # 31 Jan 31 Jan 2025
        flags=re.IGNORECASE
    )

possible_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
        "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%d %b, %Y", "%d %B, %Y",
        "%Y-%m-%d", "%d-%b-%y", "%d %b %y",
        "%d %B %Y", "%d %b %Y",
        "%d %b", "%d %B",
        "%d %b %Y", "%d %B %Y"
    ]
def extract_available_yearmonths(text: str) -> list[tuple[int, int]]:
    """Find unique (year, month) pairs present in the statement text (any common date format)."""
    seen = set()
    if not text:
        return []

    for m in date_pattern.finditer(text):
        token = m.group(0)
        # Try parsing token with multiple formats
        for fmt in possible_formats:
            try:
                dt = datetime.strptime(token, fmt)
                # Optional: coerce very old 2-digit years into 2000s if needed
                # if dt.year < 1970: dt = dt.replace(year=dt.year + 100)
                seen.add((dt.year, dt.month))
                break
            except ValueError:
                continue

    return sorted(seen, key=lambda t: (t[0], t[1]), reverse=True)

def allowed_months_for_bank(bank_name: str, today: datetime, available_yms: list[tuple[int,int]]) -> set[tuple[int,int]]:
    policy = BANK_MONTH_POLICY.get((bank_name or "").upper().strip(), {"months": 6, "include_current": True})
    n = max(0, int(policy["months"]))
    inc = bool(policy["include_current"])
    if n == 0 or not available_yms:
        return set()

    # 1) Today-anchored request
    requested = months_back_list(today, n, inc)             # list like [(2025,8), (2025,7), ...]
    avail_set = set(available_yms)
    inter = [ym for ym in requested if ym in avail_set]     # keep order

    # 2) If overlap is too small, fall back to statement-anchored months
    if len(inter) < n:
        # available_yms already sorted latest->oldest
        fallback = available_yms[:n]
        return set(fallback)

    return set(inter)


def line_yearmonth(line: str) -> tuple[int, int] | None:
    """Get (year, month) from a single transaction line (expects leading DD-MM-YYYY)."""
    m = re.search(r"^\s*(\d{2})-(\d{2})-(\d{4})", line or "")
    if not m: 
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        datetime(y, mo, d)
        return (y, mo)
    except ValueError:
        return None


def get_target_days_by_bank(bank_name: str):
    bank = (bank_name or "").upper().strip()
    spec = BANK_TO_DAYS.get(bank)
    if spec is None:
        return DEFAULT_DAYS
    if spec == "DAILY":
        return list(range(1, 32))  # 1..31
    # HERO or any N/A configured as empty -> choose what you want to do:
    if isinstance(spec, list) and len(spec) == 0:
        # Option A: return empty to skip extraction
        # return []
        # Option B (recommended): use a sensible default
        return DEFAULT_DAYS
    return spec


def is_transaction_line(line: str) -> bool:
    """
    Detects if a line contains a valid transaction based on:
    - Presence of date in supported formats (anywhere in the line)
    """

    DATE_REGEX = (
    r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'                                     # 1/1/2024 or 01-01-2024
    r'|^\d{1,2}[.]\d{1,2}[.]\d{2,4}'                                      # 01.01.2024
    r'|^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,]?\s+\d{2,4}'  # 01 Jan 2024 or 1 Jan, 24
    r'|^\d{1,2}[-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-]\d{2,4}'      # 01-Jan-24
    r'|^\d{4}[-/]\d{2}[-/]\d{2}'                                          # 2024-01-01
)

    # Normalize spacing
    line = re.sub(r'\s{2,}', ' ', line.strip())

    # Check if line contains a date pattern anywhere
    return re.search(DATE_REGEX, line, flags=re.IGNORECASE) is not None



from datetime import datetime
import re
from collections import defaultdict
from datetime import datetime
from collections import defaultdict
import re

from datetime import datetime
from collections import defaultdict
import re

def extract_last_transaction_on_or_before_day(full_text: str, target_day: int = 5, max_months: int = 13):
    """
    For each of the first max_months months:
    - If there are transactions on the target_day, pick the last one.
    - Otherwise, pick the latest transaction before the target_day.
    - Skip the month if no transaction on or before the target_day.
    """
    lines = full_text.splitlines()

    date_pattern = re.compile(
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'  # 01/01/2024 or 01-01-2024
        r'|\d{1,2}[.]\d{1,2}[.]\d{2,4}'  # 01.01.2024
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,]?\s+\d{2,4}'  # 01 Jan 2024
        r'|\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-]\d{2,4}'  # 01-Jan-24
        r'|\d{4}[-/]\d{2}[-/]\d{2}'  # 2024-01-01
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'  # 2 Jan 2025
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # 31 Jan 31 Jan 2025
        flags=re.IGNORECASE
    )

    possible_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
        "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%d %b, %Y", "%d %B, %Y",
        "%Y-%m-%d", "%d-%b-%y", "%d %b %y",
        "%d %B %Y", "%d %b %Y",
        "%d %b", "%d %B",
        "%d %b %Y", "%d %B %Y"
    ]

    date_line_map = []
    for line in lines:
        line = re.sub(r'\s{2,}', ' ', line.strip())
        match = date_pattern.search(line)
        if match:
            date_str = match.group(0)
            if len(date_str.split()) > 3:
                parts = date_str.split()
                date_str = " ".join(parts[-3:])
            for fmt in possible_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    date_line_map.append((date_obj, line))
                    break
                except ValueError:
                    continue

    grouped = defaultdict(list)
    for dt, line in date_line_map:
        grouped[(dt.year, dt.month)].append((dt, line))

    selected_lines = []
    for (year, month), entries in sorted(grouped.items(), key=lambda kv: kv[0], reverse=True)[:max_months]:
        try:
            target = datetime(year, month, target_day)
        except ValueError:
            continue

        valid_entries = [e for e in entries if e[0] <= target]
        if not valid_entries:
            continue

        valid_entries.sort(key=lambda x: x[0])
        last_entry = valid_entries[-1]
        selected_lines.append(last_entry[1])

    return selected_lines




def clean_pdf_text(full_text: str) -> str:
    """
    Extracts likely transaction rows from bank statement text and formats
    them for use with GPT. Removes noise, normalizes structure.
    """
    lines = full_text.splitlines()
    cleaned_rows = []

    for line in lines:
        if is_transaction_line(line):
            cleaned_rows.append(line.strip())

    if not cleaned_rows:
        return 0

    # Format for GPT - Markdown-style table
    output = "Below is the list of bank transactions. Each line contains a date, description, amount(s), and closing balance:\n\n"
    output += "\n".join(cleaned_rows)
    print("sayantan")
    return output




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return 'No file part', 400

        files = request.files.getlist('pdf_file')  # ✅ multiple files
        if not files or all(f.filename == '' for f in files):
            return 'No selected file', 400

        # --- Bank days from dropdown (copy list to avoid mutating globals)
        bank_name = request.form.get("bank_name", "")
        target_days = list(get_target_days_by_bank(bank_name))

        # --- Option B: only add custom days if provided (no phantom defaults)
        try:
            date_count = int(request.form.get('date_count', 0))
        except (TypeError, ValueError):
            date_count = 0

        custom_days = []
        for i in range(1, date_count + 1):
            raw = request.form.get(f'target_day_{i}')
            if not raw or raw.strip() == "":
                continue
            try:
                day = int(raw)
                if 1 <= day <= 31:
                    custom_days.append(day)
            except ValueError:
                pass

        # de-dupe DAYS while preserving order
        seen, merged_days = set(), []
        for d in target_days + custom_days:
            if d not in seen:
                seen.add(d)
                merged_days.append(d)
        target_days = merged_days

        print("Target days entered:", target_days)

        # --- Extract text from PDFs
        full_text = ""
        os.makedirs("temp", exist_ok=True)

        from werkzeug.utils import secure_filename
        import uuid

        for file in files:
            safe = secure_filename(file.filename) or f"upload-{uuid.uuid4().hex}.pdf"
            pdf_path = os.path.join("temp", f"{uuid.uuid4().hex}-{safe}")
            file.save(pdf_path)

            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        full_text += f"--- {safe} | Page {idx} ---\n{text}\n\n"

        # --- Clean and format
        formatted_text = clean_pdf_text(full_text)
        if isinstance(formatted_text, int) and formatted_text == 0:
            formatted_text = full_text


        print(formatted_text)
        # --- figure out allowed months for this bank & this PDF ---
        available_yms = extract_available_yearmonths(formatted_text)
        print("available yms",available_yms)
        today = datetime.today()
        allowed_yms = allowed_months_for_bank(bank_name, today, available_yms)
        print("allowed yms",allowed_yms)
        # Build the aggregate list by calling your existing function per day
        filtered_lines = []

        
        for day in target_days:  # Loop through each day in your list
            result = extract_last_transaction_on_or_before_day(formatted_text, target_day=day)
            # normalize to iterable
            
            if isinstance(result, list):
                candidates = result
            elif result is None:
                candidates = []
            else:
                candidates = [str(result)]
        
            for line in candidates:
                s = str(line).strip()
        
                # find a date near the start of the line; if not found there, search whole line
                m = date_pattern.search(s[:60]) or date_pattern.search(s)
                if not m:
                    continue
                date_token = m.group(0)
                dt_obj = None
                for fmt in possible_formats:
                    try:
                        dt_obj = datetime.strptime(date_token, fmt)
                        # Optional: coerce two-digit years if needed
                        # if dt_obj.year < 1970:
                        #     dt_obj = dt_obj.replace(year=dt_obj.year + 100)
                        break
                    except ValueError:
                        continue
                ym = (dt_obj.year, dt_obj.month)
                if not dt_obj:
                    continue
        
                # keep only lines whose (year, month) is allowed
                if ym in allowed_yms:
                    filtered_lines.append(s)
       
        # --- Convert to text and count lines
        if filtered_lines:
            filtered_text = "\n".join(filtered_lines)
            line_count = len(filtered_lines)
        else:
            filtered_text = ""
            line_count = 0
        print("number of line get counted",line_count)  
        # If nothing matches, skip GPT and render directly
        if line_count == 0:
            return render_template('test.html', filtered_text="No results in the selected month window.")
        '''
        # ✅ Convert to string for HTML display
        filtered_text = "\n".join(filtered_lines) if filtered_lines else "No results in the selected month window."
        '''
        #return render_template('test.html', filtered_text=filtered_text)
        
        
        gpt_result = ""
        try:
            prompt_user = f"""
You are given exactly {line_count} transaction lines. Process EVERY line. Do not skip, merge, or reorder.

Rules per line:
- Date = Dates can be in formats like DD-MM-YYYY, DD/MM/YY, DD Mon YYYY, or YYYY-MM-DD; always use the first date you see on the line. 
- Closing Balance = the VERY LAST numeric value in the line. Ignore any 'CR' or 'DR' that may follow it.
- Treat commas as thousands separators; keep two decimals in the output.

Output format:
1) For EACH of the {line_count} input lines, output exactly one line:
   Used date: DD-MM-YYYY, Closing Balance: ₹<amount>
   (Keep the same order as the input.)
2) After listing all lines, output two final lines:
   sum = ₹<sum of all balances>
   average_balance = ₹<sum divided by {line_count}>

Accuracy requirements:
- Use high precision for arithmetic; round ONLY at final display to two decimals.
- BEFORE returning, verify you produced exactly {line_count} result lines (not counting the final two summary lines). If not, correct yourself and re-run your extraction until the counts match.

Transactions:
{filtered_text}
""".strip()

            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a precise financial parsing assistant."},
                    {"role": "user", "content": prompt_user},
                ],
            )
            gpt_result = response.choices[0].message.content
        except Exception as e:
            gpt_result = f"❌ OpenAI Error: {str(e)}"
        
        
        
        return render_template(
            'result.html',
            formatted_text=formatted_text or "",
            filtered_text=filtered_text or "",
            gpt_result=gpt_result or ""
        )

    return render_template('upload.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)



'''
# 🔁 OpenAI API call
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,  # or gpt-4o if you have access
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial assistant that analyzes bank statements."
                    },
                    {
                "role": "user",
"content": f"""
Here is the extracted bank statement:
{filtered_text}

Each line contains a transaction that ends with a **closing balance**.
This balance is always the **last numeric value** in the line, and may or may not be followed by "CR" or "DR".

Rules:
1. For each line:
   - Extract the **transaction date** in the format `DD-MM-YYYY`. Use the **first date that appears in the line**, not any IDs or trailing ones.
   - Extract the **closing balance** as the **last numeric value** in the line, regardless of whether 'CR' or 'DR' appears.
2. Every line is already the final selected transaction for its month. Process **all lines**.
3. Output each line in this format:
   Used date: `DD-MM-YYYY`, Closing Balance: ₹<amount>
4. After listing all lines, compute and display the average of all extracted closing balances:
   **average_balance = ₹<average>**
5. If no valid transactions are found, output exactly:
   **average_balance = ₹0**

⚠️ Output only the formatted lines and the average. No headings, no explanations, no markdown, no bullet points.
"""
            }
                ]
            )
            gpt_result = response.choices[0].message.content
        except Exception as e:
            gpt_result = f"❌ OpenAI Error: {str(e)}"
        
        
'''
