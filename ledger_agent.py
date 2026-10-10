"""
Ledger Agent: answers plain-English questions about the Acme Appliances order ledger.

Run locally:  uvicorn ledger_agent:app --port 8000
Env vars:     LEDGER_URL   the full start URL from the assignment (keep the quotes in your shell!)
              LLM_API_KEY, LLM_BASE_URL (OpenAI-compatible), LLM_MODEL
"""
import json, os, re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

LEDGER_URL = os.environ["LEDGER_URL"]
llm = OpenAI(api_key=os.environ["LLM_API_KEY"],
             base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
             timeout=10)
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


# ---------- STEP 1: download everything ----------
def get_json(url):
    return requests.get(url, timeout=20).json()


def fetch_orders(first_url):
    """Orders come in pages (&page=1..N). Page 1 tells us N in the field 'pages'."""
    pages = get_json(first_url)["pages"]

    def one(n):
        return get_json(re.sub(r"&page=\d+", f"&page={n}", first_url))["orders"]

    with ThreadPoolExecutor(8) as ex:
        chunks = list(ex.map(one, range(1, pages + 1)))
    return [row for chunk in chunks for row in chunk]


# ---------- STEP 2: clean it, following the API's three notes ----------
def load_df():
    root = get_json(LEDGER_URL)
    links = root["links"]
    rates = get_json(links["rates"])["usd_per_unit"]      # {"USD":1,"EUR":1.12,"INR":0.01103}
    df = pd.DataFrame(fetch_orders(links["orders"]))

    # Note 2: an order id can repeat; the row with the latest updated_at is current.
    # (timestamps mix 'Z' and '+05:30', so parse everything as UTC before comparing)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("updated_at").drop_duplicates("id", keep="last").copy()

    # Note 1: business dates are Asia/Kolkata.
    local = df["created_at"].dt.tz_convert("Asia/Kolkata")
    df["date"] = local.dt.strftime("%Y-%m-%d")
    df["month"] = local.dt.strftime("%Y-%m")

    # Money: convert every amount to USD (unrounded; round only the final answer).
    df["amount_usd"] = df["amount"] * df["currency"].map(rates)

    # Note 3: only status == 'paid' is revenue (others: 'refunded', 'void').
    df = df.drop(columns=["created_at", "updated_at"])
    return df.reset_index(drop=True)


DF = load_df()
SCHEMA = {
    "columns": list(DF.columns),
    "sample_rows": DF.head(3).to_dict("records"),
    "distinct_values": {c: sorted(DF[c].dropna().unique().tolist())[:40]
                        for c in ["region", "product", "status", "currency", "month"]},
}

PROMPT = f"""You turn a question about a sales ledger into a JSON query. Reply with JSON only.
Table info: {json.dumps(SCHEMA, default=str)}
Output format:
{{"op": "sum|count|mean|max|min|nunique",
  "column": "<column to aggregate, or null for count>",
  "filters": [{{"column": "...", "op": "==|!=|>|>=|<|<=|contains", "value": ...}}],
  "group_by": "<column or null>", "top_n": <int or null>}}
Rules:
- Money is in the column "amount_usd" (already converted to USD). Always use it for money.
- "Revenue"/"sales" = sum of amount_usd with filter status == "paid".
- "Refunds": status == "refunded" (sum amount_usd for the amount, count for how many).
- Months: filter the "month" column with "==" (e.g. "2026-03"). Single days: the "date" column ("2026-03-15").
- Orders are rows; "customers" means distinct values of "customer" (use nunique).
- "Which/top/best product/region/customer": set group_by, op "sum" on amount_usd (or "count"), top_n 1.
- Use region/product spellings exactly as in the distinct_values."""


def ask_llm(question):
    r = llm.chat.completions.create(
        model=MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": question}])
    return json.loads(r.choices[0].message.content)


# ---------- STEP 3: do the maths with pandas, never with the LLM ----------
def run(spec):
    d = DF
    for f in spec.get("filters") or []:
        c, op, v = f["column"], f["op"], f["value"]
        if op == "==": d = d[d[c] == v]
        elif op == "!=": d = d[d[c] != v]
        elif op == ">": d = d[d[c] > v]
        elif op == ">=": d = d[d[c] >= v]
        elif op == "<": d = d[d[c] < v]
        elif op == "<=": d = d[d[c] <= v]
        elif op == "contains": d = d[d[c].astype(str).str.contains(str(v), case=False)]
    col, op, g = spec.get("column"), spec["op"], spec.get("group_by")
    if g:
        if op == "count" or not col:
            grp = d.groupby(g).size()
        else:
            grp = getattr(d.groupby(g)[col], op if op != "nunique" else "nunique")()
        grp = grp.sort_values(ascending=False)
        n = spec.get("top_n") or 1
        return grp.index[0] if n == 1 else grp.head(n).index.tolist()
    if op == "count": return int(len(d))
    if op == "nunique": return int(d[col].nunique())
    val = getattr(d[col], op)()
    return round(float(val), 2)


# ---------- STEP 4: the public endpoint ----------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Q(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok", "orders_loaded": int(len(DF))}


@app.post("/")
@app.post("/{path:path}")      # accept POST at any path, so the submitted URL can't 404
def answer(q: Q):
    try:
        return {"answer": run(ask_llm(q.question))}
    except Exception as e:
        return {"answer": None, "error": str(e)}