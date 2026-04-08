import json
import asyncio
import os
import time
from typing import Dict, List, Any, Set

import aiohttp
import mysql.connector

from llm_service_vllm import RecursiveExtractionService

# ── Database configuration ─────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "192.168.168.203"),
    "port":     int(os.getenv("DB_PORT", "3319")),
    "user":     os.getenv("DB_USER", "mahendra"),
    "password": os.getenv("DB_PASSWORD", "safertek"),
    "database": os.getenv("DB_NAME", "uwareone"),
}
DB_SCHEMA = os.getenv("DB_SCHEMA", "uwareone")
# ──────────────────────────────────────────────────────────────────────────

# ── SQL generation config ──────────────────────────────────────────────────
SQL_API_URL = os.getenv("SQL_API_URL", "http://192.168.168.211:8000/v1/chat/completions")
SQL_MODEL   = os.getenv("SQL_MODEL", "")   # leave blank to let the server pick

SQL_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema (as CREATE TABLE statements) and a user question, "
    "write a single valid MySQL SELECT statement that answers the question. "
    "Return only the raw SQL — no markdown, no explanation, no code fences."
)
# ──────────────────────────────────────────────────────────────────────────


# ── Helpers ────────────────────────────────────────────────────────────────

def load_data(path: str = "updated_data.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_step(n: int, label: str):
    print(f"\n{'═' * 70}")
    print(f"  STEP {n} — {label}")
    print(f"{'═' * 70}")


def _print_timing(n: int, elapsed: float):
    print(f"\n  ✓ Step {n} done in {elapsed:.2f}s")


def print_results(question: str, matched_items: List[Dict[str, Any]]):
    print(f"\nQuestion : {question}")
    if not matched_items:
        print("No matching items found.")
        return
    print(json.dumps(matched_items, indent=2))


# ── Schema building ────────────────────────────────────────────────────────

def _get_primary_keys(cursor, table_name: str) -> List[str]:
    """Return uppercase PK column names for table_name."""
    cursor.execute(
        """
        SELECT DISTINCT kcu.COLUMN_NAME
        FROM information_schema.TABLE_CONSTRAINTS tco
        JOIN information_schema.KEY_COLUMN_USAGE kcu
            ON  kcu.CONSTRAINT_NAME   = tco.CONSTRAINT_NAME
            AND kcu.CONSTRAINT_SCHEMA = tco.CONSTRAINT_SCHEMA
        WHERE tco.CONSTRAINT_TYPE = 'PRIMARY KEY'
          AND kcu.TABLE_NAME   = %s
          AND kcu.TABLE_SCHEMA = %s
        """,
        (table_name.lower(), DB_SCHEMA.lower()),
    )
    return [row[0].upper() for row in cursor.fetchall()]


def _get_column_metadata(cursor, table_name: str, columns: List[str]) -> List[Dict]:
    """
    Query INFORMATION_SCHEMA for type/constraint/FK info for the given columns.
    Deduplicates by column_name, keeping PRIMARY KEY > FOREIGN KEY > null.
    Not called for _adtl_info tables.
    """
    if not columns:
        return []

    tab_num      = table_name.upper().replace("_", "")   # SALES_ORDER → SALESORDER
    placeholders = ", ".join(["%s"] * len(columns))

    query = f"""
        SELECT
            c.TABLE_SCHEMA               AS database_name,
            c.TABLE_NAME                 AS table_name,
            c.COLUMN_NAME                AS column_name,
            c.COLUMN_TYPE                AS column_type,
            tc.CONSTRAINT_TYPE           AS key_type,
            k.REFERENCED_TABLE_NAME      AS referenced_table,
            k.REFERENCED_COLUMN_NAME     AS referenced_pk_column,
            CFF.REFERENCE_COLUMN_NAME    AS referenced_lookup_column
        FROM INFORMATION_SCHEMA.COLUMNS c
        LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
            ON  c.TABLE_SCHEMA = k.TABLE_SCHEMA
            AND c.TABLE_NAME   = k.TABLE_NAME
            AND c.COLUMN_NAME  = k.COLUMN_NAME
        LEFT JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            ON  k.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
            AND k.TABLE_NAME        = tc.TABLE_NAME
            AND k.CONSTRAINT_NAME   = tc.CONSTRAINT_NAME
        LEFT JOIN COR_FUNCTION_FIELD CFF
            ON  CFF.REFERENCE_TABLE_NAME     = k.REFERENCED_TABLE_NAME
            AND CFF.REFERENCE_PK_COLUMN_NAME = k.REFERENCED_COLUMN_NAME
            AND CFF.COLUMN_NAME              = c.COLUMN_NAME
            AND CFF.TAB_NUM                  = %s
        WHERE
            c.TABLE_SCHEMA = %s
            AND c.TABLE_NAME   = %s
            AND c.COLUMN_NAME  IN ({placeholders})
    """
    params = (
        [tab_num, DB_SCHEMA.lower(), table_name.lower()]
        + [c.upper() for c in columns]
    )
    cursor.execute(query, params)
    col_names = [desc[0].lower() for desc in cursor.description]
    raw_rows  = [dict(zip(col_names, row)) for row in cursor.fetchall()]

    # Deduplicate — keep highest-priority constraint per column
    priority = {"PRIMARY KEY": 2, "FOREIGN KEY": 1, None: 0}
    best: Dict[str, Dict] = {}
    for row in raw_rows:
        col = (row.get("column_name") or "").upper()
        if col not in best or priority.get(row["key_type"], 0) > priority.get(best[col]["key_type"], 0):
            best[col] = row

    return list(best.values())


def _synthetic_adtl_rows(table_name: str, columns: Set[str], pk_set: Set[str]) -> List[Dict]:
    """
    For *_adtl_info tables, FIELDXXX_VALUE columns have no meaningful DB
    metadata so we skip the DB query and return synthetic rows with VARCHAR(35).
    """
    rows = []
    for col in columns:
        rows.append({
            "database_name":          DB_SCHEMA,
            "table_name":             table_name,
            "column_name":            col,
            "column_type":            "varchar(35)",
            "key_type":               "PRIMARY KEY" if col in pk_set else None,
            "referenced_table":       None,
            "referenced_pk_column":   None,
            "referenced_lookup_column": None,
        })
    return rows


def build_schema(matched_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1. Register every matched table; add leaf columns for non-node items.
    2. Fetch and merge primary keys for every table.
    3. Fetch column metadata:
         - Regular tables  → INFORMATION_SCHEMA query.
         - *_adtl_info     → synthetic VARCHAR(35) rows (FIELDXXX_VALUE columns
                             have no useful metadata in the DB).
    4. Collect FK-referenced lookup tables and fetch their metadata too.
    5. Assemble and return the schema dict.
    """
    # ── 1. Collect tables + leaf columns ───────────────────────────────────
    table_columns: Dict[str, Set[str]] = {}
    for item in matched_items:
        tbl = (item.get("table") or "").strip().lower()
        if not tbl:
            continue
        table_columns.setdefault(tbl, set())
        col = (item.get("column") or "").strip()
        if col:
            table_columns[tbl].add(col.upper())

    if not table_columns:
        return {}

    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        # ── 2. Merge primary keys ───────────────────────────────────────────
        table_pks: Dict[str, Set[str]] = {}
        for tbl in list(table_columns):
            pks = _get_primary_keys(cursor, tbl)
            table_pks[tbl] = set(pks)
            table_columns[tbl].update(pks)

        # ── 3. Fetch column metadata ────────────────────────────────────────
        all_metadata: List[Dict] = []
        for tbl, cols in table_columns.items():
            if tbl.endswith("_adtl_info"):
                # Generic FIELDXXX_VALUE columns — use synthetic defaults
                rows = _synthetic_adtl_rows(tbl, cols, table_pks.get(tbl, set()))
            else:
                rows = _get_column_metadata(cursor, tbl, list(cols))
            all_metadata.extend(rows)

        # ── 4. Collect FK-referenced lookup tables ──────────────────────────
        ref_sets: Dict[str, Set[str]] = {}
        for row in all_metadata:
            rt = (row.get("referenced_table") or "").lower()
            rp = (row.get("referenced_pk_column") or "").upper()
            rl = (row.get("referenced_lookup_column") or "").upper()
            if rt and rp and rl and rt not in table_columns:
                ref_sets.setdefault(rt, set()).update({rp, rl})

        for rt, cols in ref_sets.items():
            pks = _get_primary_keys(cursor, rt)
            cols.update(pks)
            rows = _get_column_metadata(cursor, rt, list(cols))
            all_metadata.extend(rows)

        # ── 5. Assemble schema ──────────────────────────────────────────────
        schema: Dict[str, Any] = {}
        for row in all_metadata:
            tbl = (row.get("table_name") or "").upper()
            col = (row.get("column_name") or "").upper()
            if not tbl or not col:
                continue
            schema.setdefault(tbl, {"columns": {}})

            entry: Dict[str, Any] = {
                "type":     row.get("column_type"),
                "key_type": row.get("key_type"),
            }
            rt = (row.get("referenced_table") or "").upper() or None
            rp = (row.get("referenced_pk_column") or "").upper() or None
            rl = (row.get("referenced_lookup_column") or "").upper() or None
            if rt and rp and rl:
                entry["references"] = {
                    "table":         rt,
                    "pk_column":     rp,
                    "lookup_column": rl,
                }
            schema[tbl]["columns"][col] = entry

        return schema

    finally:
        cursor.close()
        conn.close()


# ── DDL generation ─────────────────────────────────────────────────────────

def schema_to_ddl(schema: Dict[str, Any]) -> List[str]:
    """
    Convert the schema dict into CREATE TABLE DDL statements.
    - PRIMARY KEY columns → table-level PRIMARY KEY constraint.
    - FOREIGN KEY columns → FOREIGN KEY ... REFERENCES ... (no lookup_column).
    """
    statements: List[str] = []

    for table_name, table_info in schema.items():
        columns = table_info.get("columns", {})
        col_lines: List[str] = []
        pk_cols:   List[str] = []
        fk_lines:  List[str] = []

        for col_name, col_info in columns.items():
            col_type = (col_info.get("type") or "TEXT").upper()
            col_lines.append(f"    {col_name} {col_type}")

            key_type = col_info.get("key_type")
            if key_type == "PRIMARY KEY":
                pk_cols.append(col_name)
            elif key_type == "FOREIGN KEY":
                refs      = col_info.get("references", {})
                ref_table = refs.get("table", "")
                ref_pk    = refs.get("pk_column", "")
                if ref_table and ref_pk:
                    fk_lines.append(
                        f"    FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_pk})"
                    )

        constraint_lines: List[str] = []
        if pk_cols:
            constraint_lines.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
        constraint_lines.extend(fk_lines)

        body = ",\n".join(col_lines + constraint_lines)
        statements.append(f"CREATE TABLE {table_name} (\n{body}\n);")

    return statements


# ── SQL generation ─────────────────────────────────────────────────────────

def _build_context(matched_items: List[Dict[str, Any]]) -> str:
    """
    Format matched items (nodes + leaves) as a business context block.
    Nodes describe sections/tables; leaves describe specific columns.
    Items without a description are still included for structural context
    — the description part is simply omitted so the model knows the
    table/field was intentionally selected.
    """
    lines: List[str] = []
    for item in matched_items:
        field      = item.get("field", "")
        table      = item.get("table", "")
        column     = item.get("column", "")
        ftype      = item.get("field_type", "")
        desc       = (item.get("description") or "").strip()

        if ftype == "node":
            short_desc = (item.get("short_description") or "").strip()
            display_desc = short_desc or desc   # prefer short, fall back to long
            line = f"[section] {field} (table: {table.upper()})"
            if display_desc:
                line += f": {display_desc}"
        else:
            ref = f"{table.upper()}.{column.upper()}" if column else table.upper()
            line = f"[field]   {field} → {ref}"
            if desc:
                line += f": {desc}"

        lines.append(line)

    return "\n".join(lines)


async def generate_sql(
    question: str,
    ddl_statements: List[str],
    matched_items: List[Dict[str, Any]],
) -> str:
    """Send DDL schema + business context + question to the LLM; return SQL."""
    schema_text  = "\n\n".join(ddl_statements)
    context_text = _build_context(matched_items)

    user_msg = (
        f"Database schema:\n{schema_text}\n\n"
        f"Business context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens":  512,
    }
    if SQL_MODEL:
        payload["model"] = SQL_MODEL

    async with aiohttp.ClientSession() as session:
        async with session.post(SQL_API_URL, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

    raw = data["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if the model ignores the instruction
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    return raw


# ── Main flow ──────────────────────────────────────────────────────────────

async def run(question: str, data: List[Dict[str, Any]]) -> tuple:
    t_total = time.perf_counter()

    # ── STEP 1: Field Extraction ───────────────────────────────────────────
    _print_step(1, "Field Extraction")
    t0 = time.perf_counter()
    service = RecursiveExtractionService()
    print(f"Top-level entities: {len(data)}")
    matched_items, summary = await service.extract_from_new_data(question, data)
    _print_timing(1, time.perf_counter() - t0)
    print_results(question, matched_items)

    if not matched_items:
        print("No matched items — stopping.")
        return matched_items, summary, {}, [], ""

    # ── STEP 2: JSON Schema Building ───────────────────────────────────────
    _print_step(2, "JSON Schema Building")
    t0 = time.perf_counter()
    schema = build_schema(matched_items)
    _print_timing(2, time.perf_counter() - t0)
    print(f"  Tables: {len(schema)} | Columns: {sum(len(v['columns']) for v in schema.values())}")
    print(json.dumps(schema, indent=2, default=str))

    # ── STEP 3: DDL Building ───────────────────────────────────────────────
    _print_step(3, "DDL Building")
    t0 = time.perf_counter()
    ddl_statements = schema_to_ddl(schema)
    _print_timing(3, time.perf_counter() - t0)
    print()
    print("\n\n".join(ddl_statements))

    # ── STEP 4: SQL Generation (LLM Request) ───────────────────────────────
    _print_step(4, "SQL Generation — LLM Request")
    print(f"  Endpoint : {SQL_API_URL}")
    print(f"  Tables   : {len(ddl_statements)}")
    t0 = time.perf_counter()
    sql = await generate_sql(question, ddl_statements, matched_items)
    _print_timing(4, time.perf_counter() - t0)

    # ── STEP 5: Result ─────────────────────────────────────────────────────
    _print_step(5, "Result")
    print(f"\n  {sql}")

    print(f"\n{'═' * 70}")
    print(f"  Total time: {time.perf_counter() - t_total:.2f}s")
    print(f"{'═' * 70}")

    return matched_items, summary, schema, ddl_statements, sql


def main():
    data = load_data()

    question = input("Enter your question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    asyncio.run(run(question, data))


if __name__ == "__main__":
    main()
