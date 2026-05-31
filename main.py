"""
main.py -- Schema linking inference.

Uses the LoRA adapter produced by train.py / exp1.py / exp2.py / exp3.py when the
adapter directory contains adapter files; falls back to the keyword-matching
baseline otherwise.

CLI:
    python main.py --input  validation_input.json \\
                   --output predictions.json \\
                   [--schemas_dir ./schemas] \\
                   [--adapter_dir  ./adapter] \\
                   [--base_model   Qwen/Qwen2.5-1.5B-Instruct] \\
                   [--schema_format typed]
"""

import argparse
import json
import os
import re

# Must match train.py exactly.
BASE_MODEL     = 'Qwen/Qwen2.5-1.5B-Instruct'
ADAPTER_DIR    = './adapter'
MAX_NEW_TOKENS = 512

# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a database assistant. "
    "Given a database schema and a natural language question, output the schema links "
    "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names that appear in the given schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

SYSTEM_PROMPT_TYPED = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type) and a natural language "
    "question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

SYSTEM_PROMPT_ABBREV = (
    "You are a database assistant. "
    "Given a database schema (column types: T=text, N=number, R=real, TM=time, B=boolean) "
    "and a natural language question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

SYSTEM_PROMPT_QHINT = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type), a list of key terms "
    "from the question, and the question itself, output the schema links as a JSON "
    "object: {\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

_FEWSHOT_PREFIX = (
    "Example:\n"
    "Schema:\n"
    "  department(dept_id, dept_name, budget)\n"
    "  employee(emp_id, name, dept_id, salary, hire_date)\n\n"
    "Question: What is the name and salary of each employee?\n"
    'Answer: {"employee": ["name", "salary"]}\n\n'
    "---\n\n"
    "Now answer:\n"
)

_FEWSHOT_TYPED_PREFIX = (
    "Example:\n"
    "Schema:\n"
    "  department(dept_id:number, dept_name:text, budget:real)\n"
    "  employee(emp_id:number, name:text, dept_id:number, salary:real, hire_date:time)\n\n"
    "Question: What is the name and salary of each employee?\n"
    'Answer: {"employee": ["name", "salary"]}\n\n'
    "---\n\n"
    "Now answer:\n"
)

# Abbreviated type map (must match exp2.py)
_TYPE_ABBREV = {
    'text':    'T',
    'number':  'N',
    'real':    'R',
    'time':    'TM',
    'boolean': 'B',
    'blob':    'BL',
    'others':  'O',
}

# ── Schema helpers ─────────────────────────────────────────────────────────────

def load_schema(db_id: str, schemas_dir: str) -> dict:
    """Return {table: [columns]} from a Spider-format schema file."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        if tidx == -1:
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema


def load_col_types(db_id: str, schemas_dir: str) -> dict:
    """Return {table: {col_name: col_type}} from a Spider-format schema file."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    col_info    = s['column_names_original']
    col_types   = s.get('column_types', [])
    types = {t: {} for t in table_names}
    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        ctype = col_types[i] if i < len(col_types) else ''
        types[table_names[tidx]][cname] = ctype
    return types


def _split_id(name: str) -> set:
    """Split a camelCase / snake_case identifier into lowercase sub-words."""
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return {t for t in re.split(r'[^a-zA-Z0-9]+', s.lower()) if len(t) >= 2}


def serialize_schema(schema: dict, fmt: str = 'compact',
                     col_types: dict = None,
                     question: str = '') -> str:
    """Serialize schema for the prompt. fmt must match the format used at training time.

    Supported formats:
      compact        — plain col names, no types  (original baseline)
      fewshot        — compact + one-shot example
      typed          — col:type annotations        (exp1 best)
      fewshot_typed  — typed + one-shot example
      sql_ddl        — CREATE TABLE syntax
      markdown       — markdown headers
      schema_abbrev  — abbreviated types (T/N/R/TM)         [exp2]
      schema_sorted  — typed + alphabetical ordering         [exp2]
      schema_top10   — typed + top-10 table pruning          [exp2]
      sorted_abbrev  — sorted + abbreviated types            [exp3]
      question_hint  — typed + "Key terms:" hint line        [exp3]
      col_filtered   — typed, cols filtered by question kws  [exp3]
    """
    # ── Legacy / non-compact formats ──
    if fmt == 'sql_ddl':
        lines = [f"CREATE TABLE {t} ({', '.join(c)})" if c else f"CREATE TABLE {t} ()"
                 for t, c in schema.items()]
        return "Schema:\n" + "\n".join(lines)

    if fmt == 'markdown':
        parts = []
        for t, c in schema.items():
            parts.append(f"### {t}")
            if c:
                parts.append("Columns: " + ", ".join(c))
        return "Database Schema:\n" + "\n".join(parts)

    # ── Exp2: abbreviated types ──
    if fmt == 'schema_abbrev':
        lines = []
        for table, cols in schema.items():
            t_types = col_types.get(table, {}) if col_types else {}
            col_strs = []
            for c in cols:
                raw_type = t_types.get(c, '')
                abbrev   = _TYPE_ABBREV.get(raw_type.lower(),
                               raw_type[:2].upper() if raw_type else '')
                col_strs.append(f"{c}:{abbrev}" if abbrev else c)
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp2: alphabetically sorted typed schema ──
    if fmt == 'schema_sorted':
        lines = []
        for table in sorted(schema.keys()):
            cols    = sorted(schema[table])
            t_types = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp2: top-10 pruned typed schema ──
    if fmt == 'schema_top10':
        pruned = prune_schema(question, schema, max_tables=10)
        lines  = []
        for table, cols in pruned.items():
            t_types  = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp3: sorted + abbreviated types ──
    if fmt == 'sorted_abbrev':
        lines = []
        for table in sorted(schema.keys()):
            cols    = sorted(schema[table])
            t_types = col_types.get(table, {}) if col_types else {}
            col_strs = []
            for c in cols:
                raw_type = t_types.get(c, '')
                abbrev   = _TYPE_ABBREV.get(raw_type.lower(),
                               raw_type[:2].upper() if raw_type else '')
                col_strs.append(f"{c}:{abbrev}" if abbrev else c)
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp3: typed schema + key terms hint line (question embedded in output) ──
    if fmt == 'question_hint':
        lines = []
        for table, cols in schema.items():
            t_types  = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        schema_text = "Schema:\n" + "\n".join(lines)
        # extract key terms: schema identifiers that overlap with question keywords
        q_words = set(re.findall(r'[a-z]{2,}', question.lower()))
        matched, seen = [], set()
        for table, cols in schema.items():
            if _split_id(table) & q_words and table not in seen:
                matched.append(table); seen.add(table)
            for col in cols:
                if _split_id(col) & q_words and col not in seen:
                    matched.append(col); seen.add(col)
        hint_line = f"Key terms: {', '.join(matched[:10])}" if matched else ""
        if hint_line:
            return f"{schema_text}\n{hint_line}\nQuestion: {question}"
        return f"{schema_text}\n\nQuestion: {question}"

    # ── Exp3: typed schema with per-table column filtering ──
    if fmt == 'col_filtered':
        MAX_COLS = 8
        q_words  = set(re.findall(r'[a-z]{2,}', question.lower()))
        lines = []
        for table, cols in schema.items():
            relevant = [c for c in cols if _split_id(c) & q_words]
            # pad with original-order cols if under MAX_COLS
            keep = relevant[:]
            for c in cols:
                if c not in set(keep):
                    keep.append(c)
                if len(keep) >= MAX_COLS:
                    break
            t_types  = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in keep]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp4: reverse-sorted typed schema ──
    if fmt == 'sorted_desc':
        lines = []
        for table in sorted(schema.keys(), reverse=True):
            cols    = sorted(schema[table], reverse=True)
            t_types = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp4: sorted schema, structured two-line output at inference ──
    if fmt == 'col_hint_output':
        lines = []
        for table in sorted(schema.keys()):
            cols    = sorted(schema[table])
            t_types = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Compact family: compact | fewshot | typed | fewshot_typed ──
    use_types   = fmt in ('typed', 'fewshot_typed')
    use_fewshot = fmt in ('fewshot', 'fewshot_typed')

    lines = []
    for t, cols in schema.items():
        if use_types and col_types:
            t_types  = col_types.get(t, {})
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
        else:
            col_strs = cols
        lines.append(f"  {t}({', '.join(col_strs)})" if col_strs else f"  {t}")

    schema_text = "Schema:\n" + "\n".join(lines)
    if use_fewshot:
        prefix = _FEWSHOT_TYPED_PREFIX if use_types else _FEWSHOT_PREFIX
        return prefix + schema_text
    return schema_text


def filter_against_schema(links: dict, schema: dict) -> dict:
    """Drop hallucinated tables/columns and restore canonical casing."""
    lc_tables = {t.lower(): t for t in schema}
    lc_cols   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    result = {}
    for table, cols in links.items():
        canonical_t = lc_tables.get(str(table).lower())
        if canonical_t is None:
            continue
        if not isinstance(cols, list):
            result[canonical_t] = []
            continue
        cols_map = lc_cols.get(canonical_t, {})
        result[canonical_t] = [cols_map[str(c).lower()]
                               for c in cols if str(c).lower() in cols_map]
    return result


# ── Model-based predictor ─────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Extract a JSON dict from raw model output; fall back to {} on failure."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _get_system_prompt(schema_format: str) -> str:
    """Return the correct system prompt for a given schema format."""
    if schema_format in ('schema_abbrev', 'sorted_abbrev'):
        return SYSTEM_PROMPT_ABBREV
    if schema_format == 'question_hint':
        return SYSTEM_PROMPT_QHINT
    if schema_format == 'col_hint_output':
        return (
            "You are a database assistant. "
            "Given a database schema (column types shown as col:type) and a natural language "
            "question, first output the relevant table names as a JSON array on one line, "
            "then output the full schema links as a JSON object on the next line: "
            "{\"TableName\": [\"col1\", \"col2\"]}. "
            "Use ONLY table and column names (without the :type suffix) from the schema. "
            "Include only the tables and columns needed to answer the question. "
            "Output exactly two lines: the Tables array, then the JSON object. No extra text."
        )
    if schema_format in ('typed', 'fewshot_typed', 'schema_sorted', 'schema_top10',
                         'col_filtered', 'sorted_desc', 'sorted_5ep'):
        return SYSTEM_PROMPT_TYPED
    return SYSTEM_PROMPT


class ModelPredictor:
    """Wraps a PEFT-adapted causal LM for schema-link prediction."""

    def __init__(self, base_model: str, adapter_dir: str, schema_format: str = 'compact'):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_cfg_path = os.path.join(adapter_dir, 'adapter_config.json')
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path) as _f:
                _cfg = json.load(_f)
            recorded = _cfg.get('base_model_name_or_path')
            if recorded and recorded != base_model:
                print(f"  [info] base_model overridden by adapter_config: {recorded}")
                base_model = recorded

        tok_src = (adapter_dir
                   if os.path.exists(os.path.join(adapter_dir, 'tokenizer_config.json'))
                   else base_model)
        print(f"  tokenizer : {tok_src}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  base model: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype="auto")

        print(f"  adapter   : {adapter_dir}")
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        self._torch = torch
        self.schema_format = schema_format

    def predict(self, question: str, schema: dict, debug: bool = False,
                col_types: dict = None) -> dict:
        sys_prompt  = _get_system_prompt(self.schema_format)
        schema_text = serialize_schema(schema, self.schema_format,
                                       col_types=col_types, question=question)
        # question_hint embeds the question inside schema_text already
        if self.schema_format == 'question_hint':
            user_content = schema_text
        else:
            user_content = f"{schema_text}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with self._torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(out[0][input_ids.shape[-1]:],
                                    skip_special_tokens=True)
        if debug:
            print(f"    [raw] {repr(raw[:500])}")
        # col_hint_output produces two lines: "Tables: [...]" then "{...json...}"
        # strip the Tables line and parse only the JSON line
        if self.schema_format == 'col_hint_output':
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            json_lines = [l for l in lines if l.startswith('{')]
            raw = json_lines[0] if json_lines else raw
        return _parse_json(raw)


# ── Keyword-matching fallback ─────────────────────────────────────────────────

def _keyword_predict(question: str, schema: dict) -> dict:
    q = set(re.findall(r'[a-z]{2,}', question.lower()))
    result = {}
    for table, cols in schema.items():
        matched = [c for c in cols if _split_id(c) & q]
        if (_split_id(table) & q) or matched:
            result[table] = matched
    return result


# ── Schema pruning ────────────────────────────────────────────────────────────

MAX_SCHEMA_TABLES = 20   # default cap for inference (compact/typed/etc.)

def prune_schema(question: str, schema: dict,
                 max_tables: int = MAX_SCHEMA_TABLES) -> dict:
    """Rank tables by keyword overlap; keep top max_tables."""
    if len(schema) <= max_tables:
        return schema
    q = set(re.findall(r'[a-z]{2,}', question.lower()))
    def score(t):
        return len(_split_id(t) & q) * 2 + sum(1 for c in schema[t] if _split_id(c) & q)
    ranked = sorted(schema, key=score, reverse=True)
    return {t: schema[t] for t in ranked[:max_tables]}


# ── Adapter detection ─────────────────────────────────────────────────────────

def _adapter_ready(adapter_dir: str) -> bool:
    if not os.path.isdir(adapter_dir):
        return False
    return any(f.endswith(('.safetensors', '.bin'))
               for f in os.listdir(adapter_dir))


# ── Per-question entry point ──────────────────────────────────────────────────

def predict_schema_links(question: str, db_id: str, schemas_dir: str,
                         predictor=None, debug: bool = False) -> dict:
    schema = load_schema(db_id, schemas_dir)

    if predictor is not None:
        col_types = None
        if predictor.schema_format in ('typed', 'fewshot_typed',
                                       'schema_abbrev', 'schema_sorted', 'schema_top10',
                                       'sorted_abbrev', 'question_hint', 'col_filtered',
                                       'sorted_desc', 'col_hint_output', 'sorted_5ep'):
            col_types = load_col_types(db_id, schemas_dir)

        if predictor.schema_format not in ('schema_top10', 'question_hint'):
            pruned = prune_schema(question, schema, max_tables=MAX_SCHEMA_TABLES)
        else:
            pruned = schema   # pass full schema; handled in serialize_schema

        raw = predictor.predict(question, pruned, debug=debug, col_types=col_types)
        return filter_against_schema(raw, schema)   # validate against full schema

    return _keyword_predict(question, schema)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',       required=True)
    ap.add_argument('--output',      required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    ap.add_argument('--adapter_dir', default=ADAPTER_DIR)
    ap.add_argument('--base_model',    default=BASE_MODEL)
    ap.add_argument('--schema_format', default='compact',
                    choices=['compact', 'sql_ddl', 'markdown',
                             'fewshot', 'typed', 'fewshot_typed',
                             'schema_abbrev', 'schema_sorted', 'schema_top10',
                             'sorted_abbrev', 'question_hint', 'col_filtered',
                             'sorted_desc', 'col_hint_output', 'sorted_5ep'],
                    help='Schema serialization format — must match what was used at training time')
    ap.add_argument('--debug', type=int, default=0, metavar='N',
                    help='Print raw model output for the first N predictions (0 = off)')
    args = ap.parse_args()

    if _adapter_ready(args.adapter_dir):
        print(f"[mode] fine-tuned model  (schema_format={args.schema_format})")
        predictor = ModelPredictor(args.base_model, args.adapter_dir, args.schema_format)
    else:
        print(f"[mode] keyword-matching baseline  "
              f"(no adapter at '{args.adapter_dir}' — run train.py first)")
        predictor = None

    with open(args.input) as f:
        items = json.load(f)

    preds = []
    for i, it in enumerate(items, 1):
        debug_this = args.debug > 0 and i <= args.debug
        if debug_this:
            print(f"\n[debug q{it['question_id']}] {it['question']}")
        links = predict_schema_links(
            it['question'], it['db_id'], args.schemas_dir, predictor,
            debug=debug_this)
        if debug_this:
            print(f"    [parsed] {links}")
        preds.append({'question_id': it['question_id'], 'schema_links': links})
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)} predicted")

    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions → {args.output}")


if __name__ == '__main__':
    main()