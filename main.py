"""
main.py -- Schema linking inference.

Uses the LoRA adapter produced by train.py when ./adapter/ contains adapter
files; falls back to the keyword-matching baseline otherwise.

CLI:
    python main.py --input  validation_input.json \\
                   --output predictions.json \\
                   [--schemas_dir ./schemas] \\
                   [--adapter_dir  ./adapter] \\
                   [--base_model   Qwen/Qwen2.5-0.5B-Instruct]
"""

import argparse
import json
import os
import re

# Must match train.py exactly.
BASE_MODEL     = 'Qwen/Qwen2.5-1.5B-Instruct'
ADAPTER_DIR    = './adapter'
MAX_NEW_TOKENS = 512

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


# ── Schema helpers ────────────────────────────────────────────────────────────

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


def serialize_schema(schema: dict, fmt: str = 'compact',
                     col_types: dict = None) -> str:
    """Serialize schema for the prompt.  fmt must match the format used at training time."""
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

    # compact family: compact | fewshot | typed | fewshot_typed
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
    """Drop hallucinated tables/columns and restore canonical casing.

    eval.py counts every identifier not in the real schema as a false positive,
    so this step is critical for precision.
    """
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


class ModelPredictor:
    """Wraps a PEFT-adapted causal LM for schema-link prediction."""

    def __init__(self, base_model: str, adapter_dir: str, schema_format: str = 'compact'):
        # Lazy imports: this class is only instantiated when the adapter exists,
        # so users without torch/peft can still run the keyword baseline.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer: prefer what's saved in the adapter dir (train.py may
        # have copied it there); fall back to the HF hub base model.
        # Prefer the base model recorded in the adapter config so the caller
        # doesn't need to keep main.py in sync with train.py manually.
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
        sys_prompt = (SYSTEM_PROMPT_TYPED
                      if self.schema_format in ('typed', 'fewshot_typed')
                      else SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"{serialize_schema(schema, self.schema_format, col_types)}\n\nQuestion: {question}"},
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

        # Decode only the newly generated tokens, not the prompt.
        raw = self.tokenizer.decode(out[0][input_ids.shape[-1]:],
                                    skip_special_tokens=True)
        if debug:
            print(f"    [raw] {repr(raw[:500])}")
        return _parse_json(raw)


# ── Keyword-matching fallback ─────────────────────────────────────────────────

def _split_id(name: str) -> set:
    """Split a camelCase / snake_case identifier into lowercase sub-words."""
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return {t for t in re.split(r'[^a-zA-Z0-9]+', s.lower()) if len(t) >= 2}


def _keyword_predict(question: str, schema: dict) -> dict:
    q = set(re.findall(r'[a-z]{2,}', question.lower()))
    result = {}
    for table, cols in schema.items():
        matched = [c for c in cols if _split_id(c) & q]
        if (_split_id(table) & q) or matched:
            result[table] = matched
    return result


# ── Schema pruning (keeps prompt under ~800 tokens) ──────────────────────────

MAX_SCHEMA_TABLES = 20

def prune_schema(question: str, schema: dict) -> dict:
    """Rank tables by keyword overlap with the question; keep top MAX_SCHEMA_TABLES."""
    if len(schema) <= MAX_SCHEMA_TABLES:
        return schema
    q = set(re.findall(r'[a-z]{2,}', question.lower()))
    def score(t):
        return len(_split_id(t) & q) * 2 + sum(1 for c in schema[t] if _split_id(c) & q)
    ranked = sorted(schema, key=score, reverse=True)
    return {t: schema[t] for t in ranked[:MAX_SCHEMA_TABLES]}


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
        if predictor.schema_format in ('typed', 'fewshot_typed'):
            col_types = load_col_types(db_id, schemas_dir)
        pruned = prune_schema(question, schema)
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
                             'fewshot', 'typed', 'fewshot_typed'],
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
