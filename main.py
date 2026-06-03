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
MAX_NEW_TOKENS = 768

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

SYSTEM_PROMPT_PKFK = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type; [PK]=primary key, [FK]=foreign key) "
    "and a natural language question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without type/key suffixes) from the schema. "
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


def load_pkfk(db_id: str, schemas_dir: str) -> dict:
    """Return {table: {col: 'PK'|'FK'|''}} for PK/FK annotations."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    col_info    = s['column_names_original']
    pk_indices  = set(s.get('primary_keys', []))
    fk_indices  = set()
    for pair in s.get('foreign_keys', []):
        fk_indices.update(pair)
    flags = {t: {} for t in table_names}
    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t = table_names[tidx]
        if i in pk_indices:
            flags[t][cname] = 'PK'
        elif i in fk_indices:
            flags[t][cname] = 'FK'
        else:
            flags[t][cname] = ''
    return flags


def merge_type_pkfk(col_types: dict, pkfk: dict) -> dict:
    """Return {table: {col: 'type[PK]'/'type[FK]'/'type'}} for schema_sorted_pkfk."""
    merged = {}
    for table, cols in col_types.items():
        merged[table] = {}
        flags = pkfk.get(table, {})
        for col, typ in cols.items():
            flag = flags.get(col, '')
            merged[table][col] = f"{typ}[{flag}]" if (typ and flag) else (typ or (f"[{flag}]" if flag else ''))
    return merged


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
    # schema_sorted_2stage uses identical serialization — only the completion format differs.
    if fmt in ('schema_sorted', 'schema_sorted_2stage'):
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

    # ── schema_sorted_pkfk: sorted + col:type[PK/FK] annotations ──
    # col_types should already be the merged {table:{col:"type[PK]"}} dict
    # (built by merge_type_pkfk() before calling serialize_schema).
    if fmt == 'schema_sorted_pkfk':
        lines = []
        for table in sorted(schema.keys()):
            cols     = sorted(schema[table])
            t_ann    = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_ann[c]}" if t_ann.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    # ── Exp5 / test8: tables sorted A→Z, columns in original schema order ──
    if fmt in ('sorted_table_orig_col', 'schema_sorted_origcol'):
        lines = []
        for table in sorted(schema.keys()):
            cols    = schema[table]           # original order preserved
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

def _repair_json(text: str) -> str:
    """Close unclosed JSON brackets/braces to recover from truncated output."""
    # Strip trailing comma (incomplete array/object entry)
    text = re.sub(r',\s*$', '', text.rstrip())
    depth_brace = depth_bracket = 0
    in_str = esc = False
    for c in text:
        if esc:
            esc = False; continue
        if c == '\\' and in_str:
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        depth_brace   += (c == '{') - (c == '}')
        depth_bracket += (c == '[') - (c == ']')
    suffix = ('"' if in_str else '') + (']' * max(0, depth_bracket)) + ('}' * max(0, depth_brace))
    return text + suffix


def _parse_json(text: str) -> dict:
    """Extract a JSON dict from raw model output with multi-stage fallback."""
    text = text.strip()
    # Stage 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Stage 2: find outermost {...} span and parse
    start = text.find('{')
    if start == -1:
        return {}
    end = text.rfind('}')
    if end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Stage 3: repair truncated JSON (close unclosed brackets)
    try:
        obj = json.loads(_repair_json(text[start:]))
        if isinstance(obj, dict):
            return obj
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
    if schema_format == 'schema_sorted_pkfk':
        return SYSTEM_PROMPT_PKFK
    if schema_format == 'schema_sorted_2stage':
        return (
            "You are a database assistant. "
            "Given a database schema (column types shown as col:type) and a natural language question, "
            "first output the relevant table names as a JSON array on one line prefixed with 'Tables: ', "
            "then output the complete schema links as a JSON object on the next line: "
            "{\"TableName\": [\"col1\", \"col2\"]}. "
            "Use ONLY table and column names (without the :type suffix) from the schema. "
            "Include only the tables and columns needed to answer the question. "
            "Output exactly two lines. No extra text."
        )
    if schema_format in ('typed', 'fewshot_typed', 'schema_sorted', 'schema_top10',
                         'col_filtered', 'sorted_desc', 'sorted_5ep',
                         'sorted_table_orig_col', 'schema_sorted_origcol'):
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
        self._base_model   = base_model   # kept for Qwen3 detection

    def _build_input(self, question: str, schema: dict, col_types: dict):
        sys_prompt  = _get_system_prompt(self.schema_format)
        schema_text = serialize_schema(schema, self.schema_format,
                                       col_types=col_types, question=question)
        user_content = schema_text if self.schema_format == 'question_hint' \
                       else f"{schema_text}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ]
        kwargs = dict(add_generation_prompt=True, return_tensors="pt")
        # Qwen3 emits <think> blocks by default; disable for structured output
        if 'qwen3' in self._base_model.lower():
            try:
                result = self.tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs)
            except TypeError:
                result = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            result = self.tokenizer.apply_chat_template(messages, **kwargs)
        # transformers 5.x returns BatchEncoding (UserDict, not dict subclass);
        # older versions return a plain tensor
        ids = result.input_ids if hasattr(result, 'input_ids') else result
        return ids.to(self.model.device)

    def _decode(self, out, input_len: int) -> dict:
        raw = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        if self.schema_format == 'col_hint_output':
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            raw = next((l for l in lines if l.startswith('{')), raw)
        elif self.schema_format == 'schema_sorted_2stage':
            # Parse only the JSON object line; ignore the "Tables: [...]" line
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            json_lines = [l for l in lines if l.startswith('{')]
            if json_lines:
                raw = json_lines[0]
            else:
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                raw = m.group() if m else '{}'
        return raw, _parse_json(raw)

    def predict(self, question: str, schema: dict, debug: bool = False,
                col_types: dict = None) -> dict:
        input_ids = self._build_input(question, schema, col_types)
        input_len = input_ids.shape[-1]

        # First pass: greedy decoding
        with self._torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw, result = self._decode(out, input_len)
        if debug:
            print(f"    [raw greedy] {repr(raw[:400])}")

        # Second pass: temperature sampling when greedy gave nothing
        if not result:
            with self._torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            raw2, result = self._decode(out, input_len)
            if debug:
                print(f"    [raw sample] {repr(raw2[:400])}")

        return result


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

_TYPED_FORMATS = frozenset({
    'typed', 'fewshot_typed', 'schema_abbrev', 'schema_sorted', 'schema_top10',
    'sorted_abbrev', 'question_hint', 'col_filtered', 'sorted_desc',
    'col_hint_output', 'sorted_5ep', 'sorted_table_orig_col', 'schema_sorted_pkfk',
    'schema_sorted_origcol', 'schema_sorted_2stage',
})
_PKFK_FORMATS = frozenset({'schema_sorted_pkfk'})
_FULL_SCHEMA_FORMATS = frozenset({'schema_top10', 'question_hint'})


def predict_schema_links(question: str, db_id: str, schemas_dir: str,
                         predictor=None, debug: bool = False,
                         hybrid_fallback: bool = True) -> dict:
    schema = load_schema(db_id, schemas_dir)

    if predictor is not None:
        col_types = load_col_types(db_id, schemas_dir) \
                    if predictor.schema_format in _TYPED_FORMATS else None
        if predictor.schema_format in _PKFK_FORMATS and col_types is not None:
            col_types = merge_type_pkfk(col_types, load_pkfk(db_id, schemas_dir))

        pruned = schema if predictor.schema_format in _FULL_SCHEMA_FORMATS \
                 else prune_schema(question, schema, max_tables=MAX_SCHEMA_TABLES)

        raw = predictor.predict(question, pruned, debug=debug, col_types=col_types)
        result = filter_against_schema(raw, schema)

        # If the model returned nothing, fall back to keyword matching
        if not result and hybrid_fallback:
            result = _keyword_predict(question, schema)

        return result

    return _keyword_predict(question, schema)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',       required=True)
    ap.add_argument('--output',      required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    ap.add_argument('--adapter_dir', default=ADAPTER_DIR)
    ap.add_argument('--base_model',    default=BASE_MODEL)
    ap.add_argument('--schema_format', default='schema_sorted',
                    choices=['compact', 'sql_ddl', 'markdown',
                             'fewshot', 'typed', 'fewshot_typed',
                             'schema_abbrev', 'schema_sorted', 'schema_top10',
                             'sorted_abbrev', 'question_hint', 'col_filtered',
                             'sorted_desc', 'col_hint_output', 'sorted_5ep',
                             'sorted_table_orig_col', 'schema_sorted_pkfk',
                             'schema_sorted_origcol', 'schema_sorted_2stage'],
                    help='Schema serialization format — must match what was used at training time')
    ap.add_argument('--debug', type=int, default=0, metavar='N',
                    help='Print raw model output for the first N predictions (0 = off)')
    ap.add_argument('--no_hybrid_fallback', action='store_true',
                    help='Disable keyword-matching fallback when model returns empty output')
    args = ap.parse_args()
    hybrid = not args.no_hybrid_fallback

    if _adapter_ready(args.adapter_dir):
        print(f"[mode] fine-tuned model  (schema_format={args.schema_format}, hybrid={hybrid})")
        predictor = ModelPredictor(args.base_model, args.adapter_dir, args.schema_format)
    else:
        print(f"[mode] keyword-matching baseline  "
              f"(no adapter at '{args.adapter_dir}' — run train_v2.py first)")
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
            debug=debug_this, hybrid_fallback=hybrid)
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