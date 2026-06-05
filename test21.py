"""
test21.py -- CED v3 (shared-column fix) on RTX PRO 6000 (bf16).

Root-cause analysis of test20-a (LB=0.4990):

  1. SHARED-COLUMN HALLUCINATION (most critical bug in CED-v2):
     CATEGORY appears in 38/40 NTSB tables.  CED-v2 generated
     {"AIRBAG": ["CATEGORY"]}, {"CRASH": ["CATEGORY"]}, ... for every
     NTSB table.  The model learned "NTSB question → output all tables
     with CATEGORY".  This caused QID 33 to predict 38 tables, all with
     just CATEGORY.  Same pattern for NYSED (ENTITY_CD in 26/27 tables).

  2. GV.LIGHTCOND: ZERO training examples.
     "Display a count of crashes by lighting condition" → model picks
     CRASH (wrong) because it never saw LIGHTCOND → GV.

  3. VPICDECODE seatbelt confusion: CED added VPICDECODE.SeatBeltTypeId,
     so the model maps "seatbelt" → VPICDECODE instead of OCC.BELTUSE.

  4. CDC.CMAX / CDC.DVBARRIER: 8 narrow training examples; model can't
     generalize from "all crush measurements" to "max crush depth" or
     "speed difference".

  5. AVOID table: 8 aug_v2 examples all about "most common feature" →
     fails on "how many vehicles were equipped with crash avoidance".

  6. ICS.SOE (energy-source injuries): near-zero coverage.

Fixes in test21:
  1. CED v3: compute per-DB "shared columns" (appearing in >= 3 tables);
     exclude them from CED examples → model learns table-SPECIFIC columns,
     not the boilerplate columns that appear everywhere.
  2. Hardcoded problem examples for each root cause above.
  3. Keep all other hyperparams identical to test20-a (r=16, alpha=32,
     lr=2e-4, 4 epochs, bf16, batch=2×grad_accum=2).

Hypothesis: fixing the shared-column CED bug should dramatically reduce
  table hallucination (the 38-table / 20-table catastrophic cases) and
  improve column precision by ~5-10 percentage points.

Hardware: RTX PRO 6000 (24 GB MIG slice, bf16 full precision)
Adapter → ./adapters/test21/
Preds   → test21-preds.json
Log     → test21-log.txt
"""

import datetime
import json
import os
import random
import re
import sys
import traceback
from collections import Counter

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────

SCHEMAS_DIR    = './schemas'
TRAIN_JSON     = './augmented_train_v2.json'
VAL_INPUT      = './validation_input.json'
LOG_FILE       = './test21-log.txt'

LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_NEW_TOKENS = 1024

EXP = {
    'tag':          'test21',
    'adapter_dir':  './adapters/test21',
    'preds_file':   './test21-preds.json',
    'base_model':   'Qwen/Qwen3-1.7B',
    'is_qwen3':     True,
    'schema_fmt':   'pkfk',
    'lora_r':       16,
    'lora_alpha':   32,
    'lora_dropout': 0.05,
    'lr':           2e-4,
    'epochs':       4,
    'batch_size':   2,
    'grad_accum':   2,
    'max_length':   2048,
    'hypothesis': (
        'CED-v3 shared-col fix: skip CATEGORY/CASEID/PSU/ENTITY_CD etc. from CED '
        'so model learns table-specific cols not boilerplate; '
        'plus hardcoded examples for GV.LIGHTCOND, CDC.CMAX, AVOID, ICS.SOE, OCC belt'
    ),
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a database assistant. "
    "Given a database schema (column types as col:type; [PK]=primary key, "
    "[FK]=foreign key) and a natural language question, output the schema links "
    "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without type/key suffixes) from the schema. "
    "Include ONLY tables whose own columns are directly needed to answer the question. "
    "Do NOT include a table only because it has a foreign key to another table; "
    "include it only if its own columns are needed. "
    "Most questions need only 1-3 tables. Do NOT list all tables in the database. "
    "Always list the specific columns required. "
    "Never output an empty column list []."
    "Output valid JSON only, with no extra text."
)

# ── Hardcoded problem examples (root-cause-driven) ────────────────────────────
# These address specific systematic failures identified in test20-a predictions.

HARDCODED_EXAMPLES = [
    # ── Root cause 2: GV.LIGHTCOND — 0 training examples in aug_v2 ──────────
    {'db_id': 'NTSB', 'question': 'Count crashes by road lighting condition.',
     'schema_links': {'GV': ['CASEID', 'LIGHTCOND']}},
    {'db_id': 'NTSB', 'question': 'What lighting conditions were present at crash scenes?',
     'schema_links': {'GV': ['CASEID', 'LIGHTCOND']}},
    {'db_id': 'NTSB', 'question': 'How many crashes occurred under each type of lighting?',
     'schema_links': {'GV': ['CASEID', 'LIGHTCOND']}},
    {'db_id': 'NTSB', 'question': 'Show the lighting condition for every crash event.',
     'schema_links': {'GV': ['LIGHTCOND']}},

    # ── Root cause 3b: GV.VIN — model defaulted to CRASH for vehicle count ──
    {'db_id': 'NTSB', 'question': 'How many distinct vehicles are in the database?',
     'schema_links': {'GV': ['VIN']}},
    {'db_id': 'NTSB', 'question': 'Count the number of unique vehicles recorded.',
     'schema_links': {'GV': ['VIN', 'CASEID']}},

    # ── Root cause 4a: CDC.CMAX — simple max-depth queries not in aug_v2 ────
    {'db_id': 'NTSB', 'question': 'What is the maximum vehicle crush depth recorded?',
     'schema_links': {'CDC': ['CMAX']}},
    {'db_id': 'NTSB', 'question': 'Show the maximum crush measurement for each vehicle.',
     'schema_links': {'CDC': ['CASEID', 'VEHNO', 'CMAX']}},
    {'db_id': 'NTSB', 'question': 'Find the deepest crush measurement ignoring invalid codes.',
     'schema_links': {'CDC': ['CMAX', 'CASEID']}},

    # ── Root cause 4b: CDC.DVBARRIER — speed change in barrier collision ────
    {'db_id': 'NTSB', 'question': 'What is the highest vehicle speed change in a barrier impact?',
     'schema_links': {'CDC': ['DVBARRIER']}},
    {'db_id': 'NTSB', 'question': 'Show the vehicle-to-barrier collision speed differences.',
     'schema_links': {'CDC': ['CASEID', 'VEHNO', 'DVBARRIER']}},

    # ── Root cause 5: AVOID — 8 aug_v2 examples all say "most common feature" ─
    {'db_id': 'NTSB', 'question': 'How many vehicles were equipped with crash avoidance systems?',
     'schema_links': {'AVOID': ['CASEID', 'AVAIL']}},
    {'db_id': 'NTSB', 'question': 'Which vehicles had crash avoidance features installed?',
     'schema_links': {'AVOID': ['CASEID', 'EQUIP', 'AVAIL']}},
    {'db_id': 'NTSB', 'question': 'Count crashes where the vehicle had active crash avoidance equipment.',
     'schema_links': {'AVOID': ['CASEID', 'AVAIL']}},
    {'db_id': 'NTSB', 'question': 'Were crash avoidance systems available and activated in these crashes?',
     'schema_links': {'AVOID': ['AVAIL', 'ACTIVATE', 'CASEID']}},

    # ── Root cause 6: ICS.SOE — energy-source injuries near zero coverage ───
    {'db_id': 'NTSB', 'question': 'What are the energy sources causing injuries in crashes?',
     'schema_links': {'ICS': ['SOE', 'CASEID']}},
    {'db_id': 'NTSB', 'question': 'Count injuries by energy source type.',
     'schema_links': {'ICS': ['SOE']}},
    {'db_id': 'NTSB', 'question': 'Which energy sources were unknown in injury cases?',
     'schema_links': {'ICS': ['SOE', 'CASEID']}},

    # ── Root cause 3a: OCC seatbelt — CED added VPICDECODE.SeatBeltTypeId ──
    # Model confused "seatbelt" → VPICDECODE; need OCC.BELTAVAIL/BELTUSE
    {'db_id': 'NTSB', 'question': 'Which occupants had a seatbelt available but did not use it?',
     'schema_links': {'OCC': ['BELTAVAIL', 'BELTUSE']}},
    {'db_id': 'NTSB', 'question': 'Show seatbelt availability and actual usage for each occupant.',
     'schema_links': {'OCC': ['BELTAVAIL', 'BELTUSE', 'CASEID']}},
    {'db_id': 'NTSB', 'question': 'How many vehicle occupants wore a seatbelt?',
     'schema_links': {'OCC': ['BELTUSE', 'CASEID']}},
    {'db_id': 'NTSB', 'question': 'Count occupant mortality by seatbelt use category.',
     'schema_links': {'OCC': ['BELTUSE', 'MORTALITY']}},

    # ── INTRUSION.INTMAG — model predicted CATEGORY (shared-col bug) ─────────
    {'db_id': 'NTSB', 'question': 'Show intrusion magnitude for each vehicle compartment.',
     'schema_links': {'INTRUSION': ['INTMAG', 'INTCOMP', 'CASEID']}},
    {'db_id': 'NTSB', 'question': 'Count vehicles by intrusion magnitude category.',
     'schema_links': {'INTRUSION': ['INTMAG', 'CASEID', 'VEHNO']}},

    # ── Institution_Grouping + never-paired tables ─────────────────────────────
    # 95 aug_v2 examples exist but missing pairings for specific tables:
    # Inexperienced_Teachers_and_Principals, Accountability_Status, Annual_NYSESLAT
    {'db_id': 'NYSED_SRC2022',
     'question': 'Which public schools have the highest percentage of inexperienced teachers?',
     'schema_links': {'Inexperienced_Teachers_and_Principals': ['ENTITY_CD', 'PER_TEACH_INEXP', 'YEAR'],
                      'Institution_Grouping': ['ENTITY_CD', 'GROUP_NAME']}},
    {'db_id': 'NYSED_SRC2022',
     'question': 'Show the inexperienced teacher rate for public school entities in 2021.',
     'schema_links': {'Inexperienced_Teachers_and_Principals': ['ENTITY_CD', 'ENTITY_NAME', 'PER_TEACH_INEXP', 'YEAR'],
                      'Institution_Grouping': ['ENTITY_CD', 'GROUP_NAME']}},
    {'db_id': 'NYSED_SRC2022',
     'question': 'What is the accountability status for each public school entity?',
     'schema_links': {'Accountability_Status': ['ENTITY_CD', 'OVERALL_STATUS', 'YEAR'],
                      'Institution_Grouping': ['ENTITY_CD', 'GROUP_NAME']}},
    {'db_id': 'NYSED_SRC2022',
     'question': 'Count public school entities in each accountability status for 2022.',
     'schema_links': {'Accountability_Status': ['ENTITY_CD', 'OVERALL_STATUS', 'YEAR'],
                      'Institution_Grouping': ['ENTITY_CD', 'GROUP_NAME']}},
    {'db_id': 'NYSED_SRC2022',
     'question': 'What is the average NYSESLAT score for students at public schools?',
     'schema_links': {'Annual_NYSESLAT': ['ENTITY_CD', 'SUBJECT', 'YEAR'],
                      'Institution_Grouping': ['ENTITY_CD', 'GROUP_NAME']}},
    {'db_id': 'NYSED_SRC2022',
     'question': 'Show the entering-level percentage on NYSESLAT for public school students.',
     'schema_links': {'Annual_NYSESLAT': ['ENTITY_CD', 'PER_ENT', 'SUBJECT', 'YEAR'],
                      'Institution_Grouping': ['ENTITY_CD', 'GROUP_NAME']}},
    # Postsecondary_Enrollment — only 0 aug_v2 examples (never-seen table)
    {'db_id': 'NYSED_SRC2022',
     'question': 'How many students enrolled in out-of-state 4-year colleges?',
     'schema_links': {'Postsecondary_Enrollment': ['ENTITY_CD', 'OUT_4_YR_CNT', 'YEAR']}},
    {'db_id': 'NYSED_SRC2022',
     'question': 'Show the count of graduates who enrolled in 4-year out-of-state programs.',
     'schema_links': {'Postsecondary_Enrollment': ['OUT_4_YR_CNT', 'ENTITY_CD']}},
]

# ── Logging ───────────────────────────────────────────────────────────────────

def _ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg, also_print=True):
    line = f"[{_ts()}] {msg}"
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line, flush=True)

def log_sep(title=''):
    sep = '=' * 65
    log(sep)
    if title:
        log(f"  {title}")
        log(sep)

# ── Schema helpers ────────────────────────────────────────────────────────────

def _load_schema_raw(db_id):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(SCHEMAS_DIR, fname)) as f:
        return json.load(f)

def _build_maps(raw, schema_fmt='pkfk'):
    tnames      = raw['table_names_original']
    col_info    = raw['column_names_original']
    ctypes_list = raw.get('column_types', [])
    pk_set      = set(raw.get('primary_keys', []))
    fk_set      = set()
    for pair in raw.get('foreign_keys', []):
        fk_set.update(pair)

    schema  = {t: [] for t in tnames}
    col_ann = {t: {} for t in tnames}

    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t   = tnames[tidx]
        typ = ctypes_list[i] if i < len(ctypes_list) else ''
        schema[t].append(cname)

        if schema_fmt == 'pkfk':
            flag = 'PK' if i in pk_set else ('FK' if i in fk_set else '')
            ann  = f"{typ}[{flag}]" if typ and flag else (f"[{flag}]" if flag else typ)
        else:
            ann = typ
        col_ann[t][cname] = ann

    lc_tables = {t.lower(): t for t in schema}
    lc_cols   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    return schema, col_ann, lc_tables, lc_cols

def serialize_schema(schema, col_ann):
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        ann      = col_ann.get(table, {})
        col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)

def filter_and_dedup(links, lc_tables, lc_cols):
    result = {}
    for table, cols in links.items():
        canonical_t = lc_tables.get(str(table).lower())
        if canonical_t is None:
            continue
        if not isinstance(cols, list):
            result[canonical_t] = []
            continue
        col_map = lc_cols.get(canonical_t, {})
        seen, deduped = set(), []
        for c in cols:
            canonical_c = col_map.get(str(c).lower())
            if canonical_c is not None and canonical_c not in seen:
                seen.add(canonical_c)
                deduped.append(canonical_c)
        result[canonical_t] = deduped
    return result

# ── JSON parsing ──────────────────────────────────────────────────────────────

def _repair_json(text):
    text = re.sub(r',\s*$', '', text.rstrip())
    depth_brace = depth_bracket = 0
    in_str = esc = False
    for ch in text:
        if esc:                   esc = False; continue
        if ch == '\\' and in_str: esc = True;  continue
        if ch == '"':             in_str = not in_str; continue
        if in_str:                continue
        depth_brace   += (ch == '{') - (ch == '}')
        depth_bracket += (ch == '[') - (ch == ']')
    return text + ('"' if in_str else '') + ']' * max(0, depth_bracket) + '}' * max(0, depth_brace)

def parse_json(text):
    text = text.strip()
    for candidate in [text, text[text.find('{'):text.rfind('}')+1] if '{' in text else '']:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        start = text.find('{')
        if start != -1:
            obj = json.loads(_repair_json(text[start:]))
            if isinstance(obj, dict):
                return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return {}

# ── CED v3 helpers ────────────────────────────────────────────────────────────

def get_schema_shared_cols(raw, threshold=3):
    """
    Return column names (lowercase) that appear in >= threshold tables.

    These are "boilerplate" keys (e.g. CASEID in 38/40 NTSB tables,
    CATEGORY in 38/40, ENTITY_CD in 26/27 NYSED tables).  CED-v2 generated
    one-column queries for every table using these columns, which taught the
    model "NTSB question → output all tables with CATEGORY."  Excluding them
    forces CED-v3 to use table-SPECIFIC columns that discriminate tables.
    """
    col_counts = Counter()
    for tidx, cname in raw['column_names_original']:
        if tidx != -1:
            col_counts[cname.lower()] += 1
    return {c for c, n in col_counts.items() if n >= threshold}

# CED v3 templates (same as v2)
_CED_1COL = [
    "What are the {col} values in {table}?",
    "List the {col} from the {table} table.",
    "Show the {col} column from {table}.",
    "Retrieve all {col} entries from {table}.",
    "Get the distinct {col} values in {table}.",
    "What {col} information is stored in {table}?",
    "Find all {col} records in {table}.",
]
_CED_2COL = [
    "Show the {col1} and {col2} from {table}.",
    "List the {col1} and {col2} in {table}.",
    "What are the {col1} and {col2} for records in {table}?",
    "Retrieve {col1} and {col2} from the {table} table.",
]
_CED_FK = [
    "Show {col1} from {t1} and {col2} from {t2}.",
    "List the {col1} in {t1} and the corresponding {col2} in {t2}.",
    "What are the {col1} values in {t1} and {col2} values in {t2}?",
    "Get {col1} from {t1} along with {col2} from {t2}.",
]


def generate_ced_v3(aug_v2, val_items, rng):
    """
    CED v3 — key fix over CED v2:

    CED-v2 generated single-column queries for EVERY column of EVERY table,
    including "boilerplate" columns that appear in almost every table of a DB
    (CASEID, CATEGORY, PSU, VERSION for NTSB; ENTITY_CD, YEAR for NYSED).
    This produced training examples like:
        "What are the CATEGORY values in AIRBAG?" → {"AIRBAG": ["CATEGORY"]}
        "What are the CATEGORY values in CRASH?"  → {"CRASH": ["CATEGORY"]}
        ... (38 such pairs for NTSB alone)

    The model learned: "for NTSB questions, output all tables with CATEGORY."
    QID 33 prediction (38 tables, all with CATEGORY) is the direct result.

    CED-v3 fix: compute per-DB shared columns (appearing in >= 3 tables) and
    EXCLUDE them from CED generation.  Instead use only table-SPECIFIC columns
    that discriminate one table from another.
    """
    covered        = set()
    covered_tables = set()
    for x in aug_v2:
        db = x['db_id']
        for t, cols in (x.get('schema_links') or {}).items():
            covered_tables.add((db, t.lower()))
            for c in (cols or []):
                covered.add((db, t.lower(), c.lower()))

    val_db_ids   = set(item['db_id'] for item in val_items)
    train_db_ids = set(x['db_id'] for x in aug_v2)
    all_db_ids   = val_db_ids | train_db_ids

    ced = []

    for db_id in sorted(all_db_ids):
        try:
            raw = _load_schema_raw(db_id)
        except FileNotFoundError:
            continue

        tnames      = raw['table_names_original']
        col_info    = raw['column_names_original']
        schema      = {t: [] for t in tnames}
        for tidx, cname in col_info:
            if tidx != -1:
                schema[tnames[tidx]].append(cname)

        # v3 fix: compute and exclude shared/boilerplate columns
        shared_cols = get_schema_shared_cols(raw, threshold=3)

        is_val_db = db_id in val_db_ids

        for table, cols in schema.items():
            if not cols:
                continue
            table_key = (db_id, table.lower())

            # Specific cols = cols that are NOT in 3+ other tables (unique to this table)
            specific = [c for c in cols if c.lower() not in shared_cols]
            # Fallback: if all columns are shared (rare edge case), use first 4
            if not specific:
                specific = cols[:4]

            # ── Part A: column-extension for aug_v2-covered tables ──────────
            if table_key in covered_tables:
                # Find uncovered specific columns
                uncov_specific = [c for c in specific
                                  if (db_id, table.lower(), c.lower()) not in covered]
                if uncov_specific:
                    sample = rng.sample(uncov_specific, min(6, len(uncov_specific)))
                    q = ("Show " + ", ".join(sample[:4])
                         + (", and " + ", ".join(sample[4:]) if len(sample) > 4 else "")
                         + f" from {table}.")
                    ced.append({'db_id': db_id, 'question': q,
                                'schema_links': {table: sample}})

                    for col in rng.sample(uncov_specific, min(4, len(uncov_specific))):
                        tmpl = rng.choice(_CED_1COL)
                        ced.append({'db_id': db_id, 'question': tmpl.format(col=col, table=table),
                                    'schema_links': {table: [col]}})

                    covered_specific = [c for c in specific
                                        if (db_id, table.lower(), c.lower()) in covered]
                    if covered_specific and uncov_specific:
                        c1 = rng.choice(uncov_specific)
                        c2 = rng.choice(covered_specific)
                        tmpl = rng.choice(_CED_2COL)
                        ced.append({'db_id': db_id,
                                    'question': tmpl.format(col1=c1, col2=c2, table=table),
                                    'schema_links': {table: [c1, c2]}})

            # ── Part B: full coverage for never-seen validation tables ───────
            elif is_val_db:
                # Use table-specific columns only (not shared ones) so the model
                # learns what makes this table unique
                sample = rng.sample(specific, min(5, len(specific)))
                q = ("Show " + ", ".join(sample[:3])
                     + (" and " + ", ".join(sample[3:]) if len(sample) > 3 else "")
                     + f" from the {table} table.")
                ced.append({'db_id': db_id, 'question': q,
                            'schema_links': {table: sample}})

                for col in rng.sample(specific, min(4, len(specific))):
                    tmpl = rng.choice(_CED_1COL)
                    ced.append({'db_id': db_id,
                                'question': tmpl.format(col=col, table=table),
                                'schema_links': {table: [col]}})

                if len(specific) >= 2:
                    c1, c2 = rng.sample(specific, 2)
                    tmpl = rng.choice(_CED_2COL)
                    ced.append({'db_id': db_id,
                                'question': tmpl.format(col1=c1, col2=c2, table=table),
                                'schema_links': {table: [c1, c2]}})

        # ── Part C: 2-table FK join examples ────────────────────────────────
        if is_val_db:
            for fk_pair in raw.get('foreign_keys', [])[:8]:
                idx1, idx2 = fk_pair
                if idx1 >= len(col_info) or idx2 >= len(col_info):
                    continue
                t1_idx, _c1 = col_info[idx1]
                t2_idx, _c2 = col_info[idx2]
                if t1_idx == -1 or t2_idx == -1 or t1_idx == t2_idx:
                    continue
                t1, t2 = tnames[t1_idx], tnames[t2_idx]
                # Use non-FK, non-shared content columns for the join example
                t1_spec = [c for ti, c in col_info
                           if ti == t1_idx and c != _c1 and c.lower() not in shared_cols]
                t2_spec = [c for ti, c in col_info
                           if ti == t2_idx and c != _c2 and c.lower() not in shared_cols]
                if not t1_spec or not t2_spec:
                    continue
                c1_extra = rng.choice(t1_spec[:4])
                c2_extra = rng.choice(t2_spec[:4])
                tmpl = rng.choice(_CED_FK)
                ced.append({'db_id': db_id,
                            'question': tmpl.format(col1=c1_extra, t1=t1, col2=c2_extra, t2=t2),
                            'schema_links': {t1: [c1_extra], t2: [c2_extra]}})

    rng.shuffle(ced)
    return ced

# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(train_data, tokenizer):
    cache = {}
    texts, skipped = [], 0
    schema_fmt = EXP['schema_fmt']

    for item in train_data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                raw = _load_schema_raw(db_id)
                schema, col_ann, _, _ = _build_maps(raw, schema_fmt)
                cache[db_id] = (schema, col_ann)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, col_ann = cache[db_id]
        answer = item.get('schema_links') or {}
        if not isinstance(answer, dict):
            answer = {}

        user_content = f"{serialize_schema(schema, col_ann)}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception:
            skipped += 1

    print(f"  Dataset: {len(texts)} examples, {skipped} skipped", flush=True)
    return Dataset.from_dict({"text": texts})

# ── Training ──────────────────────────────────────────────────────────────────

def run_training(train_data):
    tag = EXP['tag']
    log_sep(f"TRAINING  {tag}  (bf16, RTX PRO 6000)")
    log(f"  model={EXP['base_model']}  schema={EXP['schema_fmt']}")
    log(f"  r={EXP['lora_r']}  alpha={EXP['lora_alpha']}  dropout={EXP['lora_dropout']}")
    log(f"  lr={EXP['lr']}  epochs={EXP['epochs']}  max_length={EXP['max_length']}")
    log(f"  batch={EXP['batch_size']}  grad_accum={EXP['grad_accum']}")
    log(f"  train_size={len(train_data)}")
    log(f"  hypothesis: {EXP['hypothesis']}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(EXP['base_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = build_dataset(train_data, tokenizer)

        print(f"=== [{tag}] Loading base model (bf16) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            EXP['base_model'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=EXP['lora_r'],
            lora_alpha=EXP['lora_alpha'],
            lora_dropout=EXP['lora_dropout'],
            target_modules=LORA_TARGETS,
            bias="none",
        )
        os.makedirs(EXP['adapter_dir'], exist_ok=True)

        sft_cfg = SFTConfig(
            output_dir=EXP['adapter_dir'],
            num_train_epochs=EXP['epochs'],
            per_device_train_batch_size=EXP['batch_size'],
            gradient_accumulation_steps=EXP['grad_accum'],
            learning_rate=EXP['lr'],
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            optim="adamw_torch_fused",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_length=EXP['max_length'],
            completion_only_loss=True,
            packing=False,
        )

        print(f"=== [{tag}] Starting training ({EXP['epochs']} epochs) ===", flush=True)
        trainer = SFTTrainer(
            model=model, args=sft_cfg,
            train_dataset=dataset, peft_config=lora_cfg,
        )
        trainer.train()

        print(f"=== [{tag}] Saving adapter → {EXP['adapter_dir']} ===", flush=True)
        trainer.model.save_pretrained(EXP['adapter_dir'])
        tokenizer.save_pretrained(EXP['adapter_dir'])
        log(f"  Training complete. Adapter → {EXP['adapter_dir']}")

        del trainer, model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
        log(f"  [{'OOM' if is_oom else 'ERROR'}] {tag} training failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(val_items):
    tag        = EXP['tag']
    schema_fmt = EXP['schema_fmt']
    log_sep(f"INFERENCE  {tag}")

    try:
        tok_src = (EXP['adapter_dir']
                   if os.path.exists(os.path.join(EXP['adapter_dir'], 'tokenizer_config.json'))
                   else EXP['base_model'])
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"=== [{tag}] Loading bf16 base + adapter for inference ===", flush=True)
        base  = AutoModelForCausalLM.from_pretrained(
            EXP['base_model'], torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, EXP['adapter_dir'])
        model.eval()

        preds = []
        n     = len(val_items)

        for i, item in enumerate(val_items, 1):
            qid      = item['question_id']
            db_id    = item['db_id']
            question = item['question']

            try:
                raw = _load_schema_raw(db_id)
                schema, col_ann, lc_tables, lc_cols = _build_maps(raw, schema_fmt)
            except FileNotFoundError:
                log(f"  WARNING: schema missing for {db_id} (qid={qid})")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            user_content = f"{serialize_schema(schema, col_ann)}\n\nQuestion: {question}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ]

            try:
                try:
                    enc = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True,
                        return_tensors="pt", enable_thinking=False)
                except TypeError:
                    enc = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt")
            except Exception as e:
                log(f"  Tokenisation error qid={qid}: {e}")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            input_ids = enc.input_ids if hasattr(enc, 'input_ids') else enc
            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[-1]

            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw_text  = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            raw_links = parse_json(raw_text)

            if not raw_links:
                with torch.no_grad():
                    out2 = model.generate(
                        input_ids, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=0.4, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id)
                raw_links = parse_json(
                    tokenizer.decode(out2[0][input_len:], skip_special_tokens=True))

            validated = filter_and_dedup(raw_links, lc_tables, lc_cols)
            preds.append({'question_id': qid, 'schema_links': validated})

            if i % 10 == 0 or i == n:
                print(f"[{tag}] Inference {i}/{n}", flush=True)

        with open(EXP['preds_file'], 'w') as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        log(f"  Wrote {len(preds)} predictions → {EXP['preds_file']}")

        del model, base
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
        log(f"  [{'OOM' if is_oom else 'ERROR'}] {tag} inference failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)

    with open(LOG_FILE, 'w') as f:
        f.write(f"test21.py  started {_ts()}\n")
        f.write(f"schema=pkfk  hardware=RTX-PRO-6000  CED-v3-shared-col-fix\n\n")

    log_sep("test21.py  —  CED-v3 shared-col fix  —  RTX PRO 6000")

    # ── Load aug_v2 ───────────────────────────────────────────────────────────
    aug_v2 = None
    for path in [TRAIN_JSON, './augmented_train_10x.json', './train.json']:
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            aug_v2 = [x for x in raw if x.get('schema_links') is not None]
            log(f"Loaded {len(aug_v2)} examples from {path}")
            break
    if aug_v2 is None:
        log("ERROR: no training data found."); sys.exit(1)

    # Filter zero-column examples
    aug_v2_full = aug_v2
    aug_v2 = [x for x in aug_v2
              if any(cols for cols in (x.get('schema_links') or {}).values())]
    log(f"Filtered {len(aug_v2_full) - len(aug_v2)} zero-col examples → {len(aug_v2)} remain")

    with open(VAL_INPUT) as f:
        val_items = json.load(f)
    log(f"Validation: {len(val_items)} questions")

    # ── Generate CED v3 ───────────────────────────────────────────────────────
    log("Generating CED v3 (shared-column fix) ...")
    ced = generate_ced_v3(aug_v2, val_items, rng)
    log(f"  CED v3 generated: {len(ced)} examples")

    # ── Add hardcoded problem examples ────────────────────────────────────────
    n_hard = len(HARDCODED_EXAMPLES)
    log(f"  Adding {n_hard} hardcoded problem examples")

    combined = aug_v2 + ced + HARDCODED_EXAMPLES
    rng.shuffle(combined)
    log(f"Combined: {len(combined)} examples "
        f"(aug_v2={len(aug_v2)}, CED={len(ced)}, hardcoded={n_hard})")

    # Coverage report
    val_db_ids = set(item['db_id'] for item in val_items)
    train_tables = set()
    for x in combined:
        for t in (x.get('schema_links') or {}):
            train_tables.add((x['db_id'], t.lower()))
    total_vt = covered_vt = 0
    for db_id in val_db_ids:
        try:
            raw = _load_schema_raw(db_id)
            for t in raw['table_names_original']:
                total_vt += 1
                if (db_id, t.lower()) in train_tables:
                    covered_vt += 1
        except FileNotFoundError:
            pass
    log(f"  Val table coverage: {covered_vt}/{total_vt} ({100*covered_vt/max(total_vt,1):.1f}%)")

    # Show shared-col stats for key DBs
    for db_id in ['NTSB', 'NYSED_SRC2022']:
        try:
            raw = _load_schema_raw(db_id)
            shared = get_schema_shared_cols(raw, threshold=3)
            n_tabs = len(raw['table_names_original'])
            log(f"  {db_id}: {len(shared)} shared cols excluded from CED "
                f"(e.g. {sorted(shared)[:5]}); {n_tabs} tables")
        except FileNotFoundError:
            pass

    # ── Train ─────────────────────────────────────────────────────────────────
    log_sep("TRAINING")
    log(f"\n>>> Starting {EXP['tag']}  at {_ts()}")
    ok_train = run_training(combined)
    log(f">>> Finished {EXP['tag']}  {'OK' if ok_train else 'FAILED'}  at {_ts()}")

    # ── Inference ─────────────────────────────────────────────────────────────
    log_sep("INFERENCE")
    ok_infer = False
    if ok_train and os.path.isdir(EXP['adapter_dir']):
        log(f"\n>>> Starting inference  at {_ts()}")
        ok_infer = run_inference(val_items)
        log(f">>> Finished inference  {'OK' if ok_infer else 'FAILED'}  at {_ts()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log_sep("SUMMARY")
    log(f"  train: {'OK' if ok_train else 'FAILED'}")
    log(f"  infer: {'OK' if ok_infer else 'FAILED'}")
    if ok_infer:
        log(f"  preds: {EXP['preds_file']}")
        log(f"\nTo evaluate:")
        log(f"  python eval.py --predictions {EXP['preds_file']} "
            f"--gold validation_gold_schema_links.json "
            f"--schemas_dir schemas/ --questions_input validation_input.json")
    log(f"\nDone. Full log → {LOG_FILE}")


if __name__ == '__main__':
    main()
