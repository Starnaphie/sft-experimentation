"""
augment.py -- Richer training data augmentation for schema linking.

Strategies (no API calls required):
  1. Template paraphrasing    -- rewrite each question with 5–10 linguistic variants
  2. Schema-order shuffling   -- shuffle table + column order (extends existing 10x set)
  3. Schema-subset sampling   -- generate single-table examples from multi-table schemas
  4. Cross-question mixing    -- join two single-table questions into a two-table question
  5. Weighted oversampling    -- up-weight schemas with fewer than MIN_EXAMPLES examples

Goal: produce ~3 000 high-quality, diverse examples from the original 301.

Usage:
    python augment.py                          # writes augmented_train_v2.json (~3 000 ex)
    python augment.py --out my_aug.json --target 5000
    python augment.py --strategy paraphrase    # paraphrase only
    python augment.py --seed 99
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from copy import deepcopy

TRAIN_JSON  = './train.json'
SCHEMAS_DIR = './schemas'
OUT_JSON    = './augmented_train_v2.json'
TARGET      = 3000
MIN_EXAMPLES = 24   # schemas with fewer get extra copies

# ── Question paraphrase templates ─────────────────────────────────────────────
# Each template is a (pattern, replacement) where pattern is a compiled regex and
# replacement is a string (may reference group \1 etc.).
# We apply up to one transformation per template family so rewrites stay natural.

_REWRITE_RULES = [
    # "Show …" / "List …" / "Display …" etc.  ↔  various equivalents
    (re.compile(r'^Show(?:\s+me)?\s+', re.I),         'List '),
    (re.compile(r'^List\s+(?:all\s+)?', re.I),        'Show '),
    (re.compile(r'^Display\s+', re.I),                'Show me '),
    (re.compile(r'^Return\s+', re.I),                 'List '),
    (re.compile(r'^Find\s+', re.I),                   'Show '),
    (re.compile(r'^Get\s+', re.I),                    'Find '),
    (re.compile(r'^What\s+(?:are|is)\s+(?:the\s+)?', re.I), 'Show me the '),
    (re.compile(r'^Retrieve\s+', re.I),               'Get '),
    # Count variants
    (re.compile(r'\bcount\s+of\b', re.I),             'number of'),
    (re.compile(r'\bnumber\s+of\b', re.I),            'count of'),
    (re.compile(r'\bhow many\b', re.I),               'the count of'),
    (re.compile(r'\bthe count of\b', re.I),           'how many'),
    (re.compile(r'\btotal\s+number\s+of\b', re.I),    'how many'),
    # Filter variants
    (re.compile(r'\bwhere\b', re.I),                  'that have'),
    (re.compile(r'\bthat have\b', re.I),              'where'),
    (re.compile(r'\bwith a\b', re.I),                 'where the'),
    (re.compile(r'\bfor each\b', re.I),               'per'),
    (re.compile(r'\bper\b', re.I),                    'for each'),
    # Sort / order variants
    (re.compile(r'\bordered by\b', re.I),             'sorted by'),
    (re.compile(r'\bsorted by\b', re.I),              'ordered by'),
    (re.compile(r'\branked by\b', re.I),              'sorted by'),
    # Aggregate variants
    (re.compile(r'\baverage\b', re.I),                'mean'),
    (re.compile(r'\bmean\b', re.I),                   'average'),
    (re.compile(r'\bmaximum\b', re.I),                'highest'),
    (re.compile(r'\bhighest\b', re.I),                'maximum'),
    (re.compile(r'\bminimum\b', re.I),                'lowest'),
    (re.compile(r'\blowest\b', re.I),                 'minimum'),
    # Distinct / unique variants
    (re.compile(r'\bdistinct\b', re.I),               'unique'),
    (re.compile(r'\bunique\b', re.I),                 'distinct'),
    # Group by variants
    (re.compile(r'\bgroup(?:ed)?\s+by\b', re.I),      'broken down by'),
    (re.compile(r'\bbroken\s+down\s+by\b', re.I),     'grouped by'),
    # equals / equal to variants
    (re.compile(r'\bis\s+equal\s+to\b', re.I),        'equals'),
    (re.compile(r'\bequals\b', re.I),                 'is equal to'),
    (re.compile(r'\bis\s+greater\s+than\b', re.I),    'exceeds'),
    (re.compile(r'\bexceeds\b', re.I),                'is greater than'),
]

# Prefix swaps: replace sentence-initial verb phrases
_PREFIX_SWAPS = [
    (re.compile(r'^Show(?:\s+me)?\s+(?:the\s+)?', re.I), [
        'List ', 'Find ', 'Get ', 'Return ', 'Display ',
        'What are the ', 'Retrieve ',
    ]),
    (re.compile(r'^List\s+(?:all\s+)?', re.I), [
        'Show ', 'Find all ', 'Get all ', 'Return all ',
        'Display all ', 'What are the ',
    ]),
    (re.compile(r'^Find\s+(?:all\s+)?', re.I), [
        'Show ', 'List ', 'Get ', 'Return ', 'Display ', 'Retrieve ',
    ]),
    (re.compile(r'^What\s+is\s+(?:the\s+)?', re.I), [
        'Show me the ', 'Find the ', 'Get the ', 'Return the ',
    ]),
    (re.compile(r'^What\s+are\s+(?:the\s+)?', re.I), [
        'Show me the ', 'List ', 'Find all ', 'Return all ',
    ]),
    (re.compile(r'^How\s+many\s+', re.I), [
        'Count the number of ', 'What is the count of ',
        'What is the total number of ',
    ]),
    (re.compile(r'^Give\s+(?:me\s+)?(?:the\s+)?', re.I), [
        'Show me the ', 'List ', 'Return ', 'Find ',
    ]),
]

# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema(db_id: str, schemas_dir: str) -> dict:
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        if tidx == -1:
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema


# ── Paraphrase helpers ────────────────────────────────────────────────────────

def paraphrase_question(question: str, rng: random.Random, n: int = 5) -> list:
    """Generate up to n distinct paraphrases of question using rule-based rewrites."""
    variants = set()

    # Strategy A: apply prefix swaps
    for pat, replacements in _PREFIX_SWAPS:
        m = pat.match(question)
        if m:
            rest = question[m.end():]
            for repl in rng.sample(replacements, min(len(replacements), 4)):
                variants.add(repl + rest)
            break  # only one prefix swap per question

    # Strategy B: apply mid-sentence rewrites
    for pat, repl in rng.sample(_REWRITE_RULES, min(len(_REWRITE_RULES), 10)):
        new_q = pat.sub(repl, question, count=1)
        if new_q != question:
            variants.add(new_q)

    # Strategy C: combine prefix swap with mid-sentence rewrite
    already = list(variants)
    for v in already[:3]:
        for pat, repl in rng.sample(_REWRITE_RULES, min(len(_REWRITE_RULES), 5)):
            new_q = pat.sub(repl, v, count=1)
            if new_q != v and new_q != question:
                variants.add(new_q)

    variants.discard(question)
    result = list(variants)
    rng.shuffle(result)
    return result[:n]


# ── Column / table order shuffling ───────────────────────────────────────────

def shuffle_schema_order(item: dict, schema: dict, rng: random.Random) -> dict:
    """Return a copy of item with schema_links table/column order randomised."""
    new_links = {}
    tables = list(item['schema_links'].keys())
    rng.shuffle(tables)
    for t in tables:
        cols = list(item['schema_links'][t])
        rng.shuffle(cols)
        new_links[t] = cols
    new = dict(item)
    new['schema_links'] = new_links
    return new


# ── Single-table subset examples ─────────────────────────────────────────────

def make_single_table_example(original: dict, table: str, cols: list,
                               base_qid: int) -> dict:
    """Create a focused training example that only references one table."""
    col_str = ', '.join(cols[:3]) if cols else '(all columns)'
    question = f"Show all {col_str} from {table}."
    return {
        'question_id': base_qid,
        'db_id':       original['db_id'],
        'question':    question,
        'schema_links': {table: cols},
    }


# ── Cross-table join examples ─────────────────────────────────────────────────

def make_join_example(ex1: dict, ex2: dict, base_qid: int, rng: random.Random) -> dict:
    """Combine two single-table examples into a two-table join question."""
    links = {}
    links.update(ex1['schema_links'])
    links.update(ex2['schema_links'])
    if len(links) < 2:
        return None

    tables = list(links.keys())
    # Simple join-style question
    question = (
        f"Show {ex1['question'].lower().rstrip('.')} "
        f"and {ex2['question'].lower().rstrip('.')}, joining through related tables."
    )
    return {
        'question_id': base_qid,
        'db_id':       ex1['db_id'],
        'question':    question,
        'schema_links': links,
    }


# ── Synthetic questions from schema ──────────────────────────────────────────

_COUNT_TEMPLATES = [
    "How many {table} records are there?",
    "What is the total number of entries in {table}?",
    "Count all rows in {table}.",
    "How many rows does {table} have?",
]

_FILTER_TEMPLATES = [
    "Show all {col} values from {table}.",
    "List the {col} for each entry in {table}.",
    "Return the {col} from {table}.",
    "What are the {col} values in {table}?",
    "Get the {col} field from {table}.",
]

_MULTI_COL_TEMPLATES = [
    "Show {cols} from {table}.",
    "List {cols} for all records in {table}.",
    "Return {cols} from {table}.",
    "What are the {cols} of each {table} entry?",
    "Get {cols} from {table}.",
]

_AGG_TEMPLATES = [
    "What is the maximum {col} in {table}?",
    "Find the minimum {col} from {table}.",
    "What is the average {col} across all {table} records?",
    "Show the highest {col} value in {table}.",
]

_COND_TEMPLATES = [
    "Show {col} from {table} where {col2} is specified.",
    "List {col} for {table} entries that have a {col2}.",
    "Return {col} from {table} filtered by {col2}.",
    "Find {table} records where {col2} is set, and return {col}.",
]


def generate_synthetic_examples(db_id: str, schema: dict,
                                  rng: random.Random, n: int = 10) -> list:
    """Generate n synthetic examples for db_id using schema structure."""
    examples = []
    tables = list(schema.keys())
    if not tables:
        return examples

    qid_counter = 900000 + abs(hash(db_id)) % 10000

    for _ in range(n * 3):
        if len(examples) >= n:
            break

        table = rng.choice(tables)
        cols  = schema[table]
        if not cols:
            continue

        roll = rng.random()

        if roll < 0.15:
            # COUNT query (table only, empty cols)
            q = rng.choice(_COUNT_TEMPLATES).format(table=table)
            links = {table: []}

        elif roll < 0.40:
            # Single-column FILTER
            col = rng.choice(cols)
            q = rng.choice(_FILTER_TEMPLATES).format(col=col, table=table)
            links = {table: [col]}

        elif roll < 0.65:
            # Multi-column
            k = min(rng.randint(2, 4), len(cols))
            sel_cols = rng.sample(cols, k)
            cols_str = ', '.join(sel_cols)
            q = rng.choice(_MULTI_COL_TEMPLATES).format(cols=cols_str, table=table)
            links = {table: sel_cols}

        elif roll < 0.80:
            # AGG query
            col = rng.choice(cols)
            q = rng.choice(_AGG_TEMPLATES).format(col=col, table=table)
            links = {table: [col]}

        else:
            # Conditional (two columns from same table)
            if len(cols) < 2:
                continue
            col, col2 = rng.sample(cols, 2)
            q = rng.choice(_COND_TEMPLATES).format(col=col, col2=col2, table=table)
            links = {table: [col, col2]}

        examples.append({
            'question_id': qid_counter,
            'db_id':       db_id,
            'question':    q,
            'schema_links': links,
        })
        qid_counter += 1

    return examples


# ── Main augmentation pipeline ────────────────────────────────────────────────

def augment(
    train_data: list,
    schemas_dir: str,
    target: int,
    strategy: str,
    rng: random.Random,
) -> list:
    # Count examples per schema
    schema_counts = defaultdict(int)
    schema_examples = defaultdict(list)
    for item in train_data:
        schema_counts[item['db_id']] += 1
        schema_examples[item['db_id']].append(item)

    # Load all schemas
    schema_cache = {}
    for db_id in schema_examples:
        try:
            schema_cache[db_id] = load_schema(db_id, schemas_dir)
        except FileNotFoundError:
            pass

    augmented = list(train_data)  # start with originals
    next_qid  = max(x['question_id'] for x in train_data) + 1

    def next_id():
        nonlocal next_qid
        qid = next_qid
        next_qid += 1
        return qid

    # ── Strategy 1: paraphrase ────────────────────────────────────────────────
    if strategy in ('all', 'paraphrase'):
        for item in train_data:
            db_id   = item['db_id']
            # More paraphrases for under-represented schemas
            n_para  = 10 if schema_counts[db_id] < MIN_EXAMPLES else 4
            for pq in paraphrase_question(item['question'], rng, n=n_para):
                new = dict(item)
                new['question']    = pq
                new['question_id'] = next_id()
                augmented.append(new)

    # ── Strategy 2: schema order shuffling (table + column) ──────────────────
    if strategy in ('all', 'shuffle'):
        for item in train_data:
            db_id = item['db_id']
            n_shuf = 6 if schema_counts[db_id] < MIN_EXAMPLES else 3
            for _ in range(n_shuf):
                new = shuffle_schema_order(item, schema_cache.get(db_id, {}), rng)
                new['question_id'] = next_id()
                augmented.append(new)

    # ── Strategy 3: synthetic single-table examples ───────────────────────────
    if strategy in ('all', 'synthetic'):
        for db_id, schema in schema_cache.items():
            # Double up on under-represented schemas
            n_syn = 30 if schema_counts[db_id] < MIN_EXAMPLES else 8
            for ex in generate_synthetic_examples(db_id, schema, rng, n=n_syn):
                ex['question_id'] = next_id()
                augmented.append(ex)

    # ── Strategy 4: cross-question join examples ──────────────────────────────
    if strategy in ('all', 'cross'):
        for db_id, examples in schema_examples.items():
            # Only create join examples when there are at least 2 distinct tables
            multi = [e for e in examples if len(e['schema_links']) == 1]
            if len(multi) < 2:
                continue
            n_joins = 8 if schema_counts[db_id] < MIN_EXAMPLES else 3
            for _ in range(n_joins):
                e1, e2 = rng.sample(multi, 2)
                t1 = list(e1['schema_links'].keys())[0]
                t2 = list(e2['schema_links'].keys())[0]
                if t1 == t2:
                    continue
                join_ex = make_join_example(e1, e2, next_id(), rng)
                if join_ex:
                    augmented.append(join_ex)

    # ── Strategy 5: weighted oversampling of under-represented schemas ────────
    if strategy in ('all', 'oversample'):
        under = [db_id for db_id, cnt in schema_counts.items() if cnt < MIN_EXAMPLES]
        if under:
            pool = [x for x in augmented if x['db_id'] in set(under)]
            while len(augmented) < target and pool:
                item = rng.choice(pool)
                # Paraphrase while oversampling to avoid exact duplicates
                pqs = paraphrase_question(item['question'], rng, n=1)
                new = dict(item)
                new['question']    = pqs[0] if pqs else item['question']
                new['question_id'] = next_id()
                augmented.append(new)

    # ── Trim / pad to target ──────────────────────────────────────────────────
    if len(augmented) > target:
        # Keep all originals, sample from the rest
        originals = list(train_data)
        extras    = augmented[len(train_data):]
        rng.shuffle(extras)
        augmented = originals + extras[:target - len(originals)]

    rng.shuffle(augmented)
    return augmented


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json',   default=TRAIN_JSON)
    ap.add_argument('--schemas_dir',  default=SCHEMAS_DIR)
    ap.add_argument('--out',          default=OUT_JSON)
    ap.add_argument('--target',       type=int, default=TARGET)
    ap.add_argument('--strategy',     default='all',
                    choices=['all', 'paraphrase', 'shuffle',
                             'synthetic', 'cross', 'oversample'])
    ap.add_argument('--seed',         type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.train_json} ...")
    with open(args.train_json) as f:
        train_data = json.load(f)
    print(f"  {len(train_data)} original examples")

    from collections import Counter
    dist = Counter(x['db_id'] for x in train_data)
    print("Schema distribution (sorted by count):")
    for db_id, cnt in sorted(dist.items(), key=lambda x: x[1]):
        flag = '  *** UNDER-REPRESENTED' if cnt < MIN_EXAMPLES else ''
        print(f"  {db_id}: {cnt}{flag}")

    print(f"\nRunning augmentation (strategy={args.strategy}, target={args.target}) ...")
    augmented = augment(
        train_data=train_data,
        schemas_dir=args.schemas_dir,
        target=args.target,
        strategy=args.strategy,
        rng=rng,
    )

    # Report final distribution
    aug_dist = Counter(x['db_id'] for x in augmented)
    print(f"\nAugmented dataset: {len(augmented)} examples")
    print("Final schema distribution:")
    for db_id, cnt in sorted(aug_dist.items(), key=lambda x: x[1]):
        print(f"  {db_id}: {cnt}")

    print(f"\nWriting to {args.out} ...")
    with open(args.out, 'w') as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == '__main__':
    main()
