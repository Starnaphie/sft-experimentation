"""
augment.py -- Training data augmentation for schema linking (v4).

DESIGN PRINCIPLE (learned the hard way from v3)
-----------------------------------------------
The validation questions use NATURAL, INDIRECT domain language and NEVER name
raw schema identifiers, e.g.:
    "How many crashes involved vehicles equipped with crash avoidance features?"
The model's job is to MAP that language onto the right tables/columns.

v3 flooded training with synthetic/SQL-templated/hard-negative examples whose
QUESTIONS literally contained the gold table codes and column names
("What is the number of CTR1 items per OCTR record?"). That taught a degenerate
shortcut -- copy identifiers seen in the question -- which is useless on
validation and DROPPED the score. 65% of v3 was this junk.

v4 therefore uses ONLY transformations that keep questions in natural language
and never inject schema identifiers:
  paraphrase  -- surface rewrites of the REAL question (synonyms, reordering)
  shuffle     -- randomise table/column ORDER in schema_links (question untouched)
  oversample  -- extra paraphrases for under-represented schemas
All "synthetic", "sql_variants", "hard_negative", and "cross" generators from v3
are REMOVED -- they injected raw identifiers.

LLM paraphrasing (optional, off by default; requires ANTHROPIC_API_KEY) is the
only safe way to add genuinely new natural phrasings; rule-based paraphrasing is
the default engine.

Usage:
    python augment.py                          # writes augmented_train_v4.json
    python augment.py --target 3000 --seed 99
    python augment.py --use_llm_paraphrase     # add Claude Haiku paraphrases
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict, Counter

TRAIN_JSON       = './train.json'
SCHEMAS_DIR      = './schemas'
OUT_JSON         = './augmented_train_v4.json'
PARAPHRASE_CACHE = './paraphrase_cache.json'
TARGET           = 3000
MIN_EXAMPLES     = 24   # schemas with fewer get extra paraphrases

# ── Question paraphrase rules ─────────────────────────────────────────────────
# CRITICAL: these only rewrite SURFACE wording of an already-natural question.
# They never introduce table/column identifiers, so paraphrases stay on the
# same natural-language distribution as the validation set.

_REWRITE_RULES = [
    (re.compile(r'^Show(?:\s+me)?\s+', re.I),         'List '),
    (re.compile(r'^List\s+(?:all\s+)?', re.I),        'Show '),
    (re.compile(r'^Display\s+', re.I),                'Show me '),
    (re.compile(r'^Return\s+', re.I),                 'List '),
    (re.compile(r'^Find\s+', re.I),                   'Show '),
    (re.compile(r'^Get\s+', re.I),                    'Find '),
    (re.compile(r'^What\s+(?:are|is)\s+(?:the\s+)?', re.I), 'Show me the '),
    (re.compile(r'^Retrieve\s+', re.I),               'Get '),
    (re.compile(r'^Tell me\s+', re.I),                'Show me '),
    (re.compile(r'^Provide\s+', re.I),                'Give me '),
    # Count variants
    (re.compile(r'\bcount\s+of\b', re.I),             'number of'),
    (re.compile(r'\bnumber\s+of\b', re.I),            'count of'),
    (re.compile(r'\bhow many\b', re.I),               'the count of'),
    (re.compile(r'\bthe count of\b', re.I),           'how many'),
    (re.compile(r'\btotal\s+number\s+of\b', re.I),    'how many'),
    (re.compile(r'\btally of\b', re.I),               'count of'),
    # Filter variants
    (re.compile(r'\bwhere\b', re.I),                  'in which'),
    (re.compile(r'\bin which\b', re.I),               'where'),
    (re.compile(r'\bthat have\b', re.I),              'having'),
    (re.compile(r'\bhaving\b', re.I),                 'that have'),
    (re.compile(r'\bwith a\b', re.I),                 'that have a'),
    (re.compile(r'\bfor each\b', re.I),               'per'),
    (re.compile(r'\bper\b', re.I),                    'for each'),
    (re.compile(r'\bfor every\b', re.I),              'for each'),
    # Sort / order variants
    (re.compile(r'\bordered by\b', re.I),             'sorted by'),
    (re.compile(r'\bsorted by\b', re.I),              'ordered by'),
    (re.compile(r'\branked by\b', re.I),              'sorted by'),
    (re.compile(r'\barranged by\b', re.I),            'ordered by'),
    # Aggregate variants
    (re.compile(r'\baverage\b', re.I),                'mean'),
    (re.compile(r'\bmean\b', re.I),                   'average'),
    (re.compile(r'\bmaximum\b', re.I),                'highest'),
    (re.compile(r'\bhighest\b', re.I),                'largest'),
    (re.compile(r'\blargest\b', re.I),                'maximum'),
    (re.compile(r'\bminimum\b', re.I),                'lowest'),
    (re.compile(r'\blowest\b', re.I),                 'smallest'),
    (re.compile(r'\bsmallest\b', re.I),               'minimum'),
    (re.compile(r'\btotal\b', re.I),                  'sum'),
    # Distinct / unique variants
    (re.compile(r'\bdistinct\b', re.I),               'unique'),
    (re.compile(r'\bunique\b', re.I),                 'distinct'),
    (re.compile(r'\bdifferent\b', re.I),              'distinct'),
    # Group by variants
    (re.compile(r'\bgroup(?:ed)?\s+by\b', re.I),      'broken down by'),
    (re.compile(r'\bbroken\s+down\s+by\b', re.I),     'grouped by'),
    # Comparison variants
    (re.compile(r'\bis\s+equal\s+to\b', re.I),        'equals'),
    (re.compile(r'\bequals\b', re.I),                 'is equal to'),
    (re.compile(r'\bis\s+greater\s+than\b', re.I),    'exceeds'),
    (re.compile(r'\bexceeds\b', re.I),                'is greater than'),
    (re.compile(r'\bis\s+less\s+than\b', re.I),       'is below'),
    (re.compile(r'\bis\s+below\b', re.I),             'is less than'),
    (re.compile(r'\bat least\b', re.I),               'no less than'),
    (re.compile(r'\bat most\b', re.I),                'no more than'),
    # Misc natural swaps
    (re.compile(r'\bvalues\b', re.I),                 'entries'),
    (re.compile(r'\brecords\b', re.I),                'rows'),
    (re.compile(r'\brows\b', re.I),                   'records'),
]

_PREFIX_SWAPS = [
    (re.compile(r'^Show(?:\s+me)?\s+(?:the\s+)?', re.I), [
        'List ', 'Find ', 'Get ', 'Return ', 'Display ',
        'What are the ', 'Retrieve ', 'Give me the ',
    ]),
    (re.compile(r'^List\s+(?:all\s+)?', re.I), [
        'Show ', 'Find all ', 'Get all ', 'Return all ',
        'Display all ', 'What are the ',
    ]),
    (re.compile(r'^Find\s+(?:all\s+)?', re.I), [
        'Show ', 'List ', 'Get ', 'Return ', 'Display ', 'Retrieve ',
    ]),
    (re.compile(r'^What\s+is\s+(?:the\s+)?', re.I), [
        'Show me the ', 'Find the ', 'Get the ', 'Return the ', 'Give me the ',
    ]),
    (re.compile(r'^What\s+are\s+(?:the\s+)?', re.I), [
        'Show me the ', 'List ', 'Find all ', 'Return all ', 'Give me the ',
    ]),
    (re.compile(r'^How\s+many\s+', re.I), [
        'Count the number of ', 'What is the count of ',
        'What is the total number of ', 'Tell me how many ',
    ]),
    (re.compile(r'^Give\s+(?:me\s+)?(?:the\s+)?', re.I), [
        'Show me the ', 'List ', 'Return ', 'Find ',
    ]),
    (re.compile(r'^Which\s+', re.I), [
        'What ', 'Identify which ', 'Tell me which ',
    ]),
]


# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema(db_id: str, schemas_dir: str) -> dict:
    """Returns {table: [col, ...]} using original-case identifiers."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        if tidx == -1:
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema


# ── Rule-based paraphrase ─────────────────────────────────────────────────────

def paraphrase_question(question: str, rng: random.Random, n: int = 5) -> list:
    """Generate up to n distinct natural paraphrases via surface rewrites.

    Never introduces schema identifiers -- only rewrites the wording of the
    original (already natural) question, so paraphrases stay on-distribution.
    """
    variants = set()

    # A: prefix swaps
    for pat, replacements in _PREFIX_SWAPS:
        m = pat.match(question)
        if m:
            rest = question[m.end():]
            for repl in rng.sample(replacements, min(len(replacements), 5)):
                variants.add(repl + rest)
            break

    # B: single mid-sentence rewrites
    for pat, repl in rng.sample(_REWRITE_RULES, min(len(_REWRITE_RULES), 14)):
        new_q = pat.sub(repl, question, count=1)
        if new_q != question:
            variants.add(new_q)

    # C: stack a second rewrite on top of a few A/B variants for more diversity
    already = list(variants)
    for v in already[:5]:
        for pat, repl in rng.sample(_REWRITE_RULES, min(len(_REWRITE_RULES), 6)):
            new_q = pat.sub(repl, v, count=1)
            if new_q != v and new_q != question:
                variants.add(new_q)

    variants.discard(question)
    result = list(variants)
    rng.shuffle(result)
    return result[:n]


# ── LLM-based paraphrasing (optional, Anthropic API) ─────────────────────────

_LLM_SYSTEM = (
    "You are a data-augmentation assistant for a text-to-SQL schema linking task.\n"
    "Given a natural language database question, generate exactly 5 diverse paraphrases.\n"
    "Rules:\n"
    "- Keep the EXACT same meaning and the same underlying data being requested.\n"
    "- Use natural, indirect domain language. Do NOT invent table or column names.\n"
    "- Only change wording, vocabulary, and sentence structure.\n"
    "- Output exactly 5 lines numbered 1-5 with NO other text."
)


def _parse_numbered_list(text: str, n: int) -> list:
    out = []
    for line in text.strip().split('\n'):
        m = re.match(r'^\d+[.)]\s*(.+)', line.strip())
        if m:
            out.append(m.group(1).strip())
    return out[:n]


def llm_paraphrase_questions(items, api_key, cache_file, n_variants=5, verbose=True):
    """Call Claude Haiku to paraphrase each unique question; cache results."""
    import anthropic

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    unique = {item['question'] for item in items}
    to_fetch = [q for q in unique if q not in cache]
    if verbose:
        print(f"  LLM paraphrase: {len(cache)} cached, {len(to_fetch)} to fetch")

    for i, q in enumerate(to_fetch):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=[{"type": "text", "text": _LLM_SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Question: {q}"}],
            )
            cache[q] = _parse_numbered_list(resp.content[0].text, n_variants)
        except Exception as e:
            if verbose:
                print(f"  API error on q[{i}]: {e}")
            cache[q] = []
        if (i + 1) % 10 == 0:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"    ... {i+1}/{len(to_fetch)} fetched")

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    return cache


# ── Schema order shuffling ────────────────────────────────────────────────────

def shuffle_schema_order(item: dict, rng: random.Random) -> dict:
    """Randomise table + column order in schema_links. Question is untouched."""
    new_links = {}
    tables = list(item['schema_links'].keys())
    rng.shuffle(tables)
    for t in tables:
        cols = list(item['schema_links'][t])
        rng.shuffle(cols)
        new_links[t] = cols
    return {**item, 'schema_links': new_links}


# ── Quality filtering ─────────────────────────────────────────────────────────

def filter_quality(examples: list, schema_cache: dict) -> list:
    """Remove:
      - empty schema_links
      - exact (question, schema_links_json) duplicates  (order-sensitive, so
        differently-ordered shuffles survive as distinct training targets)
      - schema_links referencing tables/columns not in the schema
    """
    seen = set()
    filtered = []
    for ex in examples:
        links = ex.get('schema_links', {})
        if not links:
            continue
        key = (ex['question'].strip(), json.dumps(links, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)

        schema = schema_cache.get(ex['db_id'])
        if schema is not None:
            valid = True
            for table, cols in links.items():
                if table not in schema:
                    valid = False
                    break
                schema_cols = set(schema[table])
                for col in cols:
                    if col and col not in schema_cols:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
        filtered.append(ex)
    return filtered


# ── Main augmentation pipeline ────────────────────────────────────────────────

def augment(train_data, schemas_dir, target, rng,
            use_llm_paraphrase=False, api_key='', cache_file=PARAPHRASE_CACHE):
    schema_counts = defaultdict(int)
    for item in train_data:
        schema_counts[item['db_id']] += 1

    schema_cache = {}
    for db_id in schema_counts:
        try:
            schema_cache[db_id] = load_schema(db_id, schemas_dir)
        except FileNotFoundError:
            pass

    augmented = list(train_data)
    next_qid = max(x['question_id'] for x in train_data) + 1

    def add(item):
        nonlocal next_qid
        item = dict(item)
        item['question_id'] = next_qid
        next_qid += 1
        augmented.append(item)

    # ── LLM paraphrase (optional) ─────────────────────────────────────────────
    if use_llm_paraphrase and api_key:
        print("Running LLM paraphrasing (Anthropic API) ...")
        llm_cache = llm_paraphrase_questions(train_data, api_key, cache_file)
        for item in train_data:
            for pq in llm_cache.get(item['question'], []):
                if pq and pq != item['question']:
                    add({**item, 'question': pq})
    elif use_llm_paraphrase and not api_key:
        print("  WARNING: --use_llm_paraphrase set but ANTHROPIC_API_KEY not found; skipping.")

    # ── Rule-based paraphrase ─────────────────────────────────────────────────
    for item in train_data:
        n_para = 14 if schema_counts[item['db_id']] < MIN_EXAMPLES else 7
        for pq in paraphrase_question(item['question'], rng, n=n_para):
            add({**item, 'question': pq})

    # ── Schema-order shuffling ────────────────────────────────────────────────
    for item in train_data:
        n_shuf = 4 if schema_counts[item['db_id']] < MIN_EXAMPLES else 2
        for _ in range(n_shuf):
            add(shuffle_schema_order(item, rng))

    # ── Oversample under-represented schemas (paraphrase-based) ───────────────
    # Buffer past target since the quality filter will dedup some.
    under = {db_id for db_id, cnt in schema_counts.items() if cnt < MIN_EXAMPLES}
    if under:
        pool = [x for x in augmented if x['db_id'] in under]
        guard = 0
        while len(augmented) < int(target * 1.5) and pool and guard < target * 5:
            guard += 1
            item = rng.choice(pool)
            pqs = paraphrase_question(item['question'], rng, n=3)
            new_q = pqs[rng.randrange(len(pqs))] if pqs else item['question']
            add({**item, 'question': new_q})

    # ── Quality filter ────────────────────────────────────────────────────────
    print(f"  Before quality filter: {len(augmented)} examples")
    augmented = filter_quality(augmented, schema_cache)
    print(f"  After  quality filter: {len(augmented)} examples")

    # ── Trim to target (always keep originals) ────────────────────────────────
    if len(augmented) > target:
        orig_qids = {x['question_id'] for x in train_data}
        originals = [x for x in augmented if x['question_id'] in orig_qids]
        extras    = [x for x in augmented if x['question_id'] not in orig_qids]
        rng.shuffle(extras)
        augmented = originals + extras[:max(0, target - len(originals))]

    rng.shuffle(augmented)
    return augmented


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json',         default=TRAIN_JSON)
    ap.add_argument('--schemas_dir',        default=SCHEMAS_DIR)
    ap.add_argument('--out',                default=OUT_JSON)
    ap.add_argument('--target',             type=int, default=TARGET)
    ap.add_argument('--seed',               type=int, default=42)
    ap.add_argument('--use_llm_paraphrase', action='store_true',
                    help='Add Claude Haiku paraphrases (needs ANTHROPIC_API_KEY).')
    ap.add_argument('--paraphrase_cache',   default=PARAPHRASE_CACHE)
    args = ap.parse_args()

    rng     = random.Random(args.seed)
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    print(f"Loading {args.train_json} ...")
    with open(args.train_json) as f:
        train_data = json.load(f)
    print(f"  {len(train_data)} original examples")

    dist = Counter(x['db_id'] for x in train_data)
    print("Schema distribution (sorted by count):")
    for db_id, cnt in sorted(dist.items(), key=lambda x: x[1]):
        flag = '  *** UNDER-REPRESENTED' if cnt < MIN_EXAMPLES else ''
        print(f"  {db_id}: {cnt}{flag}")

    print(f"\nRunning augmentation (target={args.target}, llm={args.use_llm_paraphrase}) ...")
    augmented = augment(
        train_data         = train_data,
        schemas_dir        = args.schemas_dir,
        target             = args.target,
        rng                = rng,
        use_llm_paraphrase = args.use_llm_paraphrase,
        api_key            = api_key,
        cache_file         = args.paraphrase_cache,
    )

    aug_dist = Counter(x['db_id'] for x in augmented)
    print(f"\nAugmented dataset: {len(augmented)} examples")
    print("Final schema distribution:")
    for db_id, cnt in sorted(aug_dist.items(), key=lambda x: x[1]):
        flag = '  ***' if cnt < MIN_EXAMPLES else ''
        print(f"  {db_id}: {cnt}{flag}")

    print(f"\nWriting to {args.out} ...")
    with open(args.out, 'w') as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == '__main__':
    main()
