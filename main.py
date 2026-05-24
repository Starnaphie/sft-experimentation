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
BASE_MODEL     = 'Qwen/Qwen2.5-0.5B-Instruct'
ADAPTER_DIR    = './adapter'
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a database assistant. "
    "Given a database schema and a natural language question, output the schema links "
    "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
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


def serialize_schema(schema: dict) -> str:
    """Compact schema string — must match train.py's formatting_function exactly."""
    lines = []
    for table, cols in schema.items():
        lines.append(f"  {table}({', '.join(cols)})" if cols else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


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

    def __init__(self, base_model: str, adapter_dir: str):
        # Lazy imports: this class is only instantiated when the adapter exists,
        # so users without torch/peft can still run the keyword baseline.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer: prefer what's saved in the adapter dir (train.py may
        # have copied it there); fall back to the HF hub base model.
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

    def predict(self, question: str, schema: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"{serialize_schema(schema)}\n\nQuestion: {question}"},
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


# ── Adapter detection ─────────────────────────────────────────────────────────

def _adapter_ready(adapter_dir: str) -> bool:
    if not os.path.isdir(adapter_dir):
        return False
    return any(f.endswith(('.safetensors', '.bin'))
               for f in os.listdir(adapter_dir))


# ── Per-question entry point ──────────────────────────────────────────────────

def predict_schema_links(question: str, db_id: str, schemas_dir: str,
                         predictor=None) -> dict:
    schema = load_schema(db_id, schemas_dir)
    if predictor is not None:
        raw = predictor.predict(question, schema)
        return filter_against_schema(raw, schema)
    return _keyword_predict(question, schema)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',       required=True)
    ap.add_argument('--output',      required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    ap.add_argument('--adapter_dir', default=ADAPTER_DIR)
    ap.add_argument('--base_model',  default=BASE_MODEL)
    args = ap.parse_args()

    if _adapter_ready(args.adapter_dir):
        print(f"[mode] fine-tuned model")
        predictor = ModelPredictor(args.base_model, args.adapter_dir)
    else:
        print(f"[mode] keyword-matching baseline  "
              f"(no adapter at '{args.adapter_dir}' — run train.py first)")
        predictor = None

    with open(args.input) as f:
        items = json.load(f)

    preds = []
    for i, it in enumerate(items, 1):
        links = predict_schema_links(
            it['question'], it['db_id'], args.schemas_dir, predictor)
        preds.append({'question_id': it['question_id'], 'schema_links': links})
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)} predicted")

    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions → {args.output}")


if __name__ == '__main__':
    main()
