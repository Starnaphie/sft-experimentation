"""
exp1.py — Track A: run all three prompt-engineering experiments sequentially.

Each experiment runs as a separate subprocess with a clean RF state so that
stale experiment entries from prior killed runs can never cause a deadlock.

Experiments:
  fewshot       → adapter_ta_fewshot/       (compact schema + in-context example)
  typed         → adapter_ta_typed/         (compact schema with :type annotations)
  fewshot_typed → adapter_ta_fewshot_typed/ (typed schema + in-context example)

Run (leave it overnight):
    python exp1.py

Evaluate after all three finish:
    python main.py --input validation_input.json --output preds_fewshot.json \
                   --adapter_dir ./adapter_ta_fewshot --schema_format fewshot
    python eval.py --predictions preds_fewshot.json \
                   --gold validation_gold_schema_links.json \
                   --schemas_dir schemas/ --questions_input validation_input.json

    python main.py --input validation_input.json --output preds_typed.json \
                   --adapter_dir ./adapter_ta_typed --schema_format typed
    python eval.py --predictions preds_typed.json \
                   --gold validation_gold_schema_links.json \
                   --schemas_dir schemas/ --questions_input validation_input.json

    python main.py --input validation_input.json --output preds_fewshot_typed.json \
                   --adapter_dir ./adapter_ta_fewshot_typed --schema_format fewshot_typed
    python eval.py --predictions preds_fewshot_typed.json \
                   --gold validation_gold_schema_links.json \
                   --schemas_dir schemas/ --questions_input validation_input.json
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ('exp1_fewshot.py',       'fewshot',       './adapter_ta_fewshot'),
    ('exp1_typed.py',         'typed',         './adapter_ta_typed'),
    ('exp1_fewshot_typed.py', 'fewshot_typed', './adapter_ta_fewshot_typed'),
]


def clear_rf_state():
    """Kill orphaned RF/MLflow processes, then remove ~/rapidfireai/ state."""
    # RF starts a local MLflow server on port 8852; killing the parent RF process
    # leaves this server orphaned and returning 503 to the next RF instance.
    import signal
    try:
        result = subprocess.run(
            ['lsof', '-t', '-i', ':8852'], capture_output=True, text=True)
        for pid_str in result.stdout.strip().split():
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    except Exception:
        pass

    rf_dir = Path.home() / "rapidfireai"
    if not rf_dir.exists():
        return
    try:
        shutil.rmtree(rf_dir)
        print("  [rf] cleared ~/rapidfireai/")
    except Exception:
        for sub in ['db', 'logs', 'rapidfire_experiments']:
            shutil.rmtree(rf_dir / sub, ignore_errors=True)


def main():
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Force MLflow to use local file storage instead of any remote REST server.
    # On some cluster nodes RF is configured to talk to a remote MLflow server;
    # that server doesn't know about our fresh run IDs and crashes metric logging.
    mlruns = Path.home() / "rapidfireai" / "mlruns"
    env['MLFLOW_TRACKING_URI'] = f"file://{mlruns}"

    results = {}

    for script, fmt_name, adapter_dir in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {fmt_name}")
        print(f"Script    : {script}")
        print(f"Adapter   : {adapter_dir}")
        print('='*60)

        # Always start with clean RF state to prevent "currently running" stale errors
        clear_rf_state()

        ret = subprocess.run([sys.executable, script], env=env)

        if ret.returncode == 0:
            adapter_ok = os.path.isdir(adapter_dir) and any(
                f.endswith(('.safetensors', '.bin'))
                for f in os.listdir(adapter_dir)
            )
            results[fmt_name] = 'OK' if adapter_ok else 'NO ADAPTER'
            print(f"\n  [{fmt_name}] {'adapter saved' if adapter_ok else 'WARNING: adapter not found'}")
        else:
            results[fmt_name] = f'FAILED (exit {ret.returncode})'
            print(f"\n  [{fmt_name}] FAILED — continuing to next experiment")

    print(f"\n{'='*60}")
    print("Summary:")
    for fmt_name, status in results.items():
        print(f"  {fmt_name:20s}  {status}")

    print(f"\n{'='*60}")
    print("Evaluate each adapter:")
    for _, fmt_name, adapter_dir in SCRIPTS:
        if results.get(fmt_name) == 'OK':
            print(f"\n  # {fmt_name}")
            print(f"  python main.py --input validation_input.json "
                  f"--output preds_{fmt_name}.json "
                  f"--adapter_dir {adapter_dir} --schema_format {fmt_name}")
            print(f"  python eval.py --predictions preds_{fmt_name}.json "
                  f"--gold validation_gold_schema_links.json "
                  f"--schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == '__main__':
    main()
