
import os, json, subprocess, sys

# Convenience wrapper to run a tiny optimize+apply sequence
os.makedirs('out', exist_ok=True)
subprocess.check_call([sys.executable, '-m', 'autochunk.cli', 'optimize', 'examples/sample_docs', '--plan', 'out/plan.yaml', '--report', 'out/report.json'])
subprocess.check_call([sys.executable, '-m', 'autochunk.cli', 'apply', 'examples/sample_docs', '--plan', 'out/plan.yaml', '--out', 'out/chunks.jsonl'])

print('Done. Inspect out/plan.yaml and out/chunks.jsonl')
