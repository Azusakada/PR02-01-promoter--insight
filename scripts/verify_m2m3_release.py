"""Run release checks without retraining or modifying frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New test-record folder under work/')
    parser.add_argument('--run', default='runs/ecoli87_m2m3_20261013_v1')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'work') or output.exists():
        parser.error('Use a new directory inside work/')
    output.mkdir(parents=True)
    python = sys.executable
    commands = [
        ('root_tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 0),
        ('cnn_tests', [python, '-m', 'unittest', 'discover', '-s', 'CNN/tests', '-v'], 0),
        ('legacy_analysis_tests', [python, '-m', 'unittest', 'discover', '-s', 'analysis_m2m3/tests', '-v'], 0),
        ('contracts_tests', [python, 'contracts/scripts/test_contracts.py'], 0),
        ('ridge_kmer_checks', [python, 'Ridge/src/test_kmer.py'], 0),
        ('supplementary_verification', [python, '-m', 'pr02', 'verify-m2m3', '--output', args.run], 0),
        ('project_verification', [python, '-c', 'import json; from pr02.context87.data import ROOT,read_json; from pr02.context87.audit import verify; print(json.dumps(verify(read_json(ROOT/"project_config.json"),visual=True),indent=2))'], 0),
        ('overwrite_guard', [python, '-m', 'pr02', 'analyze-m2m3', '--output', args.run], 2),
        ('whitespace_check', ['git', 'diff', '--check'], 0),
    ]
    run = (ROOT/args.run).resolve()
    if not run.is_relative_to(ROOT/'runs') or not (run/'run_manifest.json').exists():
        parser.error('Analysis run must exist under runs/')
    files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    record = {'created_at': datetime.now(timezone.utc).isoformat(),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checks': []}
    for name, command, expected in commands:
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log = output/f'{name}.log'; log.write_text(result.stdout+result.stderr, encoding='utf-8')
        match = re.search(r'Ran (\d+) tests?', result.stderr+result.stdout)
        check = {'name': name, 'command': command, 'exit_code': result.returncode, 'expected_exit_code': expected,
                 'passed': result.returncode == expected, 'elapsed_seconds': round(time.monotonic()-start, 3),
                 'log': log.name, 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
        if match:
            check['tests_run'] = int(match.group(1))
        if name == 'ridge_kmer_checks':
            check['checks_run'] = 4
        if name == 'overwrite_guard':
            check['passed'] &= 'existing output' in result.stderr and all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in files.items())
        record['checks'].append(check)
        record['status'] = 'passed' if all(c['passed'] for c in record['checks']) else 'failed'
        (output/'checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'{name}: exit={result.returncode}, passed={check["passed"]}', flush=True)
        if not check['passed']:
            print(result.stdout+result.stderr)
            return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
