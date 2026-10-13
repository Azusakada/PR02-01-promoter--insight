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
    parser.add_argument('--m3-completion', help='Also verify a frozen M3 completion run and its overwrite guard')
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
    if args.m3_completion:
        completion = (ROOT/args.m3_completion).resolve()
        if not completion.is_relative_to(ROOT/'runs') or not (completion/'run_manifest.json').is_file():
            parser.error('M3 completion run must exist under runs/')
        commands.insert(6, ('m3_completion_verification', [python, '-m', 'pr02', 'verify-m3-completion', '--output', args.m3_completion], 0))
        commands.insert(-1, ('m3_completion_overwrite_guard', [python, '-m', 'pr02', 'complete-m3', '--output', args.m3_completion], 2))
    run = (ROOT/args.run).resolve()
    if not run.is_relative_to(ROOT/'runs') or not (run/'run_manifest.json').exists():
        parser.error('Analysis run must exist under runs/')
    # The checker may write only its work/ logs; data, every prior run and indexes stay frozen.
    frozen_roots = [ROOT/'data', ROOT/'runs', ROOT/'results']
    files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for root in frozen_roots for p in root.rglob('*') if p.is_file()}
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
        if name in ['overwrite_guard', 'm3_completion_overwrite_guard']:
            check['passed'] &= 'existing output' in result.stderr and all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in files.items())
        record['checks'].append(check)
        record['status'] = 'passed' if all(c['passed'] for c in record['checks']) else 'failed'
        (output/'checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'{name}: exit={result.returncode}, passed={check["passed"]}', flush=True)
        if not check['passed']:
            print(result.stdout+result.stderr)
            return 1
    unchanged = set(files) == {p for root in frozen_roots for p in root.rglob('*') if p.is_file()} and all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in files.items())
    record['checks'].append(dict(name='frozen_artifact_integrity', passed=unchanged, files_checked=len(files)))
    record['status'] = 'passed' if all(c['passed'] for c in record['checks']) else 'failed'
    (output/'checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
