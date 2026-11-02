"""One scheduler tick for the sixteen ordered RNN runs; never retry an attempt."""
import fcntl
import hashlib
import os
import subprocess
import time
from scripts.run_rnn_shared_length import BASE, REPO, read, save


def active(folder):
    with (folder / 'process.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def training_processes():
    """Conservatively block on any Python training runner, including other queues."""
    rows = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    return [line.strip() for line in rows.splitlines()
            if 'python' in line.lower() and ('scripts.run_' in line or '/run_' in line)
            and int(line.split()[0]) != os.getpid()]


def process_alive(pid):
    if pid is None:
        return False
    row = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
    return row.returncode == 0 and bool(row.stdout.strip()) and not row.stdout.strip().startswith('Z')


def launch(candidate, folder):
    folder.mkdir(exist_ok=False)
    save(folder / 'config.json', dict(candidate=candidate, protocol=str(BASE / 'protocol.json')))
    save(folder / 'status.json', dict(candidate=candidate['id'], status='starting', started_at=time.time(), completed_epochs=0))
    with (folder / 'process.log').open('x') as log:
        proc = subprocess.Popen([str(REPO / '.venv/bin/python'), '-u', '-m', 'scripts.run_rnn_shared_length', '--candidate', candidate['id']], cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(folder / 'launcher.json', dict(pid=proc.pid, started_at=time.time()))
    subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(proc.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return dict(status='starting', pid=proc.pid, new_process_started=True)


def tick():
    with (BASE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(BASE / 'protocol.json')
        attempts = []
        def report(**result):
            result.setdefault('new_process_started', False)
            current = result.get('candidate')
            if current is not None:
                index = protocol['candidates'].index(current)
                result.setdefault('next_condition', protocol['candidates'][index + 1] if index + 1 < len(protocol['candidates']) else None)
            if attempts:
                last = attempts[-1]
                if last.get('status') == 'numerical_failure':
                    result.setdefault('failure', {key: last.get(key) for key in ('T', 'epoch', 'batch', 'reason')})
            save(BASE / 'queue_state.json', dict(phase='ordered_rnn_16', checked_at=time.time(),
                 upstream_commit=protocol['upstream_commit'], safe_adam_path=protocol['safe_adam_path'],
                 safe_adam_sha256=protocol['safe_adam_sha256'], attempts=attempts, **result))
            return result
        candidates = protocol['candidates']
        # Inspect every recorded attempt before launching anything.
        for candidate in candidates:
            folder = BASE / candidate['id']
            if not folder.exists():
                continue
            if not (folder / 'status.json').exists():
                return report(candidate=candidate, status='needs_attention', reason='attempt directory without status')
            state = read(folder / 'status.json')
            attempts.append(dict(condition=candidate, **state))
            launcher = read(folder / 'launcher.json') if (folder / 'launcher.json').exists() else {}
            if active(folder) or process_alive(launcher.get('pid', state.get('pid'))):
                return report(candidate=candidate, status='running', pid=launcher.get('pid', state.get('pid')))
            if state['status'] == 'starting' and time.time() - state['started_at'] < 120:
                return report(candidate=candidate, status='starting')
            if state['status'] not in ('completed', 'numerical_failure'):
                return report(candidate=candidate, status='needs_attention', reason=state.get('reason', 'process stopped without terminal status'))
            if state['status'] == 'completed' and state.get('completed_epochs') != 200:
                return report(candidate=candidate, status='needs_attention', reason='completion without 200 epochs')
        other = training_processes()
        if other:
            return report(status='running', reason='existing training process', processes=other)
        if len(attempts) == len(candidates):
            save(BASE / 'summary.json', dict(upstream_commit=protocol['upstream_commit'], runs=attempts))
            return report(status='all_done', next_condition=None, pause_automation=True)
        for path, digest in protocol['code_sha256'].items():
            if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != digest:
                return report(status='needs_attention', reason='code hash mismatch', path=path)
        if protocol['safe_adam'] != 'grid50_scaled_second_moment_v1':
            return report(status='needs_attention', reason='uploaded Safe Adam not wired')
        for index, candidate in enumerate(candidates):
            folder = BASE / candidate['id']
            if not folder.exists():
                result = launch(candidate, folder)
                attempts.append(dict(condition=candidate, **read(folder / 'status.json')))
                return report(candidate=candidate, next_condition=candidates[index+1] if index+1 < len(candidates) else None, **result)


if __name__ == '__main__':
    import json
    print(json.dumps(tick()))
