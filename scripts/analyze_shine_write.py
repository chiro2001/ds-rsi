#!/usr/bin/env python3
"""CPU-only audit of pinned SHINE outputs and optimizer/parameter movement."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import torch

from dsrsi.training.checkpoint import CheckpointStore, file_sha256
from dsrsi.utils.atomic import atomic_json


def output_summary(records):
    rows = [r for r in records if r['view'] == 'memory']
    index = {(r['context_index'], r['probe_id']): r for r in rows}
    contexts = sorted({c for c, _ in index})
    probes = sorted({p for _, p in index})
    pairs = [(index[c, p], index[c + 1, p]) for c in contexts if c % 2 == 0 for p in probes]
    changed, unchanged, broken, changes = Counter(), Counter(), Counter(), Counter()
    detail = []
    for a, b in pairs:
        assert len(a['expected']) == len(b['expected'])
        positions = [i for i, (x, y) in enumerate(zip(a['expected'], b['expected'])) if x != y]
        assert len(positions) == 1
        k = positions[0]
        category = 'letter' if k == 0 else 'digit_before_dash' if k == 1 else 'suffix_digit'
        changes[category] += 1
        for r in (a, b):
            changed[category] += int(len(r['text']) > k and r['text'][k] == r['expected'][k])
        stable = [i for i in range(len(a['expected'])) if i != k]
        unchanged[category] += int(all(len(r['text']) == len(r['expected']) and
                                     all(r['text'][i] == r['expected'][i] for i in stable) for r in (a, b)))
        broken[category] += int(any(i >= len(a['text']) or i >= len(b['text']) or
                                  a['text'][i] != b['text'][i] for i in stable))
        detail.append(dict(context=a['context_index'], probe=a['probe_id'], changed_character=k,
                           expected=[a['expected'], b['expected']], generated=[a['text'], b['text']],
                           both_exact=a['exact_match'] and b['exact_match']))
    return dict(queries=len(rows), exact=sum(r['exact_match'] for r in rows),
                ended=sum(r['stopped_on_end'] for r in rows),
                character_correct=[sum(len(r['text']) > i and r['text'][i] == r['expected'][i] for r in rows)
                                   for i in range(6)],
                correct_three_digit_suffix=sum(r['text'].split('-')[-1] == r['expected'].split('-')[-1] for r in rows),
                exact_by_probe={p: sum(index[c, p]['exact_match'] for c in contexts) for p in probes},
                pair_both_exact=sum(a['exact_match'] and b['exact_match'] for a, b in pairs),
                pairs=len(pairs), pair_output_changed=sum(a['text'] != b['text'] for a, b in pairs),
                same_answer_across_questions=sum(len({index[c, p]['text'] for p in probes}) == 1 for c in contexts),
                contexts=len(contexts), distinct_outputs=len({r['text'] for r in rows}),
                most_common_outputs=Counter(r['text'] for r in rows).most_common(5),
                changed_character_groups={k: dict(pairs=n, changed_character_correct=changed[k],
                    changed_character_trials=2*n, both_unchanged_characters_correct=unchanged[k],
                    output_changed_at_unchanged_character=broken[k]) for k, n in changes.items()},
                pair_details=detail)


def group(name):
    if '.write_' in name:
        return 'write_a' if name.endswith('write_a') else 'write_b'
    return name.split('.')[0]


def load_checked(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for f in manifest['files']:
        if file_sha256(directory / f['file']) != f['sha256']:
            raise ValueError('checkpoint checksum differs')
    return CheckpointStore(directory.parent).load_rank(directory, manifest, rank=0), manifest


def parameter_summary(payload, initial):
    values = defaultdict(lambda: defaultdict(float))
    names = payload['optimizer']['parameter_names']
    opt = payload['optimizer']['state']
    for group_names, param_group in zip(names, opt['param_groups'], strict=True):
        for name, number in zip(group_names, param_group['params'], strict=True):
            v, old = payload['weights'][name].double(), initial['weights'][name].double()
            out = values[group(name)]
            out['parameters'] += v.numel()
            out['initial_norm_sq'] += float(old.square().sum())
            out['norm_sq'] += float(v.square().sum())
            out['change_norm_sq'] += float((v-old).square().sum())
            state = opt['state'].get(number)
            if state is None:
                continue
            m, s = state['exp_avg'].double(), state['exp_avg_sq'].double()
            step = int(state['step'])
            beta1, beta2 = param_group['betas']
            update = param_group['lr'] * (m / (1-beta1**step)) / ((s / (1-beta2**step)).sqrt() + param_group['eps'])
            out['adam_direction_norm_sq'] += float(update.square().sum())
            out['adam_first_moment_norm_sq'] += float(m.square().sum())
            out['optimizer_steps_min'] = min(out.get('optimizer_steps_min', step), step)
            out['optimizer_steps_max'] = max(out.get('optimizer_steps_max', step), step)
    result = {}
    for name, v in values.items():
        x = {k.removesuffix('_sq'): n**.5 if k.endswith('_sq') else n for k, n in v.items()}
        x['change_relative_to_initial'] = x['change_norm']/x['initial_norm'] if x['initial_norm'] else None
        x['adam_direction_rms'] = x.get('adam_direction_norm', 0.)/(x['parameters']**.5)
        result[name] = x
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir', 'initial-run', 'output']:
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    run, initial_run, out = map(Path, [args.run_dir, args.initial_run, args.output])
    if out.exists():
        raise ValueError('new report required')
    initial_dir = initial_run/'checkpoints'/json.loads((initial_run/'checkpoints/latest.json').read_text())['directory']
    initial, im = load_checked(initial_dir)
    assert im['progress']['micro_step'] == 0
    evaluations, checkpoints = {}, {}
    steps = [8, 32, 64, 96, 128]
    for step in steps:
        candidates = []
        for directory in (run/'checkpoints').glob(f'micro_{step:012d}_*'):
            m = json.loads((directory/'manifest.json').read_text())
            candidates.append((m['created_unix'], directory))
        directory = max(candidates)[1]
        payload, manifest = load_checked(directory)
        checkpoints[str(step)] = dict(checkpoint=directory.name, manifest_sha256=file_sha256(directory/'manifest.json'),
            progress=manifest['progress'], groups=parameter_summary(payload, initial))
        path = run/f'evaluation-micro{step}.json'
        if path.exists():
            evaluations[str(step)] = output_summary(json.loads(path.read_text())['records'])
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    micros = [e for e in events if e['event'] == 'micro']
    assert len(micros) == 128 and all(e['progress']['micro_step'] == i+1 for i, e in enumerate(micros))
    blocks = []
    for start in range(0, 128, 32):
        block = micros[start:start+32]
        blocks.append(dict(micro_start=start+1, micro_end=start+32,
            mean={k: statistics.mean(e[k] for e in block) for k in ['answer_loss', 'counterfactual_loss', 'loss']},
            context_counts=dict(Counter(e['episode_id'] for e in block))))
    atomic_json(out, dict(scope='CPU audit only; no model inference or training', initial_checkpoint=initial_dir.name,
        initial_manifest_sha256=file_sha256(initial_dir/'manifest.json'),
        evaluations=evaluations, checkpoints=checkpoints, training_blocks=blocks,
        context_exposures=dict(Counter(e['episode_id'] for e in micros)),
        limitations=['Adam direction is the bias-corrected normalized saved moment direction, not a next-step prediction.',
                     'Character diagnostics apply only to the six-character synthetic code fixture.',
                     'Parameter movement does not prove retained source information.']))
    print(out)


if __name__ == '__main__':
    main()
