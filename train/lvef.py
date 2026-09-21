"""ECG → 单次编码 → 纵向编码 → LVEF 回归。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import LongitudinalECGEncoder, SingleECGEncoder
from utils import ECGQualityError, iter_samples, read_ecg


class LVEFRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.single = SingleECGEncoder()
        self.longitudinal = LongitudinalECGEncoder(self.single.output_dim)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, waveforms, days, mask):
        valid = self.single(waveforms[mask])
        events = valid.new_zeros(*mask.shape, self.single.output_dim)
        events[mask] = valid
        return self.head(self.longitudinal(events, days, mask)).squeeze(-1)


def prepare_samples(path, seed):
    rows, excluded = [], Counter()
    units = Counter()
    for sample in iter_samples(path):
        label = sample['echo'].get('lvef')
        if label is None:
            excluded['missing_lvef'] += 1
            continue
        units[str(label.get('unit'))] += 1
        try:
            target = float(label['result'])
        except (TypeError, ValueError):
            excluded['nonnumeric_lvef'] += 1
            continue
        if not math.isfinite(target) or not 0 <= target <= 100:
            excluded['out_of_range_lvef'] += 1
            continue
        if label.get('unit') not in (None, '', '%', 'percent'):
            raise ValueError(f"Unexpected LVEF unit: {label.get('unit')!r}")
        if not sample['ecg']:
            excluded['no_ecg'] += 1
            continue
        rows.append({**sample, 'target': target})
    subjects = sorted({row['subject_id'] for row in rows})
    if len(subjects) < 10:
        raise ValueError('At least 10 labeled patients are required')
    random.Random(seed).shuffle(subjects)
    first, second = int(len(subjects) * .7), int(len(subjects) * .85)
    groups = dict(train=subjects[:first], val=subjects[first:second], test=subjects[second:])
    membership = {subject: name for name, group in groups.items() for subject in group}
    splits = {name: [] for name in groups}
    for row in rows:
        splits[membership[row['subject_id']]].append(row)
    audit = {'excluded': dict(excluded), 'raw_lvef_units': dict(units), 'patients': groups,
             'samples': {name: len(group) for name, group in splits.items()}}
    return splits, audit


class ECGDataset(Dataset):
    def __init__(self, samples, max_events):
        self.samples = samples
        self.max_events = max_events

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        anchor = datetime.fromisoformat(sample['echo_time'])
        events = sorted(sample['ecg'], key=lambda event: event['time'], reverse=True)
        waveforms, days = [], []
        skipped_events = 0
        for event in events:
            elapsed = (anchor - datetime.fromisoformat(event['time'])).total_seconds() / 86400
            if not 0 < elapsed <= 180:
                raise ValueError(f"Invalid historical ECG time in {sample['sample_id']}")
            try:
                record = read_ecg(event['path'], missing='interpolate', max_missing_fraction=0.01)
            except ECGQualityError:
                skipped_events += 1
                continue
            if record['fs'] != 500 or record['signal'].shape != (12, 5000):
                raise ValueError(f"Unexpected ECG format: {event['path']}")
            waveforms.append(torch.from_numpy(record['signal']))
            days.append(elapsed)
            if self.max_events and len(waveforms) >= self.max_events:
                break
        if not waveforms:
            raise ECGQualityError(f"No usable ECG events for sample {sample['sample_id']}")
        waveforms.reverse()
        days.reverse()
        return (
            torch.stack(waveforms), torch.tensor(days), sample['target'],
            sample['sample_id'], skipped_events,
        )


def collate_samples(rows):
    count = max(row[0].shape[0] for row in rows)
    waveforms = torch.zeros(len(rows), count, 12, 5000)
    days = torch.zeros(len(rows), count)
    mask = torch.zeros(len(rows), count, dtype=torch.bool)
    targets = torch.tensor([row[2] for row in rows], dtype=torch.float32)
    skipped_events = sum(row[4] for row in rows)
    for index, (signals, times, _, _, _) in enumerate(rows):
        length = signals.shape[0]
        waveforms[index, :length] = signals
        days[index, :length] = times
        mask[index, :length] = True
    return waveforms, days, mask, targets, [row[3] for row in rows], skipped_events


def run_epoch(
    model,
    loader,
    device,
    center,
    scale,
    optimizer=None,
    max_batches=0,
    description='',
    show_progress=True,
    scaler=None,
    amp_enabled=False,
    amp_dtype=torch.bfloat16,
):
    model.train(optimizer is not None)
    predictions, targets, sample_ids = [], [], []
    total = min(len(loader), max_batches) if max_batches else len(loader)
    progress = tqdm(loader, total=total, desc=description, unit='batch', disable=not show_progress)
    skipped_ecg_events = 0
    for batch_index, (waveforms, days, mask, target, identifiers, skipped) in enumerate(progress):
        waveforms, days, mask, target = [
            value.to(device, non_blocking=True) for value in (waveforms, days, mask, target)
        ]
        skipped_ecg_events += skipped
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(waveforms, days, mask)
                loss = nn.functional.smooth_l1_loss(output, (target - center) / scale)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
        predictions.extend((output.detach().cpu() * scale + center).tolist())
        targets.extend(target.tolist())
        sample_ids.extend(identifiers)
        running_predictions = np.asarray(predictions)
        running_targets = np.asarray(targets)
        progress.set_postfix(
            loss=f'{loss.item():.4f}',
            mae=f'{np.abs(running_predictions - running_targets).mean():.2f}pp',
            skipped=skipped_ecg_events,
        )
        if max_batches and batch_index + 1 >= max_batches:
            break
    progress.close()
    predictions, targets = np.array(predictions), np.array(targets)
    metrics = {
        'n': len(targets),
        'mae_pp': float(np.abs(predictions - targets).mean()),
        'rmse_pp': float(np.sqrt(np.mean((predictions - targets) ** 2))),
        'mean_baseline_mae_pp': float(np.abs(center - targets).mean()),
        'mean_baseline_rmse_pp': float(np.sqrt(np.mean((center - targets) ** 2))),
        'skipped_ecg_events': skipped_ecg_events,
    }
    return metrics, [dict(sample_id=identifier, target=float(target), prediction=float(prediction))
                     for identifier, target, prediction in zip(sample_ids, targets, predictions)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', default='dataset/processed/samples.jsonl')
    parser.add_argument('--output', default='train/runs/lvef')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-events', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--amp-dtype', choices=('bf16', 'fp16'), default='bf16')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or min(args.max_events, args.workers, args.max_batches) < 0:
        parser.error('epochs/batch-size must be positive; limits/workers must be nonnegative')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr must be positive and finite')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check GPU access or explicitly choose --device cpu')
    splits, audit = prepare_samples(args.samples, args.seed)
    center = float(np.mean([row['target'] for row in splits['train']]))
    scale = max(float(np.std([row['target'] for row in splits['train']])), 1.0)
    output_dir = Path(args.output)
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        if existing - {'config.json'}:
            raise FileExistsError(
                f"Output directory contains training artifacts: {output_dir}. "
                "Choose a new --output directory."
            )
    else:
        output_dir.mkdir(parents=True)
    with (output_dir / 'config.json').open('w') as stream:
        json.dump({**vars(args), 'target_mean': center, 'target_std': scale,
                   'smoke_test': bool(args.max_batches), 'split_audit': audit}, stream, indent=2)
    print(json.dumps({'samples': audit['samples'], 'excluded': audit['excluded'],
                      'target_mean': center, 'target_std': scale, 'device': str(device)}), flush=True)
    loader_options = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'collate_fn': collate_samples,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': args.workers > 0,
    }
    if args.workers > 0:
        loader_options['prefetch_factor'] = 2
    loaders = {
        name: DataLoader(ECGDataset(rows, args.max_events), shuffle=name == 'train', **loader_options)
        for name, rows in splits.items()
    }
    model = LVEFRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = device.type == 'cuda' and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    best = float('inf')
    for epoch in range(args.epochs):
        train_metrics, _ = run_epoch(
            model, loaders['train'], device, center, scale, optimizer, args.max_batches,
            description=f'Epoch {epoch + 1}/{args.epochs} train', show_progress=not args.no_progress,
            scaler=scaler, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        val_metrics, _ = run_epoch(
            model, loaders['val'], device, center, scale, max_batches=args.max_batches,
            description=f'Epoch {epoch + 1}/{args.epochs} val', show_progress=not args.no_progress,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        report = dict(epoch=epoch + 1, train=train_metrics, val=val_metrics)
        print(json.dumps(report), flush=True)
        with (output_dir / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
        if val_metrics['mae_pp'] < best:
            best = val_metrics['mae_pp']
            torch.save({'model': model.state_dict(), 'target_mean': center, 'target_std': scale,
                        'epoch': epoch + 1, 'config': vars(args)}, output_dir / 'best.pt')
    if not args.max_batches:
        checkpoint = torch.load(output_dir / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        metrics, predictions = run_epoch(
            model, loaders['test'], device, center, scale,
            description='Test', show_progress=not args.no_progress,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        with (output_dir / 'test.json').open('w') as stream:
            json.dump({'metrics': metrics, 'predictions': predictions}, stream, indent=2)
        print(json.dumps({'test': metrics}), flush=True)
    print(f"Completed {'smoke test' if args.max_batches else 'training'}: {output_dir}", flush=True)


if __name__ == '__main__':
    main()
