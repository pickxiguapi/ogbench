"""Train a LeWM-style decoder from frozen encoder latents and source RGB frames.

The LeWM and subgoal-generator parameters are never loaded or updated.  The
input HDF5 cache is already checkpoint-bound and contains encoder targets ``z``;
RGB supervision is read from the exact JPEG-backed Lance table used by LeWM.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from lewm_visual_decoder import CLSDecoder, ConvDecoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('pusht', 'cube', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--latent-hdf5', required=True)
    parser.add_argument('--lance-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-rows', type=int, default=200000)
    parser.add_argument('--val-rows', type=int, default=20000)
    parser.add_argument('--decode-workers', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument(
        '--foreground-weight',
        type=float,
        default=0.0,
        help='Extra weight on pixels that differ from the batch mean; 0 is official plain MSE.',
    )
    parser.add_argument('--foreground-threshold', type=float, default=0.08)
    parser.add_argument('--foreground-dilation', type=int, default=9)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-batches', type=int, default=0)
    parser.add_argument('--decoder-type', choices=('conv', 'cls'), default='conv')
    return parser.parse_args()


class BatchSource:
    def __init__(self, latent_hdf5, lance_path, workers):
        import lancedb
        from lancedb.permutation import Permutation

        self.h5 = h5py.File(latent_hdf5, 'r')
        table_path = Path(lance_path)
        table = lancedb.connect(str(table_path.parent)).open_table(table_path.stem)
        if len(self.h5['z']) != table.count_rows():
            raise ValueError('Latent HDF5 and Lance row counts differ.')
        self.rows = Permutation.identity(table).select_columns(['pixels']).with_format('arrow')
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))

    @staticmethod
    def decode(blob):
        with Image.open(io.BytesIO(blob)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()

    def fetch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        batch = self.rows.__getitems__(indices.tolist())
        blobs = batch.column(batch.schema.get_field_index('pixels')).to_pylist()
        pixels = np.stack(list(self.executor.map(self.decode, blobs)))
        order = np.argsort(indices)
        inverse = np.argsort(order)
        latents = self.h5['z'][indices[order]][inverse].astype(np.float32, copy=False)
        return latents, pixels

    def close(self):
        self.executor.shutdown(wait=True)
        self.h5.close()


def episode_split_indices(h5, seed, train_rows, val_rows):
    episode_ids = np.asarray(h5['episode_idx'])
    unique = np.unique(episode_ids)
    split = max(1, min(len(unique) - 1, int(0.9 * len(unique))))
    train_pool = np.flatnonzero(np.isin(episode_ids, unique[:split]))
    val_pool = np.flatnonzero(np.isin(episode_ids, unique[split:]))
    rng = np.random.default_rng(seed)
    train = rng.choice(train_pool, size=min(train_rows, len(train_pool)), replace=False)
    val = rng.choice(val_pool, size=min(val_rows, len(val_pool)), replace=False)
    return train.astype(np.int64), val.astype(np.int64)


def tensors(latents, pixels, device):
    z = torch.from_numpy(latents).to(device, non_blocking=True)
    target = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device, non_blocking=True).float()
    target = target.div_(127.5).sub_(1.0)
    return z, target


def reconstruction_losses(
    reconstruction,
    target,
    *,
    foreground_weight,
    foreground_threshold,
    foreground_dilation,
):
    squared_error = torch.square(reconstruction - target)
    full_mse = squared_error.mean()
    if foreground_weight <= 0:
        return full_mse, full_mse, full_mse.new_zeros(()), full_mse.new_zeros(())
    if foreground_dilation <= 0 or foreground_dilation % 2 != 1:
        raise ValueError('foreground_dilation must be a positive odd integer.')
    reference = target.detach().mean(dim=0, keepdim=True)
    mask = (target.detach() - reference).abs().amax(dim=1, keepdim=True)
    mask = (mask > foreground_threshold).to(squared_error.dtype)
    if foreground_dilation > 1:
        mask = F.max_pool2d(
            mask,
            kernel_size=foreground_dilation,
            stride=1,
            padding=foreground_dilation // 2,
        )
    mask_fraction = mask.mean()
    foreground_mse = (squared_error * mask).sum() / (
        mask.sum().clamp_min(1.0) * squared_error.shape[1]
    )
    objective = full_mse + foreground_weight * foreground_mse
    return objective, full_mse, foreground_mse, mask_fraction


@torch.no_grad()
def evaluate(
    model,
    source,
    indices,
    batch_size,
    device,
    smoke_batches,
    preview_path,
    *,
    foreground_weight,
    foreground_threshold,
    foreground_dilation,
):
    model.eval()
    total = 0.0
    count = 0
    objective_total = 0.0
    foreground_total = 0.0
    mask_fraction_total = 0.0
    preview = None
    for batch_number, start in enumerate(range(0, len(indices), batch_size)):
        if smoke_batches and batch_number >= smoke_batches:
            break
        z, target = tensors(*source.fetch(indices[start : start + batch_size]), device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            reconstruction = model(z)
            objective, full_mse, foreground_mse, mask_fraction = reconstruction_losses(
                reconstruction,
                target,
                foreground_weight=foreground_weight,
                foreground_threshold=foreground_threshold,
                foreground_dilation=foreground_dilation,
            )
        total += float(full_mse.item()) * len(z)
        objective_total += float(objective.item()) * len(z)
        foreground_total += float(foreground_mse.item()) * len(z)
        mask_fraction_total += float(mask_fraction.item()) * len(z)
        count += len(z)
        if preview is None:
            n = min(8, len(z))
            preview = torch.cat([target[:n], reconstruction[:n]], dim=0).float().cpu()
    mse = total / max(count, 1)
    if preview is not None:
        save_image(make_grid((preview + 1) * 0.5, nrow=min(8, preview.shape[0] // 2)), preview_path)
    return {
        'mse': mse,
        'psnr': -10.0 * math.log10(max(mse / 4.0, 1e-12)),
        'objective': objective_total / max(count, 1),
        'foreground_mse': foreground_total / max(count, 1),
        'mask_fraction': mask_fraction_total / max(count, 1),
    }


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source = BatchSource(args.latent_hdf5, args.lance_path, args.decode_workers)
    train_indices, val_indices = episode_split_indices(
        source.h5, args.seed, args.train_rows, args.val_rows
    )
    embed_dim = int(source.h5['z'].shape[1])
    checkpoint_sha256 = str(source.h5.attrs['checkpoint_sha256'])
    if args.decoder_type == 'conv':
        config = {
            'cls_dim': embed_dim,
            'image_size': 224,
            'out_channels': 3,
            'base_channels': 512,
            'min_channels': 32,
            'start_size': 7,
            'num_res_blocks': 2,
        }
        model = ConvDecoder(**config).to(device)
        checkpoint_type = 'lewm_official_conv_visual_decoder_v2'
    else:
        config = {
            'cls_dim': embed_dim,
            'hidden_dim': 256,
            'depth': 4,
            'heads': 8,
            'dim_head': 64,
            'mlp_dim': 512,
            'dropout': 0.1,
            'image_size': 224,
            'patch_size': 16,
        }
        model = CLSDecoder(**config).to(device)
        checkpoint_type = 'lewm_cls_visual_decoder_v1'
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_indices) / args.batch_size)
    if args.smoke_batches:
        steps_per_epoch = min(steps_per_epoch, args.smoke_batches)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup = max(1, total_steps // 100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, 1.0)
        * 0.5
        * (1.0 + math.cos(math.pi * max(0, step - warmup) / max(1, total_steps - warmup))),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'args': vars(args),
        'model_config': config,
        'lewm_checkpoint_sha256': checkpoint_sha256,
        'train_rows_selected': len(train_indices),
        'val_rows_selected': len(val_indices),
        'protocol': 'frozen encoder latent to same-frame JPEG RGB; decoder-only MSE',
        'decoder_type': args.decoder_type,
        'loss': (
            'full_mse'
            if args.foreground_weight <= 0
            else 'full_mse + foreground_weight * batch_variation_foreground_mse'
        ),
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    best = float('inf')
    rng = np.random.default_rng(args.seed)
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            shuffled = rng.permutation(train_indices)
            model.train()
            total = 0.0
            objective_total = 0.0
            count = 0
            for batch_number, start in enumerate(range(0, len(shuffled), args.batch_size)):
                if args.smoke_batches and batch_number >= args.smoke_batches:
                    break
                z, target = tensors(*source.fetch(shuffled[start : start + args.batch_size]), device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    reconstruction = model(z)
                    objective, full_mse, _, _ = reconstruction_losses(
                        reconstruction,
                        target,
                        foreground_weight=args.foreground_weight,
                        foreground_threshold=args.foreground_threshold,
                        foreground_dilation=args.foreground_dilation,
                    )
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += float(full_mse.item()) * len(z)
                objective_total += float(objective.item()) * len(z)
                count += len(z)
                if batch_number % 100 == 0:
                    print(
                        f'epoch={epoch} batch={batch_number} '
                        f'train_mse={total / count:.6f} '
                        f'train_objective={objective_total / count:.6f}',
                        flush=True,
                    )
            validation = evaluate(
                model, source, val_indices, args.batch_size, device, args.smoke_batches,
                output_dir / f'preview_epoch_{epoch:03d}.png',
                foreground_weight=args.foreground_weight,
                foreground_threshold=args.foreground_threshold,
                foreground_dilation=args.foreground_dilation,
            )
            payload = {
                'type': checkpoint_type,
                'model': model.state_dict(),
                'model_config': config,
                'epoch': epoch,
                'best_val_objective': min(best, validation['objective']),
                'validation': validation,
                'manifest': manifest,
            }
            atomic_save(output_dir / 'latest.pt', payload)
            if validation['objective'] < best:
                best = validation['objective']
                atomic_save(output_dir / 'best.pt', payload)
            print(
                f'epoch={epoch} train_mse={total / max(count, 1):.6f} '
                f'train_objective={objective_total / max(count, 1):.6f} '
                f'val_mse={validation["mse"]:.6f} '
                f'val_foreground_mse={validation["foreground_mse"]:.6f} '
                f'val_mask_fraction={validation["mask_fraction"]:.4f} '
                f'val_psnr={validation["psnr"]:.3f} '
                f'seconds={time.time() - started:.1f}',
                flush=True,
            )
    finally:
        source.close()


if __name__ == '__main__':
    main()
