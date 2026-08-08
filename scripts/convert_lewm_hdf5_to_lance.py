"""Convert a LEWM HDF5 dataset to its indexed, JPEG-backed Lance format."""

import argparse
import sys
from pathlib import Path

import numpy as np


def iter_episodes(source, begin, end):
    for episode_idx in range(begin, end):
        episode = source.load_episode(episode_idx)
        # Stable WM records the final action of each episode as NaN because no
        # action is taken after the terminal frame. OGBench excludes terminal
        # transitions from sampling, but Lance vector columns require every
        # stored value to be finite. Store a harmless zero placeholder for
        # those otherwise-unused terminal actions.
        actions = np.asarray(episode['action'])
        if not np.isfinite(actions).all():
            episode['action'] = np.nan_to_num(
                actions, nan=0.0, posinf=0.0, neginf=0.0
            )
        yield episode_idx, episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stable_wm_root')
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.stable_wm_root).resolve()))
    from stable_worldmodel.data import load_dataset
    from stable_worldmodel.data.format import get_format
    from stable_worldmodel.data.utils import _episode_to_step_lists

    # OGBench training only consumes pixels and actions. Projecting these
    # columns avoids decoding/copying unrelated privileged fields and also
    # excludes auxiliary vectors that may legitimately contain NaNs.
    source = load_dataset(
        args.source,
        format='hdf5',
        keys_to_load=['pixels', 'action'],
    )
    writer_cls = get_format('lance')
    total = len(source.lengths)

    # Lance commits each write_episodes call atomically. Committing 100
    # episodes at a time makes this multi-hour conversion resumable without
    # ever exposing a half-written episode.
    chunk_episodes = 100
    destination = Path(args.destination)
    completed = 0
    if destination.exists():
        import lancedb

        table = lancedb.connect(str(destination.parent)).open_table(
            destination.stem
        )
        reader = table.to_lance().scanner(
            columns=['episode_idx']
        ).to_reader()
        for batch in reader:
            values = batch.column(0).to_numpy(zero_copy_only=False)
            if len(values):
                completed = max(completed, int(values.max()) + 1)
        print(f'Resuming at episode {completed}/{total}', flush=True)

    with writer_cls.open_writer(
        destination,
        mode='append',
        jpeg_quality=args.jpeg_quality,
    ) as writer:
        for begin in range(completed, total, chunk_episodes):
            end = min(begin + chunk_episodes, total)
            episodes = (
                _episode_to_step_lists(
                    episode, int(source.lengths[episode_idx])
                )
                for episode_idx, episode in iter_episodes(source, begin, end)
            )
            writer.write_episodes(episodes)
            print(f'{end}/{total} episodes converted', flush=True)


if __name__ == '__main__':
    main()
