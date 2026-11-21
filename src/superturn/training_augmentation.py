"""Apply seeded audio augmentation to a training batch in input order."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from .augment import augment

_worker_assets = None


def _set_worker_assets(assets):
    global _worker_assets
    _worker_assets = assets


def _render(job, assets):
    pcm, seed, speaker, pause_samples = job
    noise, impulse, voices = assets
    other_voices = [audio for key, audio in voices.items() if key != speaker]
    return augment(pcm, seed, noise_paths=noise, ir_paths=impulse,
                   voices=other_voices, pause_samples=pause_samples)


def _render_in_worker(job):
    return _render(job, _worker_assets)


class TrainingAugmentation:
    def __init__(self, workers, assets):
        if workers < 1:
            raise ValueError("augmentation_workers must be at least one")
        self.assets = assets
        self.pool = None
        if workers > 1:
            # Spawn starts each audio worker without inheriting CUDA state.
            self.pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_set_worker_assets, initargs=(assets,),
            )

    def render(self, jobs):
        if self.pool is None:
            return [_render(job, self.assets) for job in jobs]
        return list(self.pool.map(_render_in_worker, jobs, chunksize=4))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
