# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.type_aliases import ChemicalSystems

logger = logging.getLogger("mlip")


def _read_from_reader(reader: ChemicalSystemsReader) -> ChemicalSystems:
    """Top-level (picklable) worker entry point."""
    return reader.load()


def read_chemical_systems_in_parallel(
    readers: list[ChemicalSystemsReader],
    num_workers: int,
) -> dict[int, ChemicalSystems]:
    """Read from multiple readers concurrently, keyed by `id(reader)`.

    Args:
        readers: The readers to read from. Duplicates are read only once.
        num_workers: Number of concurrent workers to use.

    Returns:
        A dict mapping `id(reader)` to that reader's chemical systems, for every
        reader in `readers`.
    """
    unique_readers = list({id(r): r for r in readers}.values())

    # Readers that can't be sharded fall back to a single whole-file job
    if num_workers <= 1 or not unique_readers:
        return {id(reader): reader.load() for reader in unique_readers}

    logger.info(
        "Reading from %d reader(s) with up to %d worker(s)...",
        len(unique_readers),
        num_workers,
    )

    with tempfile.TemporaryDirectory() as download_dir:
        download_dir_path = Path(download_dir)

        def _prepare(
            indexed_reader: tuple[int, ChemicalSystemsReader],
        ) -> list[ChemicalSystemsReader]:
            idx, reader = indexed_reader
            reader_dir = download_dir_path / str(idx)
            reader_dir.mkdir(parents=True, exist_ok=True)
            return reader.prepare_parallel_readers(reader_dir, num_workers)

        indexed_readers = list(enumerate(unique_readers))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            sub_readers_list = list(pool.map(_prepare, indexed_readers))

        jobs: list[tuple[int, ChemicalSystemsReader]] = [
            (id(reader), sub_reader)
            for (_, reader), sub_readers in zip(indexed_readers, sub_readers_list)
            for sub_reader in sub_readers
        ]

        logger.debug("Split into %d parallel reading job(s).", len(jobs))

        results: dict[int, ChemicalSystems] = {id(r): [] for r in unique_readers}
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [
                (reader_id, executor.submit(_read_from_reader, job_reader))
                for reader_id, job_reader in jobs
            ]
            for reader_id, future in futures:
                try:
                    results[reader_id].extend(future.result())
                except BrokenProcessPool as e:
                    raise RuntimeError(
                        "A reader worker process died unexpectedly (likely OOM). "
                        "Try lowering `num_reader_workers`."
                    ) from e

    logger.info("Finished reading systems from %d reader(s).", len(unique_readers))
    return results
