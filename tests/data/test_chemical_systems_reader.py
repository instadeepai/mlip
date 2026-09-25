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

import shutil

from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.type_aliases import ChemicalSystems


class _DummyReader(ChemicalSystemsReader):
    def load(self) -> ChemicalSystems:
        return [self._read_file(fp) for fp in self.filepaths]

    def _read_file(self, filepath) -> ChemicalSystems:
        return []


def test_resolve_local_filepaths_avoids_basename_collision(tmp_path):
    source_dir_a = tmp_path / "spice"
    source_dir_b = tmp_path / "mptrj"
    source_dir_a.mkdir()
    source_dir_b.mkdir()
    file_a = source_dir_a / "train.xyz"
    file_b = source_dir_b / "train.xyz"
    file_a.write_text("from spice")
    file_b.write_text("from mptrj")

    reader = _DummyReader(
        filepaths=[file_a, file_b],
        data_download_fun=shutil.copyfile,
    )

    local_paths = reader._resolve_local_filepaths(tmp_path / "downloads")

    assert len(local_paths) == 2
    assert local_paths[0] != local_paths[1]
    assert local_paths[0].read_text() == "from spice"
    assert local_paths[1].read_text() == "from mptrj"
