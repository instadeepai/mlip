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

import numpy as np
import pytest

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.helpers.filtering_utils import (
    filter_charged_systems,
    filter_systems_by_allowed_elements,
    filter_systems_by_allowed_total_charges,
    filter_systems_by_elements,
    filter_systems_by_excluded_elements,
    filter_systems_by_excluded_total_charges,
    filter_systems_by_predicate,
    filter_systems_with_unseen_atoms,
    filter_systems_with_unseen_charge,
    process_chemical_systems,
    set_system_none_charges_to_zero,
)


def _make_system(atomic_numbers, n_atoms=None):
    """Helper to create a minimal ChemicalSystem."""
    zs = np.asarray(atomic_numbers)
    n = len(zs) if n_atoms is None else n_atoms
    return ChemicalSystem(
        atomic_numbers=zs,
        positions=np.zeros((n, 3)),
    )


class TestFilterSystemsWithUnseenAtoms:
    def test_all_atoms_seen(self):
        systems = [_make_system([1, 6]), _make_system([1, 8])]
        result = filter_systems_with_unseen_atoms(systems, [1, 6, 8])
        assert len(result) == 2

    def test_one_has_unseen_atom(self):
        systems = [_make_system([1, 6]), _make_system([1, 99])]
        result = filter_systems_with_unseen_atoms(systems, [1, 6, 8])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0].atomic_numbers, [1, 6])

    def test_empty_input(self):
        result = filter_systems_with_unseen_atoms([], [1, 6])
        assert result == []

    def test_all_unseen(self):
        systems = [_make_system([99]), _make_system([100])]
        result = filter_systems_with_unseen_atoms(systems, [1, 6])
        assert result == []

    def test_accepts_numpy_array(self):
        systems = [_make_system([1, 6])]
        result = filter_systems_with_unseen_atoms(systems, np.array([1, 6]))
        assert len(result) == 1


class TestFilterSystemsByPredicate:
    def test_always_true(self):
        systems = [_make_system([1]), _make_system([6])]
        result = filter_systems_by_predicate(systems, lambda _: True)
        assert len(result) == 2

    def test_always_false(self):
        systems = [_make_system([1]), _make_system([6])]
        result = filter_systems_by_predicate(systems, lambda _: False)
        assert result == []

    def test_predicate_on_num_atoms(self):
        small = _make_system([1])
        big = _make_system([1, 6, 8])
        result = filter_systems_by_predicate(
            [small, big], lambda s: len(s.atomic_numbers) > 1
        )
        assert len(result) == 1
        assert len(result[0].atomic_numbers) == 3


class TestFilterSystemsByElements:
    def test_returns_callable(self):
        fn = filter_systems_by_elements([1, 6])
        assert callable(fn)

    def test_filters_correctly(self):
        fn = filter_systems_by_elements([1, 6])
        systems = [_make_system([1, 6]), _make_system([1, 8])]
        result = fn(systems)
        assert len(result) == 1


class TestFilterSystemsByExcludedElements:
    def test_excludes_systems_with_excluded_atoms(self):
        fn = filter_systems_by_excluded_elements([99])
        systems = [_make_system([1, 6]), _make_system([1, 99])]
        result = fn(systems)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0].atomic_numbers, [1, 6])

    def test_keeps_all_when_no_excluded(self):
        fn = filter_systems_by_excluded_elements([99])
        systems = [_make_system([1, 6]), _make_system([1, 8])]
        result = fn(systems)
        assert len(result) == 2


class TestFilterChargedSystems:
    def test_raises_attribute_error(self):
        """ChemicalSystem has no total_charge field — documents known limitation."""
        system = _make_system([1, 6])
        with pytest.raises(AttributeError):
            filter_charged_systems([system])

    def test_filter_systems_with_unseen_charges(self):
        sys1 = _make_system([1])
        sys2 = _make_system([2])
        sys3 = _make_system([3])
        sys1.charge = 0
        sys2.charge = 1
        sys3.charge = 2
        allowed_charges = [0, 1]
        result = filter_systems_with_unseen_charge([sys1, sys2, sys3], allowed_charges)
        # assert the correct systems are left
        assert len(result) == 2
        assert sys1 in result
        assert sys2 in result
        assert sys3 not in result


class TestProcessChemicalSystems:
    def test_none_filters_passthrough(self):
        systems = [_make_system([1])]
        result = process_chemical_systems(systems, None)
        assert result is systems

    def test_empty_filters_passthrough(self):
        systems = [_make_system([1])]
        result = process_chemical_systems(systems, [])
        assert result is systems

    def test_single_filter(self):
        systems = [_make_system([1]), _make_system([99])]
        fn = filter_systems_by_elements([1])
        result = process_chemical_systems(systems, [fn])
        assert len(result) == 1

    def test_two_filters_composed(self):
        systems = [
            _make_system([1]),
            _make_system([1, 6]),
            _make_system([1, 99]),
        ]
        fn1 = filter_systems_by_elements([1, 6])
        fn2 = filter_systems_by_excluded_elements([6])
        result = process_chemical_systems(systems, [fn1, fn2])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0].atomic_numbers, [1])


class TestBuilderConfigBasedPreprocessingFns:
    def test_filter_systems_by_allowed_elements(self):
        fn = filter_systems_by_allowed_elements([6, 8])
        sys1 = _make_system([6])
        sys2 = _make_system([8, 6])
        sys3 = _make_system([12, 24])
        result = fn([sys1, sys2, sys3])
        # Should keep sys1 and sys2, drop sys3
        assert len(result) == 2
        result_zs = set()
        for sys in result:
            result_zs.update(sys.atomic_numbers)
        for z in result_zs:
            assert z in [6, 8]

    def test_filter_systems_by_excluded_elements(self):
        fn = filter_systems_by_excluded_elements([6, 8])
        sys1 = _make_system([6, 1])
        sys2 = _make_system([1, 2])
        sys3 = _make_system([8, 2])
        sys4 = _make_system([12, 24])
        result = fn([sys1, sys2, sys3, sys4])
        # Should keep sys2 and sys4, drop sys1 and sys3
        assert len(result) == 2
        result_zs = set()
        for sys in result:
            result_zs.update(sys.atomic_numbers)
        for z in result_zs:
            assert z not in [6, 8]

    def test_filter_systems_by_allowed_total_charges(self):
        sys1 = _make_system([1])
        sys2 = _make_system([2])
        sys3 = _make_system([3])
        sys1.charge = 0
        sys2.charge = 1
        sys3.charge = 2
        fn = filter_systems_by_allowed_total_charges([1, 2])
        result = fn([sys1, sys2, sys3])
        # Should keep sys2 and sys3
        assert len(result) == 2
        assert sys2 in result
        assert sys3 in result
        assert sys1 not in result

    def test_filter_systems_by_excluded_total_charges(self):
        sys1 = _make_system([1])
        sys2 = _make_system([2])
        sys3 = _make_system([3])
        sys1.charge = 0
        sys2.charge = 1
        sys3.charge = 2
        fn = filter_systems_by_excluded_total_charges([0, 2])
        result = fn([sys1, sys2, sys3])
        # Should keep sys2, drop sys1 and sys3
        assert len(result) == 1
        assert sys2 in result
        assert sys1 not in result
        assert sys3 not in result

    def test_set_system_none_charges_to_zero(self):
        sys1 = _make_system([1])
        sys2 = _make_system([2])
        fn = set_system_none_charges_to_zero
        result = fn([sys1, sys2])
        assert len(result) == 2
        assert result[0].charge == 0
        assert result[1].charge == 0
