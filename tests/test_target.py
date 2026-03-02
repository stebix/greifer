"""Unit tests for TargetManager."""

import numpy as np
from numpy.testing import assert_allclose

import pytest

from greifer.target import TargetManager


class TestConstruction:

    def test_single_target(self):
        tm = TargetManager(["A"])
        assert tm.names == ["A"]
        assert tm.active_name == "A"

    def test_multiple_targets(self):
        tm = TargetManager(["A", "B", "C"])
        assert tm.names == ["A", "B", "C"]

    def test_first_name_is_initial_active(self):
        tm = TargetManager(["X", "Y"])
        assert tm.active_name == "X"

    def test_empty_names_raises(self):
        with pytest.raises(ValueError, match="At least one target"):
            TargetManager([])

    def test_each_target_starts_at_identity(self):
        tm = TargetManager(["A", "B"])
        for name in tm.names:
            tm.switch_to(name)
            assert_allclose(tm.active_accumulator.matrix, np.eye(4))


class TestSwitching:

    def test_switch_to_changes_active(self):
        tm = TargetManager(["A", "B"])
        tm.switch_to("B")
        assert tm.active_name == "B"

    def test_switch_to_returns_current_matrix(self):
        tm = TargetManager(["A", "B"])
        tm.update(1.0, 0, 0, 0, 0, 0)  # move A
        matrix = tm.switch_to("B")
        assert_allclose(matrix, np.eye(4))  # B is untouched

    def test_switch_to_unknown_raises(self):
        tm = TargetManager(["A"])
        with pytest.raises(KeyError, match="Unknown target"):
            tm.switch_to("Z")

    def test_switch_preserves_previous_target_state(self):
        tm = TargetManager(["A", "B"])
        tm.update(1.0, 0, 0, 0, 0, 0)  # move A
        matrix_a = tm.active_accumulator.matrix.copy()

        tm.switch_to("B")
        tm.update(0, 2.0, 0, 0, 0, 0)  # move B

        matrix_back = tm.switch_to("A")
        assert_allclose(matrix_back, matrix_a)


class TestUpdate:

    def test_update_delegates_to_active(self):
        tm = TargetManager(["A", "B"])
        tm.update(1.0, 0, 0, 0, 0, 0)
        assert tm.active_accumulator.matrix[0, 3] != 0.0

    def test_update_does_not_affect_inactive(self):
        tm = TargetManager(["A", "B"])
        tm.update(1.0, 0, 0, 0, 0, 0)  # moves A only

        tm.switch_to("B")
        assert_allclose(tm.active_accumulator.matrix, np.eye(4))

    def test_update_returns_cumulative_matrix(self):
        tm = TargetManager(["A"])
        matrix = tm.update(1.0, 0, 0, 0, 0, 0)
        assert_allclose(matrix, tm.active_accumulator.matrix)


class TestAddTarget:

    def test_add_new_target(self):
        tm = TargetManager(["A"])
        tm.add_target("B")
        assert "B" in tm.names

    def test_add_target_is_idempotent(self):
        tm = TargetManager(["A"])
        tm.add_target("B")
        tm.switch_to("B")
        tm.update(1.0, 0, 0, 0, 0, 0)  # move B
        matrix_before = tm.active_accumulator.matrix.copy()

        tm.add_target("B")  # should not reset B
        assert_allclose(tm.active_accumulator.matrix, matrix_before)

    def test_added_target_starts_at_identity(self):
        tm = TargetManager(["A"])
        tm.add_target("B")
        tm.switch_to("B")
        assert_allclose(tm.active_accumulator.matrix, np.eye(4))

    def test_names_preserves_insertion_order(self):
        tm = TargetManager(["A"])
        tm.add_target("C")
        tm.add_target("B")
        assert tm.names == ["A", "C", "B"]
