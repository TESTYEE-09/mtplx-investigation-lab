import numpy as np

from scripts.eval_mtp_corrector import (
    _build_persistent_replay_lookup,
    _persistent_replay_indices,
)


def test_persistent_replay_can_stop_before_current_depth():
    prompt_ids = np.array(["p0", "p0", "p0"])
    window_indices = np.array([7, 7, 7], dtype=np.int64)
    depths = np.array([1, 2, 3], dtype=np.int64)
    lookup = _build_persistent_replay_lookup(prompt_ids, window_indices, depths)

    assert _persistent_replay_indices(
        2,
        prompt_ids=prompt_ids,
        window_indices=window_indices,
        depths=depths,
        row_lookup=lookup,
    ) == [0, 1, 2]
    assert _persistent_replay_indices(
        2,
        prompt_ids=prompt_ids,
        window_indices=window_indices,
        depths=depths,
        row_lookup=lookup,
        include_current=False,
    ) == [0, 1]
    assert _persistent_replay_indices(
        0,
        prompt_ids=prompt_ids,
        window_indices=window_indices,
        depths=depths,
        row_lookup=lookup,
        include_current=False,
    ) == []
