import numpy as np
import torch
from gym import spaces

from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_episode_ids,
)
from habitat_baselines.rl.ver.ver_rollout_storage import (
    VERRolloutStorage,
    _permute_tensor_first_dim_in_place,
)


def test_in_place_permutation_matches_index_select():
    original = torch.arange(7 * 3 * 2).reshape(7, 3, 2)
    for ordering in (
        torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        torch.tensor([1, 2, 0, 4, 3, 6, 5]),
        torch.tensor([6, 5, 4, 3, 2, 1, 0]),
    ):
        expected = original.index_select(0, ordering)
        actual = original.clone()
        _permute_tensor_first_dim_in_place(actual, ordering)
        assert torch.equal(actual, expected)


def test_ver_generator_preserves_one_history_per_packed_sequence():
    storage = VERRolloutStorage(
        variable_experience=True,
        numsteps=2,
        num_envs=2,
        observation_space=spaces.Dict(
            {
                "marker": spaces.Box(
                    low=-1.0e6,
                    high=1.0e6,
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        ),
        action_space=spaces.Discrete(6),
        recurrent_hidden_state_size=4,
        num_recurrent_layers=1,
        action_shape=(1,),
        discrete_actions=True,
    )

    # Environment zero contributes two different episode fragments.  A
    # conventional RNN receives one initial state per unique environment, but
    # WSLA needs the state stored at the beginning of all three packed
    # sequences, including both fragments from environment zero.
    episode_ids = np.asarray([10, 10, 11, 11, 20, 20])
    environment_ids = np.asarray([0, 0, 0, 0, 1, 1])
    step_ids = np.asarray([2, 3, 5, 6, 0, 1])
    full_info = build_pack_info_from_episode_ids(
        episode_ids, environment_ids, step_ids
    )
    storage.episode_ids_cpu = episode_ids
    storage.environment_ids_cpu = environment_ids
    storage.step_ids_cpu = step_ids
    storage.sequence_lengths = full_info["sequence_lengths"]
    storage.num_seqs_at_step = full_info["num_seqs_at_step"]
    storage.select_inds = full_info["select_inds"]
    storage.last_sequence_in_batch_mask = full_info[
        "last_sequence_in_batch_mask"
    ]

    markers = torch.arange(6, dtype=torch.float32)
    storage.buffers["observations"]["marker"][:, 0] = markers
    storage.buffers["recurrent_hidden_states"][:] = markers[:, None, None]

    batch = next(storage.recurrent_generator(None, num_mini_batch=1))
    sequence_info = batch["rnn_build_seq_info"]
    sequence_starts = sequence_info["sequence_starts"]
    sequence_histories = sequence_info["sequence_start_hidden_states"]
    expected_markers = batch["observations"]["marker"].index_select(
        0, sequence_starts
    )[:, 0]

    assert batch["observations"]["marker"].shape[0] == 4  # learner samples
    assert batch["recurrent_hidden_states"].shape[0] == 2  # unique envs
    assert sequence_histories.shape == (3, 1, 4)  # packed sequences
    torch.testing.assert_close(sequence_histories[:, 0, 0], expected_markers)
