from types import SimpleNamespace
from unittest.mock import Mock

from ovon.trainers.wsla_trainer import WSLAVERTrainer


def _trainer(*, step: int, total: int):
    return SimpleNamespace(
        num_steps_done=step,
        config=SimpleNamespace(
            habitat_baselines=SimpleNamespace(total_num_steps=total)
        ),
        report_worker=SimpleNamespace(time_taken=12.5),
        save_checkpoint=Mock(),
    )


def test_completed_stage_writes_stable_final_checkpoint():
    trainer = _trainer(step=101, total=100)

    saved = WSLAVERTrainer._save_completed_stage_checkpoint(trainer)

    assert saved is True
    trainer.save_checkpoint.assert_called_once_with(
        "ckpt.final.pth", {"step": 101, "wall_time": 12.5}
    )


def test_incomplete_stage_does_not_write_final_checkpoint():
    trainer = _trainer(step=99, total=100)

    saved = WSLAVERTrainer._save_completed_stage_checkpoint(trainer)

    assert saved is False
    trainer.save_checkpoint.assert_not_called()
