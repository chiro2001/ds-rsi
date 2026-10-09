import pytest

from dsrsi.data.revisions import AB_DATA_REVISION, data_identity, make_generator
from dsrsi.data.robust import AB_DATA_REVISION as V2, data_identity as old_identity, RobustEpisodeGenerator


def test_new_revision_randomizes_target_numeric_rank_and_keeps_pairs_consistent():
    generator = make_generator(seed=7, distractors=7)
    ranks = set()
    for world in range(120):
        a = generator.episode(world, family="single_fact", language="en", variant=0)
        b = generator.episode(world, family="single_fact", language="en", variant=1)
        ranks.add(a["annotations"]["target_numeric_rank"])
        assert a["annotations"]["target_numeric_rank"] == b["annotations"]["target_numeric_rank"]
        assert a["annotations"]["target_entity"] in a["probes"][0]["query"]
        support, = a["probes"][0]["support_groups"][0]
        assert a["annotations"]["target_entity"] in next(e["content"] for e in a["source_events"] if e["event_id"] == support)
        assert [e for e in a["source_events"] if e["event_id"] != support] == [e for e in b["source_events"] if e["event_id"] != support]
    assert len(ranks) == 5


def test_v2_dispatch_preserves_implementation_identity_and_samples_exactly():
    assert data_identity(V2) == old_identity(V2)
    assert make_generator(revision=V2).episode(3) == RobustEpisodeGenerator().episode(3)
    assert "robust_order.py" in data_identity(AB_DATA_REVISION)["files"]


def test_versioned_a_to_a_weights_initialization_is_distinct_from_resume(tmp_path):
    import torch
    from dsrsi.training.runner import Trainer, TrainConfig
    from dsrsi.data.compiler import EpisodeCompiler
    from test_qwen_bridge import tiny
    from test_episode_objective import SmallVocabularyTokenizer
    compiler = EpisodeCompiler(SmallVocabularyTokenizer(), max_query_tokens=1000)
    torch.manual_seed(81)
    source = Trainer(tiny(), compiler, tmp_path / "v2", TrainConfig(distractors=2, data_revision=V2), base_identity="base")
    source.run(max_micro_steps=1, install_signals=False)
    torch.manual_seed(81)
    target = Trainer(tiny(), compiler, tmp_path / "v3", TrainConfig(distractors=2),
                     base_identity="base", initialize_from=source.run_dir)
    assert target.progress.micro_step == 0 and not target.optimizer.state
    assert target.identity["initialization"]["transition"] == "A-to-A-reset-training-state"
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, target.model.state_dict()[name]), name
    with pytest.raises(ValueError, match="change data revision"):
        Trainer(tiny(), compiler, tmp_path / "same", TrainConfig(distractors=2, data_revision=V2),
                base_identity="base", initialize_from=source.run_dir)
