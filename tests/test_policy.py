import pytest
from inspect_robots.policy import Policy, PolicyConfig

from inspect_robots_dropbear import dropbear_policy


def test_factory_constructs_only_dreamzero_yam_without_connecting(monkeypatch) -> None:
    """Catch a discovery path that opens a Dropbear connection."""
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: pytest.fail("constructor touched the network"),
    )

    policy = dropbear_policy(model="dreamzero-yam")

    assert isinstance(policy, Policy)
    assert policy.info.name == "dropbear"
    assert policy.info.action_space.shape == (14,)
    assert policy.info.control_hz == 15.0
    assert policy.config == PolicyConfig(action_horizon=24, replan_interval=1)


def test_factory_rejects_every_other_model_offline() -> None:
    """Catch a factory that accepts a Dropbear model other than DreamZero-YAM."""
    with pytest.raises(ValueError, match="only dreamzero-yam is supported"):
        dropbear_policy(model="dreamzero-droid")
