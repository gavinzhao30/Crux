import pytest
from pydantic import ValidationError

from core.config import RunConfig


def test_selection_and_memory_can_be_varied_independently():
    config = RunConfig(memory="local_role_only", proposers=1)
    assert config.selection == "eig"
    assert config.memory == "local_role_only"
    assert config.proposers == 1


def test_unknown_selection_is_rejected():
    with pytest.raises(ValidationError):
        RunConfig(selection="majority_vote")
