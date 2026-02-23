import os

import pytest

from tfmemprof.tf_env import configure_tensorflow_logging


def test_configure_tensorflow_logging_sets_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    configure_tensorflow_logging()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "1"


def test_configure_tensorflow_logging_respects_existing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")

    configure_tensorflow_logging()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"
