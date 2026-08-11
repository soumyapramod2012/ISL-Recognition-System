from pathlib import Path

from src.preprocessing.config import (
    PROJECT_ROOT,
    MODELS_DIR,
)


def test_project_root_exists():

    assert PROJECT_ROOT is not None

    assert Path(
        PROJECT_ROOT
    ).exists()


def test_models_directory_exists():

    assert MODELS_DIR is not None

    assert Path(
        MODELS_DIR
    ).exists()