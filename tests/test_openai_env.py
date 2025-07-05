import os
import pytest
from packaging import version

openai = pytest.importorskip("openai")


def test_openai_env():
    if os.getenv("OPENAI_API_KEY") is None:
        print("Warning: OPENAI_API_KEY is missing")

    assert version.parse(openai.__version__) >= version.parse("1.23.0")

    try:
        openai.models.list()
    except (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.APIConnectionError,
    ):
        pytest.skip("OpenAI API unauthorized or unreachable")

