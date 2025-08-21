import os
from typing import Optional


def check_password_env() -> Optional[str]:
    return os.environ.get("STREAMLIT_PASSWORD")


def validate_password(input_password: str) -> bool:
    expected = check_password_env()
    if not expected:
        return True
    return input_password == expected
