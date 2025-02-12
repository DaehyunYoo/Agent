# pytest 공통 설정
# agent_system/tests/conftest.py
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터를 생성합니다."""
    return pd.DataFrame({
        'question': ['What is criminal law?'],
        'answer': [2],
        'A': ['Civil law'],
        'B': ['Public law'],
        'C': ['Private law'],
        'D': ['International law'],
        'Category': ['Criminal Law'],
        'Human Accuracy': [0.85]
    })

@pytest.fixture
def test_env_vars(monkeypatch):
    """테스트용 환경 변수를 설정합니다."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-api-key')