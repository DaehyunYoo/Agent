import pytest
from agent_system.src.data.loader import KMMLUDataLoader

def test_loader_initialization():
    """데이터 로더 초기화 테스트"""
    loader = KMMLUDataLoader()
    assert loader.raw_dir.exists()
    assert loader.processed_dir.exists()

def test_load_criminal_law_test(sample_data):
    """Criminal Law 테스트 데이터 로드 테스트"""
    loader = KMMLUDataLoader()
    df = loader.load_criminal_law_test()
    assert not df.empty
    assert all(col in df.columns for col in ['question', 'answer', 'A', 'B', 'C', 'D'])

def test_validate_data(sample_data):
    """데이터 검증 기능 테스트"""
    loader = KMMLUDataLoader()
    assert loader.validate_data(sample_data) == True