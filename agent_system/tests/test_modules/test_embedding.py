import pytest
import numpy as np
from agent_system.src.agent.embedding import EmbeddingAgent

@pytest.mark.asyncio
async def test_embedding_initialization():
    """임베딩 에이전트 초기화 테스트"""
    agent = EmbeddingAgent()
    assert agent.model == "text-embedding-small"
    assert agent.api_key is not None

@pytest.mark.asyncio
async def test_create_embedding(mocker):
    """단일 텍스트 임베딩 생성 테스트"""
    # 가상의 임베딩 벡터 생성
    mock_embedding = [0.1] * 384  # text-embedding-small의 차원

    # AsyncMock을 사용하여 비동기 응답 모의 설정
    mock_response = mocker.Mock()
    mock_response.data = [mocker.Mock(embedding=mock_embedding)]
    
    # 비동기 모의 객체 생성
    async def mock_create(*args, **kwargs):
        return mock_response
        
    # OpenAI API 호출 모의 처리
    mocker.patch('openai.embeddings.create', side_effect=mock_create)
    
    agent = EmbeddingAgent()
    embedding = await agent.create_embedding("Test text")
    
    assert len(embedding) == 384
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.asyncio
async def test_batch_embeddings(mocker):
    """배치 임베딩 생성 테스트"""
    # 가상의 임베딩 벡터들 생성
    mock_embeddings = [[0.1] * 384 for _ in range(2)]
    
    # AsyncMock을 사용하여 비동기 응답 모의 설정
    mock_response = mocker.Mock()
    mock_response.data = [mocker.Mock(embedding=emb) for emb in mock_embeddings]
    
    # 비동기 모의 객체 생성
    async def mock_create(*args, **kwargs):
        return mock_response
        
    # OpenAI API 호출 모의 처리
    mocker.patch('openai.embeddings.create', side_effect=mock_create)
    
    agent = EmbeddingAgent()
    texts = ["First text", "Second text"]
    embeddings = await agent.create_batch_embeddings(texts)
    
    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)
    assert all(isinstance(x, float) for emb in embeddings for x in emb)

def test_calculate_similarity():
    """코사인 유사도 계산 테스트"""
    agent = EmbeddingAgent()
    
    # 테스트용 벡터 생성
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    vec3 = [-1.0, 0.0, 0.0]
    vec4 = [0.0, 1.0, 0.0]
    
    assert abs(agent.calculate_similarity(vec1, vec2) - 1.0) < 1e-6
    assert abs(agent.calculate_similarity(vec1, vec3) + 1.0) < 1e-6
    assert abs(agent.calculate_similarity(vec1, vec4)) < 1e-6

@pytest.mark.asyncio
async def test_find_most_similar():
    """가장 유사한 임베딩 찾기 테스트"""
    agent = EmbeddingAgent()
    
    query_embedding = [1.0, 0.0, 0.0]
    candidate_embeddings = [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    
    results = await agent.find_most_similar(query_embedding, candidate_embeddings, top_k=2)
    
    assert len(results) == 2
    assert results[0]['index'] == 0
    assert abs(results[0]['similarity'] - 1.0) < 1e-6
    assert results[1]['index'] == 1