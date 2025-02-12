import pytest
from agent_system.src.agent.rag import RAGSystem

@pytest.mark.asyncio
async def test_rag_initialization():
    """RAG 시스템 초기화 테스트"""
    rag = RAGSystem()
    assert rag.llm is not None
    assert rag.embedding_agent is not None
    assert rag.data_loader is not None
    assert len(rag.document_embeddings) == 0
    assert len(rag.documents) == 0

@pytest.mark.asyncio
async def test_initialize_system(mocker):
    """시스템 초기화 및 임베딩 생성 테스트"""
    # 가상의 데이터프레임 생성
    mock_df = mocker.Mock()
    mock_df.iterrows.return_value = [
        (0, {
            'question': 'Test question?',
            'A': 'Option A',
            'B': 'Option B',
            'C': 'Option C',
            'D': 'Option D',
            'answer': 1,
            'Category': 'Criminal Law'
        })
    ]
    
    # 가상의 임베딩 생성
    mock_embeddings = [[0.1] * 384]  # text-embedding-small 차원
    
    # 모의 객체 설정
    async def mock_create_embeddings(*args, **kwargs):
        return mock_embeddings
        
    rag = RAGSystem()
    mocker.patch.object(rag.data_loader, 'load_criminal_law_test', return_value=mock_df)
    mocker.patch.object(rag.embedding_agent, 'create_batch_embeddings', side_effect=mock_create_embeddings)
    
    await rag.initialize()
    
    assert len(rag.documents) == 1
    assert len(rag.document_embeddings) == 1

@pytest.mark.asyncio
async def test_retrieve_relevant_documents(mocker):
    """관련 문서 검색 테스트"""
    rag = RAGSystem()
    
    # 테스트 데이터 설정
    rag.documents = ["Test document 1", "Test document 2"]
    rag.document_embeddings = [[0.1] * 384, [0.2] * 384]
    
    # 가상의 임베딩 생성
    mock_query_embedding = [0.1] * 384
    
    async def mock_create_embedding(*args, **kwargs):
        return mock_query_embedding
        
    async def mock_find_similar(*args, **kwargs):
        return [
            {'index': 0, 'similarity': 0.95},
            {'index': 1, 'similarity': 0.85}
        ]
    
    # 모의 객체 설정
    mocker.patch.object(rag.embedding_agent, 'create_embedding', side_effect=mock_create_embedding)
    mocker.patch.object(rag.embedding_agent, 'find_most_similar', side_effect=mock_find_similar)
    
    results = await rag.retrieve_relevant_documents("test query", top_k=2)
    
    assert len(results) == 2
    assert results[0]['similarity'] == 0.95
    assert results[1]['similarity'] == 0.85
    assert results[0]['document'] == "Test document 1"

@pytest.mark.asyncio
async def test_generate_answer(mocker):
    """답변 생성 테스트"""
    rag = RAGSystem()
    
    # 가상의 관련 문서 검색 결과
    mock_relevant_docs = [
        {'document': 'Test document 1', 'similarity': 0.95},
        {'document': 'Test document 2', 'similarity': 0.85}
    ]
    
    # 가상의 LLM 응답
    mock_llm_response = {
        'answer': 'A',
        'model': 'gpt-4o-mini',
        'tokens_used': 150
    }
    
    async def mock_retrieve_docs(*args, **kwargs):
        return mock_relevant_docs
        
    async def mock_generate_answer(*args, **kwargs):
        return mock_llm_response
    
    # 모의 객체 설정
    mocker.patch.object(rag, 'retrieve_relevant_documents', side_effect=mock_retrieve_docs)
    mocker.patch.object(rag.llm, 'generate_answer', side_effect=mock_generate_answer)
    
    question = "Test question?"
    options = {
        'A': 'Option A',
        'B': 'Option B',
        'C': 'Option C',
        'D': 'Option D'
    }
    
    response = await rag.generate_answer(question, options)
    
    assert response['answer'] == 'A'
    assert response['model'] == 'gpt-4o-mini'
    assert response['tokens_used'] == 150
    assert len(response['relevant_documents']) == 2

def test_prompt_construction():
    """프롬프트 구성 테스트"""
    rag = RAGSystem()
    
    question = "What is criminal law?"
    options = {
        'A': 'Civil law',
        'B': 'Public law',
        'C': 'Private law',
        'D': 'International law'
    }
    context = "Criminal law is a branch of public law..."
    
    prompt = rag._construct_prompt(question, options, context)
    
    assert question in prompt
    assert all(option in prompt for option in options.values())
    assert context in prompt
    assert "provide your answer as a single letter" in prompt.lower()