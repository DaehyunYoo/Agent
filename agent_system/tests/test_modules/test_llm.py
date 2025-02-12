import pytest
from agent_system.src.agent.llm import LLMAgent

def test_llm_initialization(test_env_vars):
    """LLM 초기화 테스트"""
    agent = LLMAgent()
    assert agent.api_key == 'test-api-key'
    assert agent.model == 'gpt-4o-mini'

def test_extract_answer_letter():
    """답안 추출 기능 테스트"""
    agent = LLMAgent()
    assert agent.extract_answer_letter("The answer is A") == "A"
    assert agent.extract_answer_letter("B is correct") == "B"
    assert agent.extract_answer_letter("invalid answer") is None

def test_validate_answer():
    """답안 검증 기능 테스트"""
    agent = LLMAgent()
    assert agent.validate_answer("A", 1) == True
    assert agent.validate_answer("B", 1) == False

@pytest.mark.integration
def test_generate_answer(mocker):
    """실제 API 호출을 모의 응답으로 테스트"""
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content="A"))]
    mock_response.usage = mocker.Mock(total_tokens=10)
    
    mocker.patch('openai.chat.completions.create', return_value=mock_response)
    
    agent = LLMAgent()
    prompt = """Question: What is criminal law?"""
    
    response = agent.generate_answer(prompt)
    assert 'answer' in response
    assert 'tokens_used' in response