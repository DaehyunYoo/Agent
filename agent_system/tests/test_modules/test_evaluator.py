import pytest
import json
from pathlib import Path
from agent_system.src.evaluation.evaluator import KMMLUEvaluator

@pytest.mark.asyncio
async def test_evaluator_initialization():
    """평가 시스템 초기화 테스트"""
    evaluator = KMMLUEvaluator()
    assert evaluator.rag_system is not None
    assert evaluator.data_loader is not None
    assert evaluator.results_dir.exists()

@pytest.mark.asyncio
async def test_evaluate_single_question(mocker):
    """단일 질문 평가 테스트"""
    evaluator = KMMLUEvaluator()
    
    # 가상의 RAG 시스템 응답 설정
    mock_rag_response = {
        'answer': 'The answer is A',
        'relevant_documents': [
            {'document': 'test doc', 'similarity': 0.9}
        ],
        'tokens_used': 100
    }
    
    async def mock_generate_answer(*args, **kwargs):
        return mock_rag_response
    
    # 모의 객체 설정
    mocker.patch.object(evaluator.rag_system, 'generate_answer', 
                       side_effect=mock_generate_answer)
    
    question = "Test question?"
    options = {
        'A': 'Option A',
        'B': 'Option B',
        'C': 'Option C',
        'D': 'Option D'
    }
    correct_answer = 1
    
    result = await evaluator.evaluate_single_question(
        question, options, correct_answer
    )
    
    assert result['question'] == question
    assert result['options'] == options
    assert result['correct_answer'] == correct_answer
    assert result['predicted_answer'] == 'A'
    assert result['is_correct'] == True
    assert result['tokens_used'] == 100

@pytest.mark.asyncio
async def test_evaluate_test_set(mocker):
    """전체 테스트셋 평가 테스트"""
    evaluator = KMMLUEvaluator()
    
    # 가상의 테스트 데이터 생성
    test_data = [
        (0, {
            'question': 'Test question 1?',
            'A': 'Option A',
            'B': 'Option B',
            'C': 'Option C',
            'D': 'Option D',
            'answer': 1
        }),
        (1, {
            'question': 'Test question 2?',
            'A': 'Option A',
            'B': 'Option B',
            'C': 'Option C',
            'D': 'Option D',
            'answer': 2
        })
    ]
    
    mock_df = mocker.Mock()
    mock_df.iterrows.return_value = test_data
    mock_df.__len__ = lambda x: len(test_data)  # len() 메서드 추가
    
    # 가상의 평가 결과 설정
    mock_eval_result = {
        'question': 'Test question',
        'options': {'A': 'Option A'},
        'correct_answer': 1,
        'predicted_answer': 'A',
        'is_correct': True,
        'model_response': 'A',
        'relevant_documents': [],
        'tokens_used': 100
    }
    
    async def mock_evaluate_question(*args, **kwargs):
        return mock_eval_result
    
    # 모의 객체 설정
    mocker.patch.object(evaluator.data_loader, 'load_criminal_law_test', 
                       return_value=mock_df)
    mocker.patch.object(evaluator, 'evaluate_single_question', 
                       side_effect=mock_evaluate_question)
    mocker.patch.object(evaluator, 'save_results')
    
    results = await evaluator.evaluate_test_set()
    
    assert results['total_questions'] == 2
    assert results['correct_predictions'] == 2
    assert results['accuracy'] == 1.0
    assert len(results['results']) == 2

def test_save_results(tmp_path):
    """결과 저장 테스트"""
    evaluator = KMMLUEvaluator()
    evaluator.results_dir = tmp_path
    
    test_results = {
        'total_questions': 10,
        'correct_predictions': 8,
        'accuracy': 0.8,
        'total_tokens_used': 1000,
        'average_tokens_per_question': 100,
        'results': [
            {
                'question': 'Test question',
                'options': {'A': 'Option A'},
                'correct_answer': 1,
                'predicted_answer': 'A',
                'is_correct': True,
                'model_response': 'A',
                'relevant_documents': [],
                'tokens_used': 100
            }
        ]
    }
    
    evaluator.save_results(test_results)
    
    # JSON 파일 확인
    json_path = tmp_path / 'evaluation_results.json'
    assert json_path.exists()
    with open(json_path, 'r', encoding='utf-8') as f:
        saved_results = json.load(f)
    assert saved_results['accuracy'] == 0.8
    
    # CSV 파일 확인
    csv_path = tmp_path / 'evaluation_metrics.csv'
    assert csv_path.exists()
    
    # Batch API 파일 확인
    input_path = tmp_path / 'batch_api_input.jsonl'
    output_path = tmp_path / 'batch_api_output.jsonl'
    assert input_path.exists()
    assert output_path.exists()

@pytest.mark.asyncio
async def test_system_initialization(mocker):
    """시스템 초기화 통합 테스트"""
    evaluator = KMMLUEvaluator()
    
    # RAG 시스템 초기화 모의 처리
    async def mock_initialize(*args, **kwargs):
        return None
        
    mocker.patch.object(evaluator.rag_system, 'initialize', 
                       side_effect=mock_initialize)
    
    await evaluator.initialize()
    assert evaluator.rag_system is not None