import logging
from pathlib import Path
import json
from typing import List, Dict, Any
import pandas as pd
from ..agent.rag import RAGSystem
from ..data.loader import KMMLUDataLoader
from .timer import timer
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMMLUEvaluator:
    """KMMLU Criminal Law 테스트셋에 대한 평가 시스템"""
    
    def __init__(self):
        """평가 시스템 초기화"""
        self.rag_system = RAGSystem()
        self.data_loader = KMMLUDataLoader()
        self.results_dir = Path(__file__).parents[3] / 'outputs'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self):
        """RAG 시스템 초기화"""
        self.rag_system.initialize()
        logger.info("Evaluation system initialized")
        
    def evaluate_single_question(self, 
                               question: str, 
                               options: Dict[str, str], 
                               correct_answer: int) -> Dict[str, Any]:
        """단일 질문에 대한 평가 수행"""
        try:
            # RAG 시스템을 통한 답변 생성
            response = self.rag_system.generate_answer(question, options)
            
            # 답안 추출 및 검증
            predicted_answer = self.rag_system.llm.extract_answer_letter(response['answer'])
            is_correct = self.rag_system.llm.validate_answer(predicted_answer, correct_answer)
            
            return {
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'model_response': response['answer'],
                'relevant_documents': response['relevant_documents'],
                'tokens_used': response.get('tokens_used', 0)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating question: {str(e)}")
            raise
            
    def evaluate_test_set(self) -> Dict[str, Any]:
        """전체 테스트셋 평가 수행"""
        try:
            df = self.data_loader.load_criminal_law_test()
            total_questions = len(df)
            batch_size = 10  # 배치 크기 설정
            
            results = []
            correct_predictions = 0
            total_tokens = 0
            
            # 배치 처리
            for i in range(0, total_questions, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                for _, row in batch_df.iterrows():
                    options = {
                        'A': row['A'],
                        'B': row['B'],
                        'C': row['C'],
                        'D': row['D']
                    }
                    
                    result = self.evaluate_single_question(
                        row['question'],
                        options,
                        row['answer']
                    )
                    
                    results.append(result)
                    if result['is_correct']:
                        correct_predictions += 1
                    total_tokens += result['tokens_used']
                    
                # 중간 결과 로깅
                current_accuracy = correct_predictions / (i + len(batch_df))
                logger.info(f"Batch {i//batch_size + 1} completed. Current accuracy: {current_accuracy:.2%}")
            
            accuracy = correct_predictions / total_questions
            
            return {
                'total_questions': total_questions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'total_tokens_used': total_tokens,
                'average_tokens_per_question': total_tokens / total_questions,
                'results': results
            }
        
        except Exception as e:
            logger.error(f"Error evaluating test set: {str(e)}")
            raise

            
    def save_results(self, results: Dict[str, Any]):
        """
        평가 결과 저장
        
        Args:
            results (Dict[str, Any]): 평가 결과
        """
        try:
            # 상세 결과를 JSON으로 저장
            results_path = self.results_dir / 'evaluation_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 주요 메트릭을 CSV로 저장
            metrics = {
                'total_questions': [results['total_questions']],
                'correct_predictions': [results['correct_predictions']],
                'accuracy': [results['accuracy']],
                'total_tokens_used': [results['total_tokens_used']],
                'average_tokens_per_question': [results['average_tokens_per_question']]
            }
            
            metrics_df = pd.DataFrame(metrics)
            metrics_path = self.results_dir / 'evaluation_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            
            # OpenAI Batch API용 파일 생성
            self.save_batch_api_files(results['results'])
            
            logger.info(f"Evaluation results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def save_batch_api_files(self, results: List[Dict[str, Any]]):
        """
        OpenAI Batch API 입출력 파일 저장
        
        Args:
            results (List[Dict[str, Any]]): 평가 결과 리스트
        """
        try:
            # Input JSONL 생성
            inputs = []
            for result in results:
                input_entry = {
                    'question': result['question'],
                    'options': result['options']
                }
                inputs.append(input_entry)
            
            input_path = self.results_dir / 'batch_api_input.jsonl'
            with open(input_path, 'w', encoding='utf-8') as f:
                for entry in inputs:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Output JSONL 생성
            outputs = []
            for result in results:
                output_entry = {
                    'model_response': result['model_response'],
                    'predicted_answer': result['predicted_answer'],
                    'correct_answer': result['correct_answer'],
                    'is_correct': result['is_correct'],
                    'tokens_used': result['tokens_used']
                }
                outputs.append(output_entry)
            
            output_path = self.results_dir / 'batch_api_output.jsonl'
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in outputs:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            logger.error(f"Error saving batch API files: {str(e)}")
            raise
        
def main():
    try:
        total_start_time = time.time()
        
        # 1. 시스템 초기화 시간 측정
        with timer("System Initialization"):
            evaluator = KMMLUEvaluator()
            evaluator.initialize()
        
        # 2. 평가 수행 시간 측정 (batch API 응답 대기 시간 제외)
        evaluation_start_time = time.time()
        results = None
        
        with timer("Test Set Evaluation"):
            results = evaluator.evaluate_test_set()
        
        evaluation_time = (time.time() - evaluation_start_time) / 60  # 분으로 변환
        
        # 3. 결과 저장 시간 측정
        with timer("Results Saving"):
            evaluator.save_results(results)
        
        # 4. 총 소요 시간 계산 및 로깅
        total_time = (time.time() - total_start_time) / 60  # 분으로 변환
        
        # 최종 시간 정보 로깅
        logger.info("\n=== Time Analysis ===")
        logger.info(f"Total processing time: {total_time:.2f} minutes")
        logger.info(f"Evaluation time (excluding API calls): {evaluation_time:.2f} minutes")
        logger.info(f"Evaluation completed with accuracy: {results['accuracy']:.2%}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()