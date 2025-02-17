import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from ..agent.rag import RAGSystem
from ..data.loader import KMMLUDataLoader
import asyncio
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMMLUEvaluator:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.data_loader = KMMLUDataLoader()
        self.output_dir = Path(__file__).parents[3] / 'outputs'
        self.output_dir.mkdir(exist_ok=True)
        
    def initialize(self):
        try:
            logger.info("Initializing evaluation system...")
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.rag_system.initialize()
            logger.info("Evaluation system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing evaluation system: {str(e)}")
            raise

    async def evaluate_test_set_async(self) -> Dict[str, Any]:
        try:
            df = self.data_loader.load_criminal_law_test()
            logger.info(f"Loaded test set with {len(df)} questions")
            
            # 프롬프트 준비
            all_prompts = []
            all_metadata = []
            
            for idx, row in df.iterrows():
                try:
                    relevant_docs = self.rag_system.retrieve_relevant_documents(row['question'])
                    context = "\n\n".join([doc['document'] for doc in relevant_docs])
                    
                    prompt = self.rag_system._construct_prompt(
                        row['question'],
                        {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']},
                        context
                    )
                    
                    metadata = {
                        'question': row['question'],
                        'options': {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']},
                        'correct_answer': row['answer'],
                        'relevant_documents': relevant_docs
                    }
                    
                    all_prompts.append(prompt)
                    all_metadata.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Error preparing question {idx}: {str(e)}")
                    continue
            
            logger.info(f"Prepared {len(all_prompts)} prompts for evaluation")
            
            # Save batch API input
            input_tasks = []
            for idx, prompt in enumerate(all_prompts):
                task = {
                    "custom_id": f"task-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.rag_system.llm.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are an expert in criminal law. 
                                Analyze the question carefully and select the most appropriate answer 
                                from the given options. Provide your answer as a single letter (A, B, C, or D)."""
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0,
                        "max_tokens": 50
                    }
                }
                input_tasks.append(task)

            with open(self.output_dir / 'batch_api_input.jsonl', 'w', encoding='utf-8') as f:
                for task in input_tasks:
                    f.write(json.dumps(task) + '\n')
            
            # Batch API를 통한 응답 생성
            responses = await self.rag_system.llm.generate_batch_answers_async(all_prompts)
            
            # Save batch API output
            with open(self.output_dir / 'batch_api_output.jsonl', 'w', encoding='utf-8') as f:
                for response in responses:
                    f.write(json.dumps(response) + '\n')
            
            # 결과 처리
            all_results = []
            for response, metadata in zip(responses, all_metadata):
                try:
                    predicted_answer = self.rag_system.llm.extract_answer_letter(response['answer'])
                    if predicted_answer is None:
                        continue
                        
                    is_correct = self.rag_system.llm.validate_answer(
                        predicted_answer,
                        metadata['correct_answer']
                    )
                    
                    result = {
                        'question': metadata['question'],
                        'options': metadata['options'],
                        'correct_answer': metadata['correct_answer'],
                        'predicted_answer': predicted_answer,
                        'is_correct': is_correct,
                        'model_response': response['answer'],
                        'tokens_used': response['tokens_used']
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    continue
            
            if not all_results:
                raise ValueError("No valid results were generated")
            
            # 결과 통계 계산
            correct_predictions = sum(1 for r in all_results if r['is_correct'])
            total_tokens = sum(r['tokens_used'] for r in all_results)
            
            accuracy = correct_predictions / len(all_results)
            
            y_true = [r['correct_answer'] - 1 for r in all_results]
            y_pred = [ord(r['predicted_answer']) - ord('A') for r in all_results]
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            return {
                'total_questions': len(df),
                'processed_questions': len(all_results),
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'f1_score': f1,
                'total_tokens_used': total_tokens,
                'average_tokens_per_question': total_tokens / len(all_results),
                'results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating test set: {str(e)}")
            raise
            
    def evaluate_test_set(self) -> Dict[str, Any]:
        """동기적 래퍼 메서드"""
        return asyncio.run(self.evaluate_test_set_async())

    def save_results(self, results: Dict[str, Any]):
        try:
            # Save evaluation results
            with open(self.output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Save evaluation metrics
            metrics_df = pd.DataFrame([{
                'total_questions': results['total_questions'],
                'processed_questions': results['processed_questions'],
                'correct_predictions': results['correct_predictions'],
                'accuracy': round(results['accuracy'], 6),
                'f1_score': round(results['f1_score'], 6),
                'total_tokens_used': results['total_tokens_used'],
                'average_tokens_per_question': results['average_tokens_per_question']
            }])
            
            metrics_df.to_csv(self.output_dir / 'evaluation_metrics.csv', index=False)
            
            logger.info("Results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

async def main_async():
    try:
        total_start_time = time.time()
        
        logger.info("Starting evaluation process...")
        evaluator = KMMLUEvaluator()
        evaluator.initialize()
        
        results = await evaluator.evaluate_test_set_async()
        
        if results is not None:
            evaluator.save_results(results)
            logger.info(f"Accuracy: {results['accuracy']:.2%}")
            logger.info(f"F1 Score: {results['f1_score']:.2%}")
        
        total_time = (time.time() - total_start_time) / 60
        logger.info(f"Total processing time: {total_time:.2f} minutes")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()