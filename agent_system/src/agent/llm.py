import os
import logging
from typing import List, Dict, Any, Optional
import openai
import asyncio
from dotenv import load_dotenv
import json
from pathlib import Path
from openai import OpenAI
import time

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self):
        """LLM Agent 초기화"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        self.output_dir = Path('outputs/batch_api')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_batch_answers_async(self, prompts: List[str], batch_size: int = 20) -> List[Dict[str, Any]]:
        """Batch API를 사용한 대량 처리"""
        try:
            # 배치 요청 준비
            tasks = []
            for idx, prompt in enumerate(prompts):
                task = {
                    "custom_id": f"task-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
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
                tasks.append(task)

            # JSONL 파일 생성
            timestamp = int(time.time())
            input_file = self.output_dir / f'batch_input_{timestamp}.jsonl'
            with open(input_file, 'w', encoding='utf-8') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')

            # 파일 업로드
            batch_file = self.client.files.create(
                file=open(input_file, 'rb'),
                purpose="batch"
            )

            # 배치 작업 생성
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            logger.info(f"Created batch job with ID: {batch_job.id}")

            # 작업 완료 대기
            while True:
                batch_job = self.client.batches.retrieve(batch_job.id)
                if batch_job.status == "completed":
                    break
                await asyncio.sleep(5)  # 5초마다 상태 체크

            # 결과 파일 다운로드
            result_file = self.client.files.content(batch_job.output_file_id).content
            result_file_path = self.output_dir / f'batch_output_{timestamp}.jsonl'
            
            with open(result_file_path, 'wb') as f:
                f.write(result_file)

            # 결과 처리
            results = []
            with open(result_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    response = {
                        'answer': result['response']['body']['choices'][0]['message']['content'],
                        'tokens_used': result['response']['body']['usage']['total_tokens'],
                        'task_id': result['custom_id']
                    }
                    results.append(response)

            # 원래 순서대로 결과 정렬
            sorted_results = sorted(results, key=lambda x: int(x['task_id'].split('-')[1]))
            return sorted_results

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def generate_batch_answers(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """동기적 배치 처리 래퍼 메서드"""
        return asyncio.run(self.generate_batch_answers_async(prompts))

    def extract_answer_letter(self, response: str) -> Optional[str]:
        """모델 응답에서 답안 문자(A, B, C, D) 추출"""
        if not response:
            return None
            
        response = response.upper()
        valid_answers = {'A', 'B', 'C', 'D'}
        
        words = response.split()
        for word in words:
            if word in valid_answers:
                return word
        return None

    def validate_answer(self, model_answer: str, correct_answer: int) -> bool:
        """답안 검증"""
        answer_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        extracted_answer = self.extract_answer_letter(model_answer)
        
        if not extracted_answer:
            return False
            
        return answer_map[extracted_answer] == correct_answer