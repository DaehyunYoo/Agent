import os
import logging
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
from pathlib import Path

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    """GPT-4o-mini를 사용하는 LLM Agent 클래스"""
    
    def __init__(self):
        """LLM Agent 초기화"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        openai.api_key = self.api_key
        self.model = "gpt-4o-mini"  # GPT-4o-mini 모델 지정
        
    def generate_answer(self, prompt: str) -> Dict[str, Any]:
        """
        주어진 프롬프트에 대한 응답을 생성합니다.
        
        Args:
            prompt (str): 질문과 보기를 포함한 프롬프트
            
        Returns:
            Dict[str, Any]: 모델의 응답과 메타데이터를 포함한 딕셔너리
        """
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are an expert in criminal law. 
                    Analyze the question carefully and select the most appropriate answer 
                    from the given options. Provide your answer as a single letter (A, B, C, or D)."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # 결정적인 응답을 위해 temperature를 0으로 설정
                max_tokens=50,  # 간단한 답변을 위한 토큰 제한
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'model': self.model,
                'tokens_used': response.usage.total_tokens,
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise


    def extract_answer_letter(self, response: str) -> Optional[str]:
        """
        모델 응답에서 답안 문자(A, B, C, D)를 추출
        """
        if not response:
            return None
            
        response = response.upper()
        valid_answers = {'A', 'B', 'C', 'D'}
        
        # 주어진 문자열에서 유효한 답안 찾기
        # A가 단어의 일부로 포함된 경우(예: "invalid answer")도 체크
        words = response.split()
        for word in words:
            if word in valid_answers:
                return word
        return None

    def validate_answer(self, model_answer: str, correct_answer: int) -> bool:
        """
        모델의 답안이 정답과 일치하는지 확인
        
        Args:
            model_answer (str): 모델이 생성한 답안 (A, B, C, D)
            correct_answer (int): 정답 번호 (1, 2, 3, 4)
            
        Returns:
            bool: 답안 일치 여부
        """
        answer_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        extracted_answer = self.extract_answer_letter(model_answer)
        
        if not extracted_answer:
            return False
            
        return answer_map[extracted_answer] == correct_answer
    