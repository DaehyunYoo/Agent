import os
import logging
from typing import List, Dict, Any
import numpy as np
import openai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingAgent:
    """OpenAI의 text-embedding-ada-002 모델을 사용하는 임베딩 에이전트"""
    
    def __init__(self):
        """임베딩 에이전트 초기화"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        openai.api_key = self.api_key
        self.model = "text-embedding-ada-002"
    
    def create_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩을 생성합니다."""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    def create_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩을 일괄 생성합니다."""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            raise

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """두 임베딩 벡터 간의 코사인 유사도를 계산합니다."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cos_sim)

    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 1) -> List[Dict[str, Any]]:
        """쿼리 임베딩과 가장 유사한 상위 k개의 후보 임베딩을 찾습니다."""
        similarities = []
        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                'index': idx,
                'similarity': similarity
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]