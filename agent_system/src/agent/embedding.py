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
    
    def __init__(self):
        """임베딩 에이전트 초기화"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        openai.api_key = self.api_key
        self.model = "text-embedding-3-small"
    
    def create_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 생성"""
        try:
            # 텍스트 전처리 및 강화
            processed_text = self._enhance_query(text)
            
            response = openai.embeddings.create(
                model=self.model,
                input=processed_text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def create_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 일괄 생성"""
        try:
            # 모든 텍스트 전처리 및 강화
            processed_texts = [self._enhance_query(text) for text in texts]
            
            response = openai.embeddings.create(
                model=self.model,
                input=processed_texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            raise

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """두 임베딩 벡터 간의 유사도 계산"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # L2 정규화
            vec1_normalized = vec1 / np.linalg.norm(vec1)
            vec2_normalized = vec2 / np.linalg.norm(vec2)
            
            # 코사인 유사도 계산
            cos_sim = np.dot(vec1_normalized, vec2_normalized)
            
            return float(cos_sim)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise

    def find_most_similar(self, 
                        query_embedding: List[float], 
                        candidate_embeddings: List[List[float]], 
                        top_k: int = 9,           # top_k 기본값 증가
                        similarity_threshold: float = 0.7  # 유사도 임계값
                        ) -> List[Dict[str, Any]]:
        """쿼리 임베딩과 가장 유사한 상위 k개의 후보 임베딩 찾기"""
        try:
            similarities = []
            for idx, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                if similarity >= similarity_threshold:  # 임계값 이상만 포함
                    similarities.append({
                        'index': idx,
                        'similarity': similarity
                    })
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 로깅 추가
            if similarities:
                logger.info(f"Found {len(similarities)} documents above threshold. "
                        f"Top similarity: {similarities[0]['similarity']:.4f}")
            else:
                logger.warning("No documents found above similarity threshold")
                
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리
        - 불필요한 공백 제거
        - 특수 문자 처리
        - 텍스트 정규화
        """
        # 기본 전처리
        text = text.strip()
        text = ' '.join(text.split())  # 다중 공백 제거
        
        # Criminal Law 도메인 특화 전처리
        # 법률 용어와 괄호 내용을 보존
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('/', ' / ')
        
        return text

    def _enhance_query(self, text: str) -> str:
        """검색 쿼리 강화
        - 중요 키워드 강조
        - 도메인 컨텍스트 추가
        """
        # 법률 도메인 컨텍스트 추가
        context = "In the context of criminal law: "
        return context + self._preprocess_text(text)