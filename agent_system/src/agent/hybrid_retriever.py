from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import numpy as np

class HybridRetriever:
    """하이브리드 검색 시스템"""
    
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        
    def prepare_corpus(self, documents: List[str]):
        """BM25를 위한 코퍼스 준비"""
        # 문서를 토큰화
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        # BM25 모델 초기화
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def get_bm25_scores(self, query: str, documents: List[str]) -> List[float]:
        """BM25 점수 계산"""
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)
    
    def combine_scores(self, 
                      embedding_scores: List[float], 
                      bm25_scores: List[float], 
                      alpha: float = 0.7) -> List[float]:
        """임베딩과 BM25 점수 결합"""
        # 점수 정규화
        norm_embedding = np.array(embedding_scores) / max(embedding_scores)
        norm_bm25 = np.array(bm25_scores) / max(bm25_scores)
        
        # 가중치 결합
        combined_scores = alpha * norm_embedding + (1 - alpha) * norm_bm25
        return combined_scores.tolist()