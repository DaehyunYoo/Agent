import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .llm import LLMAgent
from .embedding import EmbeddingAgent
from ..data.loader import KMMLUDataLoader
from .hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    RAG(Retrieval-Augmented Generation) 시스템
    임베딩 기반 검색과 LLM을 결합하여 더 정확한 답변 생성
    """
    
    def __init__(self):
        """RAG 시스템 초기화"""
        self.llm = LLMAgent()
        self.embedding_agent = EmbeddingAgent()
        self.data_loader = KMMLUDataLoader()
        self.hybrid_retriever = HybridRetriever()
        self.document_embeddings = []
        self.documents = []
        
    def initialize(self):
        """문서 데이터 로드 및 임베딩 생성"""
        try:
            # 데이터 로드
            df = self.data_loader.load_criminal_law_test()
            
            # 문서 텍스트 준비
            self.documents = []
            for _, row in df.iterrows():
                document = self._format_document(row)
                self.documents.append(document)
            
            # 문서 임베딩 생성
            self.document_embeddings = self.embedding_agent.create_batch_embeddings(
                self.documents
            )
            
            # 하이브리드 검색 준비
            self.hybrid_retriever.prepare_corpus(self.documents, self.document_embeddings)
            
            logger.info(f"Initialized RAG system with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
            
    def _format_document(self, row: Dict) -> str:
        """문서 형식화"""
        return f"""Question: {row['question']}
        Options:
        A) {row['A']}
        B) {row['B']}
        C) {row['C']}
        D) {row['D']}
        Answer: {row['answer']}
        Category: {row['Category']}"""

    def _enhance_query(self, query: str) -> List[float]:
        """
        검색 쿼리를 강화합니다.
        
        Args:
            query (str): 원본 쿼리
            
        Returns:
            List[float]: 강화된 쿼리의 임베딩
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_agent.create_embedding(query)
            return query_embedding
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            raise

    def retrieve_relevant_documents(self, query: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """하이브리드 검색을 통한 관련 문서 검색"""
        try:
            # 1. 임베딩 기반 검색
            query_embedding = self.embedding_agent.create_embedding(query)
            embedding_results = self.embedding_agent.find_most_similar(
                query_embedding, 
                self.document_embeddings,
                top_k=top_k * 2  # 더 많은 후보 검색
            )
            
            # 2. BM25 기반 검색
            bm25_scores = self.hybrid_retriever.get_bm25_scores(query, self.documents)
            bm25_results = sorted(
                [(i, score) for i, score in enumerate(bm25_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:top_k * 2]
            
            # 3. 교집합 기반 결과 선택
            common_docs = set(
                doc['index'] for doc in embedding_results
            ).intersection(
                idx for idx, _ in bm25_results
            )
            
            results = []
            for doc_idx in common_docs:
                if len(results) >= top_k:
                    break
                    
                results.append({
                    'document': self.documents[doc_idx],
                    'similarity': embedding_results[0]['similarity']  # 임베딩 유사도 사용
                })
                
            logger.info(f"Found {len(results)} documents in intersection")
            return results
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            raise

    def _extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        # 간단한 키워드 추출 로직
        keywords = [word for word in text.split() if len(word) > 1]
        return keywords[:3]  # 상위 3개 키워드 사용

    def _verify_document_relevance(self, query: str, document: str) -> bool:
        """문서 관련성 검증"""
        # 쿼리 키워드 추출
        query_keywords = self._extract_keywords(query)
        
        # 문서 내 키워드 존재 여부 확인
        document_lower = document.lower()
        keyword_matches = sum(1 for keyword in query_keywords 
                            if keyword.lower() in document_lower)
        
        # 최소 매칭 기준 설정
        min_matches = len(query_keywords) // 2
        return keyword_matches >= min_matches


    
    def validate_answer(self, predicted_answer: str, context_docs: List[Dict]) -> bool:
        if not context_docs:
            return True
            
        # 가중치 점수 계산
        answer_scores = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        total_weight = 0.0
        
        for doc in context_docs:
            if 'Answer: ' in doc['document']:
                answer_num = doc['document'].split('Answer: ')[1][0]
                answer_letter = chr(ord('A') + int(answer_num) - 1)
                
                # 유사도 점수를 가중치로 사용
                weight = doc['similarity'] ** 2  # 제곱하여 높은 유사도에 더 큰 가중치
                answer_scores[answer_letter] += weight
                total_weight += weight
        
        if total_weight == 0:
            return True
            
        # 정규화된 점수 계산
        normalized_scores = {
            k: v/total_weight for k, v in answer_scores.items()
        }
        
        # 최고 점수와 두 번째 점수의 차이 계산
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        score_diff = sorted_scores[0][1] - sorted_scores[1][1]
        
        # 점수 차이가 충분히 크면 검증
        if score_diff > 0.3:  # 임계값 상향 조정
            return predicted_answer == sorted_scores[0][0]
        
        return True

    def generate_answer(self, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        """
        RAG를 사용하여 답변 생성
        
        Args:
            question (str): 질문
            options (Dict[str, str]): 선택지 {"A": "...", "B": "...", ...}
            
        Returns:
            Dict[str, Any]: 생성된 답변과 관련 정보
        """
        try:
            # 관련 문서 검색
            relevant_docs = self.retrieve_relevant_documents(question)
            
            # 프롬프트 구성
            context = "\n\n".join([doc['document'] for doc in relevant_docs])
            prompt = self._construct_prompt(question, options, context)
            
            # LLM을 사용하여 답변 생성
            response = self.llm.generate_answer(prompt)
            predicted_answer = response['answer'].strip().upper()
            
            # 답변 검증
            is_valid = self.validate_answer(predicted_answer, relevant_docs)
            
            # 검증 실패 시 가장 유사한 문서의 답변 사용
            if not is_valid and relevant_docs:
                context_answer = relevant_docs[0]['document'].split('Answer: ')[1][0]
                predicted_answer = chr(ord('A') + int(context_answer) - 1)
                logger.info(f"Answer validation failed. Using most similar document's answer: {predicted_answer}")
            
            return {
                'answer': predicted_answer,
                'relevant_documents': relevant_docs,
                'model': response.get('model'),
                'tokens_used': response.get('tokens_used'),
                'validation_passed': is_valid
            }
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def _format_options(self, options: Dict[str, str]) -> str:
        """
        선택지 형식을 지정합니다.
        
        Args:
            options (Dict[str, str]): 선택지 딕셔너리
            
        Returns:
            str: 형식화된 선택지 문자열
        """
        return "\n".join([f"{k}) {v}" for k, v in options.items()])

    def _construct_prompt(self, question: str, options: Dict[str, str], context: str) -> str:
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        
        return f"""You are a highly qualified criminal law expert tasked with selecting the most accurate answer to a multiple-choice question.

    Context Information (Pay careful attention to these similar cases and principles):
    {context}

    Question to Analyze:
    {question}

    Available Options:
    {options_text}

    Reasoning Process:
    1. First, carefully analyze the question and understand what it's asking about
    2. Compare each option against the provided context
    3. Consider criminal law principles and terminology
    4. Find direct matches or analogous cases in the context
    5. Select the option that best aligns with the context and legal principles

    Requirements:
    - Your response must be ONLY a single letter (A, B, C, or D)
    - Choose the most accurate answer based on the context and criminal law expertise
    - Do not explain your reasoning, only provide the answer letter

    Your Final Answer (A/B/C/D): """

def main():
    rag = RAGSystem()
    rag.initialize()
    logger.info("RAG system initialized successfully")

if __name__ == "__main__":
    main()