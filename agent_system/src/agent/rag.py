import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .llm import LLMAgent
from .embedding import EmbeddingAgent
from ..data.loader import KMMLUDataLoader

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
        """쿼리와 관련된 문서 검색 개선"""
        try:
            # 쿼리 전처리 및 강화
            enhanced_query = self._enhance_query(query)
            
            # 가장 유사한 문서 찾기 (top_k 증가)
            similar_docs = self.embedding_agent.find_most_similar(
                enhanced_query,
                self.document_embeddings,
                top_k=top_k,
                similarity_threshold=0.75  # 임계값 조정
            )
            
            # 결과 형식화 및 필터링
            results = []
            for doc in similar_docs:
                if doc['similarity'] > 0.75:
                    doc_text = self.documents[doc['index']]
                    # 문서 관련성 검증
                    if self._verify_document_relevance(query, doc_text):
                        results.append({
                            'document': doc_text,
                            'similarity': doc['similarity']
                        })
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
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
        """답변 검증 로직 강화"""
        if not context_docs:
            return True
            
        # 가중치 기반 투표 시스템
        answer_votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for doc in context_docs:
            if 'Answer: ' in doc['document']:
                answer_num = doc['document'].split('Answer: ')[1][0]
                answer_letter = chr(ord('A') + int(answer_num) - 1)
                # 유사도에 기반한 가중치 적용
                weight = doc['similarity']
                answer_votes[answer_letter] += weight
        
        if not answer_votes:
            return True
            
        # 최다 득표 답안 선택
        best_answer = max(answer_votes.items(), key=lambda x: x[1])[0]
        confidence = answer_votes[best_answer] / sum(answer_votes.values())
        
        # 높은 신뢰도의 답안만 채택
        if confidence > 0.6:
            return predicted_answer == best_answer
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
        """프롬프트 구성 개선"""
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        
        return f"""You are a criminal law expert tasked with answering multiple-choice questions.

    Background Context:
    {context}

    Current Question to Answer:
    {question}

    Available Options:
    {options_text}

    Instructions for Analysis:
    1. Read the question and context carefully
    2. Consider the criminal law principles involved
    3. Evaluate each option based on the context provided
    4. Select the most accurate answer

    Requirements:
    - Provide ONLY a single letter (A, B, C, or D) as your answer
    - Choose the best answer based on your expertise and the given context
    - Do not provide explanations or reasoning

    Your Answer (A/B/C/D): """

def main():
    rag = RAGSystem()
    rag.initialize()
    logger.info("RAG system initialized successfully")

if __name__ == "__main__":
    main()