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

    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """관련 문서 검색 개선"""
        try:
            # 쿼리 확장 - 핵심 키워드 추출
            keywords = self._extract_keywords(query)
            expanded_query = f"{query} {' '.join(keywords)}"
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_agent.create_embedding(expanded_query)
            
            # 유사도 임계값 설정
            similarity_threshold = 0.75
            
            # 가장 유사한 문서 찾기
            similar_docs = self.embedding_agent.find_most_similar(
                query_embedding,
                self.document_embeddings,
                top_k=top_k
            )
            
            # 임계값 이상의 문서만 선택
            filtered_docs = [
                doc for doc in similar_docs 
                if doc['similarity'] > similarity_threshold
            ]
            
            # 결과 형식화
            results = []
            for doc in filtered_docs:
                results.append({
                    'document': self.documents[doc['index']],
                    'similarity': doc['similarity']
                })
                    
            return results
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def _extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        # 간단한 키워드 추출 로직
        keywords = [word for word in text.split() if len(word) > 1]
        return keywords[:3]  # 상위 3개 키워드 사용

    def generate_answer(self, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        """RAG를 사용하여 답변 생성"""
        try:
            # 관련 문서 검색
            relevant_docs = self.retrieve_relevant_documents(question)
            
            # 프롬프트 구성
            context = "\n\n".join([doc['document'] for doc in relevant_docs])
            prompt = self._construct_prompt(question, options, context)
            
            # LLM을 사용하여 답변 생성
            response = self.llm.generate_answer(prompt)
            
            return {
                'answer': response['answer'],
                'relevant_documents': relevant_docs,
                'model': response.get('model'),
                'tokens_used': response.get('tokens_used')
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def _construct_prompt(self, question: str, options: Dict[str, str], context: str) -> str:
        """LLM 프롬프트 구성 개선"""
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        
        return f"""You are an expert in criminal law, specializing in analyzing and answering questions about legal concepts and principles.
        
        Task: Based on the given context and your expertise, select the most accurate answer to the criminal law question.

        Context:
        {context}

        Question: {question}

        Options:
        {options_text}

        Instructions:
        1. Carefully analyze the question and all answer options
        2. Consider the relevant context provided
        3. Use your criminal law expertise to evaluate each option
        4. Select the most accurate answer
        5. Provide your answer as a single letter (A, B, C, or D) without explanation

        Answer: """

def main():
    rag = RAGSystem()
    rag.initialize()
    logger.info("RAG system initialized successfully")

if __name__ == "__main__":
    main()