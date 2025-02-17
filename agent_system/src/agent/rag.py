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
        검색 쿼리를 강화
        
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
    
    def validate_answer(self, predicted_answer: str, context_docs: List[Dict]) -> bool:
        """
        단순화된 답변 검증
        가장 유사도가 높은 문서의 답안만 확인
        """
        if not context_docs or not predicted_answer:
            return True
            
        # 가장 유사도가 높은 문서만 확인
        top_doc = max(context_docs, key=lambda x: x['similarity'])
        if 'Answer: ' not in top_doc['document']:
            return True
            
        answer_num = top_doc['document'].split('Answer: ')[1][0]
        answer_letter = chr(ord('A') + int(answer_num) - 1)
        
        # 유사도가 높은 경우에만 검증 수행
        if top_doc['similarity'] > 0.8:
            return predicted_answer == answer_letter
        
        return True

    def generate_answer(self, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        """
        단순화된 답변 생성
        """
        try:
            relevant_docs = self.retrieve_relevant_documents(question)
            context = "\n\n".join([doc['document'] for doc in relevant_docs])
            prompt = self._construct_prompt(question, options, context)
            
            response = self.llm.generate_answer(prompt)
            predicted_answer = response['answer'].strip().upper()
            
            # 간단한 검증만 수행
            is_valid = self.validate_answer(predicted_answer, relevant_docs)
            
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

    def _construct_prompt(self, question: str, options: Dict[str, str], context: str) -> str:
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        
        return f"""As a criminal law expert, analyze the following question using the provided context. Select the most accurate answer (A, B, C, or D).

    Context:
    {context}

    Question:
    {question}

    Options:
    {options_text}

    Answer with only a single letter (A/B/C/D):"""