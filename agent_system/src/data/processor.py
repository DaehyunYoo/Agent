import pandas as pd
import logging
from typing import List, Dict, Any
from .loader import KMMLUDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """데이터 전처리를 담당하는 클래스"""
    
    def __init__(self):
        self.loader = KMMLUDataLoader()
        
    def preprocess_questions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """질문 데이터를 전처리하여 LLM에 적합한 형식으로 변환합니다."""
        processed_data = []
        
        for _, row in df.iterrows():
            # 질문 형식 구성
            prompt = self.format_prompt(row)
            
            processed_data.append({
                'question_id': _,
                'prompt': prompt,
                'correct_answer': row['answer'],
                'category': row['Category'],
                'human_accuracy': row['Human Accuracy']
            })
            
        return processed_data
        
    def format_prompt(self, row: pd.Series) -> str:
        """LLM을 위한 프롬프트 형식을 구성합니다."""
        return f"""Answer the following multiple choice question about criminal law. 
        Choose the most appropriate answer from options A, B, C, or D.
        
        Question: {row['question']}
        
        Options:
        A) {row['A']}
        B) {row['B']}
        C) {row['C']}
        D) {row['D']}
        
        Provide your answer as a single letter (A, B, C, or D).
        """
    
    def process_and_save(self):
        """전체 데이터 처리 파이프라인을 실행합니다."""
        try:
            # 데이터 로드
            df = self.loader.load_criminal_law_test()
            
            # 데이터 검증
            if not self.loader.validate_data(df):
                raise ValueError("Data validation failed")
            
            # 데이터 전처리
            processed_data = self.preprocess_questions(df)
            
            # 전처리된 데이터를 DataFrame으로 변환
            processed_df = pd.DataFrame(processed_data)
            
            # 저장
            self.loader.save_processed_data(processed_df, 'processed_criminal_law.csv')
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_and_save()