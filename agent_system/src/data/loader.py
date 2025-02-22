import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMMLUDataLoader:
    """KMMLU 데이터셋을 로드"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir (str, optional): 데이터 디렉토리 경로. 
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parents[3] / 'data'
        
    def load_criminal_law_test(self) -> pd.DataFrame:
        """Criminal Law 테스트 데이터셋 로드"""
        test_path = self.data_dir / 'data_Criminal-Law-test.csv'
        
        try:
            df = pd.read_csv(test_path)
            logger.info(f"Successfully loaded {len(df)} test samples from {test_path}")
            return df
        except FileNotFoundError:
            logger.error(f"Test dataset not found at {test_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading test dataset: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """데이터프레임의 구조와 내용 검증"""
        required_columns = ['question', 'answer', 'A', 'B', 'C', 'D', 
                          'Category', 'Human Accuracy']
        
        # 필수 컬럼 존재 확인
        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required columns in dataset")
            return False
            
        # 데이터 타입 검증
        try:
            assert df['answer'].dtype in ['int64', 'int32'], "Answer column should be integer"
            assert df['Human Accuracy'].dtype in ['float64', 'float32'], "Human Accuracy should be float"
            assert all(df['answer'].between(1, 4)), "Answer should be between 1 and 4"
            assert all(df['Human Accuracy'].between(0, 1)), "Human Accuracy should be between 0 and 1"
        except AssertionError as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
            
        return True
