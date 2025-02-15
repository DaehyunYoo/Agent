# Criminal Law Agent System

KMMLU의 Criminal Law 카테고리 테스트셋을 평가하는 Agent System입니다. 
GPT-4o-mini와 text-embedding-3-small을 활용한 RAG(Retrieval-Augmented Generation) 시스템을 구현했습니다.

## 시스템 구조

```
├── agent_system/
│   ├── src/
│   │   ├── agent/             # Agent 관련 모듈
│   │   │   ├── embedding.py   # 임베딩 처리
│   │   │   ├── llm.py        # LLM 인터페이스
│   │   │   ├── rag.py        # RAG 시스템
│   │   │   └── hybrid_retriever.py  # 하이브리드 검색
│   │   ├── data/             # 데이터 처리
│   │   │   ├── loader.py     # 데이터 로드
│   │   │   └── processor.py  # 데이터 전처리
│   │   └── evaluation/       # 평가 시스템
│   │       └── evaluator.py  # 성능 평가
│   └── tests/                # 테스트 코드
├── data/                     # 데이터 저장
├── scripts/                  # 실행 스크립트
└── outputs/                  # 평가 결과 저장
```

## 상세 구현 내용
### 1. 컨테이너 구성
- **파일:** `Dockerfile`, `docker-compose.yml`
- **내용:**
  ```yaml
  # docker-compose.yml
  services:
    agent-system:
      build: .
      volumes:
        - .:/app
        - ./data:/app/data
        - ./outputs:/app/outputs:rw
      env_file:
        - .env
      environment:
        - PYTHONPATH=/app
        - OPENAI_API_KEY=${OPENAI_API_KEY}
      command: ./scripts/run.sh
  ```

### 2. Dependency 관리
- **파일:** `pyproject.toml`
```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.11"
openai = ">=1.61.1,<2.0.0"
pandas = ">=2.2.3,<3.0.0"
numpy = "^2.2.2"
rank-bm25 = ">=0.2.2"
python-dotenv = ">=1.0.1,<2.0.0"
```

### 3. 실행 스크립트
- **파일:** `scripts/run.sh`
- **실행 순서:**
  1. outputs 디렉토리 생성
  2. RAG 시스템 구축 및 평가 수행
  3. 결과 확인 및 출력

### 4. RAG 시스템 데이터 처리
- **Raw Data 처리:**
  - **파일:** `agent_system/src/data/loader.py`
  - **클래스:** `KMMLUDataLoader`
  - **주요 함수:**
    - `load_criminal_law_test()`: Criminal Law 테스트 데이터 로드
    - `validate_data()`: 데이터 구조 및 내용 검증

### 5. RAG 시스템 구현
- **메인 RAG 시스템:**
  - **파일:** `agent_system/src/agent/rag.py`
  - **클래스:** `RAGSystem`
  - **주요 함수:**
    - `initialize()`: 시스템 초기화
    - `generate_answer()`: 답변 생성
    - `retrieve_relevant_documents()`: 관련 문서 검색

- **임베딩 처리:**
  - **파일:** `agent_system/src/agent/embedding.py`
  - **클래스:** `EmbeddingAgent`
  - **주요 함수:**
    - `create_embedding()`: 단일 텍스트 임베딩
    - `create_batch_embeddings()`: 일괄 임베딩 생성
    - `calculate_similarity()`: 코사인 유사도 계산

- **하이브리드 검색:**
  - **파일:** `agent_system/src/agent/hybrid_retriever.py`
  - **클래스:** `HybridRetriever`
  - **주요 함수:**
    - `prepare_corpus()`: 검색을 위한 코퍼스 준비
    - `get_bm25_scores()`: BM25 점수 계산
    - `combine_scores()`: 임베딩-BM25 점수 결합

### 6. OpenAI Batch API 평가
- **평가 시스템:**
  - **파일:** `agent_system/src/evaluation/evaluator.py`
  - **클래스:** `KMMLUEvaluator`
  - **주요 함수:**
    - `evaluate_test_set()`: 전체 테스트셋 평가
    - `evaluate_single_question()`: 단일 문항 평가
    - `save_batch_api_files()`: API 입출력 파일 생성

- **LLM 인터페이스:**
  - **파일:** `agent_system/src/agent/llm.py`
  - **클래스:** `LLMAgent`
  - **주요 함수:**
    - `generate_answer()`: GPT-4o-mini 기반 답변 생성
    - `validate_answer()`: 답안 검증

### 7. 결과 저장
- **파일:** `agent_system/src/evaluation/evaluator.py`
- **주요 함수:**
  - `save_results()`: 평가 결과 JSON 저장
  - `save_batch_api_files()`: Batch API 입출력 파일 생성

## 설치 및 실행

### 1. 환경 설정 및 설치
```bash
# 1. API 키 설정
echo "OPENAI_API_KEY=your-api-key" > .env

# 2. 컨테이너 빌드 및 의존성 설치
docker-compose build

# 3. 컨테이너 실행
docker-compose up -d
```

### 2. 실행 스크립트
- **파일:** `scripts/run.sh`
- **실행 단계:**
  1. outputs 디렉토리 생성 (`mkdir -p /app/outputs`)
  2. RAG 시스템 구축 및 평가 실행 (`python -m agent_system.src.evaluation.evaluator`)
     - RAG 시스템 초기화
     - 데이터 로드 및 처리
     - 평가 수행
  3. 결과 확인 및 출력

## 데이터 처리

### Raw Data 처리
- **데이터 위치:** `data/raw/data_Criminal-Law-test.csv`
- **처리 코드:**
  - **파일:** `agent_system/src/data/loader.py`

## 성능 평가 결과

- **현재 정확도**: 0.755 ~ 0.775
- **평가 데이터셋**: KMMLU Criminal Law 테스트셋
- **평가 결과**: `outputs/` 디렉토리에서 확인 가능
  - evaluation_results.json: 상세 평가 결과
  - evaluation_metrics.csv: 주요 성능 지표
  - batch_api_input/output.jsonl: API 입출력 기록

### 1. 최종 성능
| 메트릭 | 값 |
|--------|-----|
| 정확도 | 75 ~ 77.5% |
| 처리 시간 | 4~5분 (API 응답 대기 시간 제외) |

### 2. 단계별 성능 개선
| 개발 단계 | 정확도 | 개선 내용 |
|-----------|--------|-----------|
| 초기 구현 | 38% | 기본 RAG 시스템 구현 |
| 임베딩 로직 개선 | 46% | - 텍스트 전처리 추가<br>- 임베딩 생성 개선<br>- 유사도 계산 로직 강화 |
| RAG 시스템 개선 | 66% | - 문서 검색 로직 강화<br>- 프롬프트 엔지니어링 개선<br>- 답변 검증 시스템 도입 |
| 하이브리드 검색 도입 | 68% | - BM25 알고리즘 통합<br>- 검색 결과 결합 전략 최적화 |
| Comparison Prompting | 76% | - 비교 기반 프롬프트 도입<br>- 맥락 활용도 향상 |

### 3. 추가 시도 및 결과
| 시도한 방법 | 정확도 | 결과 분석 |
|------------|--------|------------|
| 맥락 증강 | 43% | 성능 크게 저하 |
| 답변 검증 시스템 개선 | 76% | 유의미한 변화 없음 |
| Comparison + Role Prompting | 71% | 성능 저하 |
| Cross-encoder Reranking | 76% | 성능 변화 없음, 처리 시간 증가 |
| Faiss 벡터 검색 | 48% | 성능 크게 저하 |