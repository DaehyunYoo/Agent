# 요구사항별 구현 상세

## 1. 도커 환경 구성
- **파일:** `docker-compose.yml`, `Dockerfile`
```yaml
# docker-compose.yml의 주요 구성
services:
  agent-system:
    build: .
    volumes:
      - .:/app
      - ./data:/app/data
      - ./outputs:/app/outputs:rw
```

## 2. Poetry/Pyproject 의존성 관리
- **파일:** `pyproject.toml`, `poetry.lock`
```toml
# pyproject.toml의 주요 구성
[tool.poetry.dependencies]
python = ">=3.10,<3.11"
openai = ">=1.61.1,<2.0.0"
pandas = ">=2.2.3,<3.0.0"
```

## 3. 원클릭 실행 스크립트
- **파일:** `scripts/run.sh`
- **실행 순서:**
  1. 데이터 처리
  2. RAG 시스템 구축
  3. KMMLU 평가

## 4. RAG 시스템 데이터 처리
- **Raw Data 처리:**
  - **파일:** `agent_system/src/data/loader.py`
  - **클래스:** `KMMLUDataLoader`
  - **주요 함수:**
    - `load_criminal_law_test()`
    - `validate_data()`
    - `get_question_answer_pairs()`

- **데이터 정제:**
  - **파일:** `agent_system/src/data/processor.py`
  - **클래스:** `DataProcessor`
  - **주요 함수:**
    - `preprocess_questions()`
    - `format_prompt()`
    - `process_and_save()`

## 5. RAG 시스템 구현
- **메인 RAG 시스템:**
  - **파일:** `agent_system/src/agent/rag.py`
  - **클래스:** `RAGSystem`
  - **주요 함수:**
    - `initialize()`
    - `generate_answer()`
    - `retrieve_relevant_documents()`

- **임베딩 처리:**
  - **파일:** `agent_system/src/agent/embedding.py`
  - **클래스:** `EmbeddingAgent`
  - **주요 함수:**
    - `create_embedding()`
    - `create_batch_embeddings()`
    - `calculate_similarity()`

- **하이브리드 검색:**
  - **파일:** `agent_system/src/agent/hybrid_retriever.py`
  - **클래스:** `HybridRetriever`
  - **주요 함수:**
    - `prepare_corpus()`
    - `get_bm25_scores()`
    - `combine_scores()`

## 6. OpenAI Batch API 평가
- **평가 시스템:**
  - **파일:** `agent_system/src/evaluation/evaluator.py`
  - **클래스:** `KMMLUEvaluator`
  - **주요 함수:**
    - `evaluate_test_set()`
    - `evaluate_single_question()`
    - `save_batch_api_files()`

- **LLM 인터페이스:**
  - **파일:** `agent_system/src/agent/llm.py`
  - **클래스:** `LLMAgent`
  - **주요 함수:**
    - `generate_answer()`
    - `batch_generate_answers()`
    - `validate_answer()`

## 7. 결과 저장
- **파일:** `agent_system/src/evaluation/evaluator.py`
- **함수:**
  - `save_results()`: 평가 결과 JSON 저장
  - `save_batch_api_files()`: API 입출력 JSONL 파일 생성

## 8. 테스트 구현
- **위치:** `agent_system/tests/`
- **주요 테스트 파일:**
  - `test_embedding.py`
  - `test_evaluator.py`
  - `test_llm.py`
  - `test_loader.py`
  - `test_rag.py`

## 9. 데이터 저장 구조
```
data/
├── raw/
│   └── data_Criminal-Law-test.csv
└── processed/
    └── processed_criminal_law.csv

outputs/
├── evaluation_results.json
├── evaluation_metrics.csv
├── batch_api_input.jsonl
└── batch_api_output.jsonl
```
