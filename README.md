# 뤼튼테크놀로지스 AI팀 인턴 - 과제 전형

KMMLU의 Criminal Law 카테고리 테스트셋을 평가하는 Agent System입니다. 
GPT-4o-mini와 text-embedding-3-small을 활용한 RAG 시스템을 구현했습니다.

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

## 구현 내용
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
openai = ">=1.0.0"
```

### 3. Agent System 실행 스크립트
- **파일:** `scripts/run.sh`
- **실행 순서:**
  1. outputs 디렉토리 생성
  2. RAG 시스템 구축 및 평가 수행
  3. 결과 확인 및 출력

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
- **실행:**
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

- **현재 정확도**: 45.5 %
- **평가 데이터셋**: KMMLU Criminal Law 테스트셋
- **평가 결과**: `outputs/` 디렉토리에서 확인 가능
  - evaluation_results.json: 상세 평가 결과
  - evaluation_metrics.csv: 주요 성능 지표
  - batch_api_input/output.jsonl: API 입출력 기록

### 1. 최종 성능
| 메트릭 | 값 |
|--------|-----|
| Accuracy | 45.5% |
| F1 score | 44.62 |
| 처리 시간 | 54.20분 |

### 2. 주요 개선 사항
1. **하이브리드 검색 구현**: 
   - `hybrid_retriever.py`의 `combine_scores()` 함수에서 임베딩 기반 검색과 BM25 검색 결과 결합
   - `rag.py`의 `retrieve_relevant_documents()` 함수에서 교집합 기반 결과 선택 로직 구현

2. **도메인 특화 임베딩 & 컨텍스트 강화**: 
   - `embedding.py`의 `_enhance_query()` 함수에서 법률 도메인 컨텍스트 추가
   - `embedding.py`의 `_preprocess_text()` 함수에서 법률 용어 특화 전처리 적용

3. **최적화된 프롬프트 구성**: 
   - `rag.py`의 `_construct_prompt()` 함수에서 전문가 시스템 역할 정의와 명확한 지시사항 포함
   - `llm.py`의 시스템 프롬프트 설정

4. **비동기 배치 처리**: 
   - `llm.py`의 `generate_batch_answers_async()` 함수에서 효율적인 배치 API 처리
   - `evaluator.py`의 `evaluate_test_set_async()` 함수에서 비동기 평가 로직 구현

5. **답변 검증 및 추출 메커니즘**: 
   - `rag.py`의 `validate_answer()` 함수에서 유사도 기반 검증 로직
   - `llm.py`의 `extract_answer_letter()` 함수에서 답안 문자 추출 최적화