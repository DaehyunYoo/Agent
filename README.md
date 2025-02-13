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
│   ├── raw/                  # 원본 데이터
│   └── processed/           # 전처리된 데이터
└── outputs/                  # 평가 결과 저장
```

## 주요 기능

1. **RAG 시스템**
   - GPT-4o-mini 기반 LLM
   - text-embedding-3-small 기반 임베딩
   - BM25와 임베딩 결합한 하이브리드 검색

2. **데이터 처리**
   - Raw data 자동 로드 및 전처리
   - 데이터 검증 및 포맷팅
   - 전처리된 데이터 저장 및 관리

3. **평가 시스템**
   - OpenAI batch API 활용
   - 자동화된 성능 평가
   - 결과 저장 및 분석

## 설치 및 실행

### 필수 요구사항
- Docker
- Docker Compose
- OpenAI API 키

### 설치 방법

1. 환경 설정
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your-api-key" > .env
```

2. 도커 컨테이너 실행
```bash
# 이미지 빌드 및 실행
docker-compose up --build
```

### 실행 방법

시스템 구축부터 평가까지 모든 과정이 자동화되어 있습니다:
```bash
./scripts/run.sh
```

## 의존성 관리

Poetry를 통한 의존성 관리가 구현되어 있습니다:

```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.11"
openai = ">=1.61.1,<2.0.0"
pandas = ">=2.2.3,<3.0.0"
numpy = "^2.2.2"
rank-bm25 = ">=0.2.2"
python-dotenv = ">=1.0.1,<2.0.0"
```

## 도커 구성

`docker-compose.yml`을 통해 모든 환경 설정이 자동화되어 있습니다:
```yaml
version: '3.8'

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

## 성능

- **현재 정확도**: 0.76
- **평가 데이터셋**: KMMLU Criminal Law 테스트셋
- **평가 결과**: `outputs/` 디렉토리에서 확인 가능
  - evaluation_results.json: 상세 평가 결과
  - evaluation_metrics.csv: 주요 성능 지표
  - batch_api_input/output.jsonl: API 입출력 기록