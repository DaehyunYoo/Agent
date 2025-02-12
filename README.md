# KMMLU Agent System

형사법(Criminal Law) 질문에 대한 응답을 생성하는 Agent System입니다.

## 프로젝트 구조
Agent/
├── agent_system/         # 메인 패키지
│   ├── src/
│   │   ├── agent/       # Agent 관련 모듈
│   │   ├── data/        # 데이터 처리 모듈
│   │   └── evaluation/  # 평가 관련 모듈
│   └── tests/           # 테스트 코드
├── data/                # 데이터 디렉토리
├── outputs/             # 결과 저장 디렉토리
└── scripts/             # 실행 스크립트

## 설치 및 실행 가이드

1. 패키지 설치
```bash
# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -
# 의존성 설치
poetry install
```

2. 컨테이너 셋업
```bash
# 이미지 빌드
docker-compose build

# 컨테이너 실행
docker-compose up -d
```

3. Agent System 구축 및 평가
```bash
# 전체 시스템 실행(데이터 전처리부터 평가까지)
./scripts/run.sh

# 개별 실행
python -m agent_system.src.data.processor # 데이터 처리
python -m agent_system.src.agent.rag # RAG 시스템 구축
python -m agent_system.src.evaluation.evaluator # 평가

```

### Poetry/Pyproject 사용
pyproject.toml 파일을 통해 패키지 관리를 하며, Poetry를 사용하여 의존성을 관리합니다.


## 성능 결과


## 주요 기능

GPT-4o-mini 기반 LLM
text-embedding-3-small 임베딩 모델
RAG(Retrieval-Augmented Generation) 시스템
배치 처리 지원
