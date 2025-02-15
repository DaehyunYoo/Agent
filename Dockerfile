FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치 
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치 
RUN pip install poetry==1.7.1

# 프로젝트 의존성 파일 복사
COPY pyproject.toml poetry.lock ./

# Poetry 설정 및 의존성 설치
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction

# rank_bm25 설치
RUN pip install rank_bm25

# 소스 코드 복사
COPY . .

# Poetry로 프로젝트 설치
RUN poetry install

# 실행 권한 부여
RUN chmod +x scripts/run.sh

# 환경 변수 설정
ENV PYTHONPATH=/app

# 실행 명령
CMD ["./scripts/run.sh"]