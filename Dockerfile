FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 필수 패키지 개별 설치 (버전 제한 완화)
RUN pip install --no-cache-dir \
    openai \
    pandas \
    numpy \
    rank-bm25 \
    python-dotenv \
    nltk \
    scikit-learn

# 소스 코드 복사
COPY . .

# 실행 스크립트 권한 설정
RUN chmod +x scripts/run.sh

# 환경 변수 설정
ENV PYTHONPATH=/app

# 실행 명령
CMD ["./scripts/run.sh"]