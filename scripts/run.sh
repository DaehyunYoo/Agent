#!/bin/bash

# 스크립트 실패 시 즉시 중단
set -e

echo "Starting KMMLU Agent System Evaluation..."

# 1. 데이터 전처리
echo "Step 1: Preprocessing data..."
python -m agent_system.src.data.processor

# 2. RAG 시스템 초기화
echo "Step 2: Initializing RAG system..."
python -m agent_system.src.agent.rag

# 3. 평가 실행
echo "Step 3: Running evaluation..."
python -m agent_system.src.evaluation.evaluator

# 4. 결과 확인
echo "Step 4: Checking results..."
if [ -f "outputs/evaluation_metrics.csv" ]; then
    echo "Evaluation completed successfully!"
    echo "Results are available in the outputs directory"
else
    echo "Error: Evaluation results not found"
    exit 1
fi