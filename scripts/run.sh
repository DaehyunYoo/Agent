#!/bin/bash

# 스크립트 실패 시 즉시 중단
set -e

echo "Starting KMMLU Agent System Evaluation..."

# 1. outputs 디렉토리 생성
echo "Creating outputs directory..."
mkdir -p /app/outputs

# 2. 데이터 전처리
echo "Step 1: Preprocessing data..."
python -m agent_system.src.data.processor

# 3. RAG 시스템 초기화
echo "Step 2: Initializing RAG system..."
python -m agent_system.src.agent.rag

# 4. 평가 실행
echo "Step 3: Running evaluation..."
python -m agent_system.src.evaluation.evaluator

# 5. 결과 확인
echo "Step 4: Checking results..."
if [ -f "/app/outputs/evaluation_metrics.csv" ]; then
    echo "Evaluation completed successfully!"
    echo "Results files generated:"
    echo "- evaluation_results.json"
    echo "- evaluation_metrics.csv"
    echo "- batch_api_input.jsonl"
    echo "- batch_api_output.jsonl"
    
    # 결과 요약 출력
    echo -e "\nEvaluation Summary:"
    cat /app/outputs/evaluation_metrics.csv
else
    echo "Creating new evaluation results..."
fi