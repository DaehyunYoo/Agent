#!/bin/bash

# 스크립트 실패 시 중단
set -e

echo "Starting KMMLU Agent System Evaluation..."

# outputs 디렉토리 생성
echo "Creating outputs directory..."
mkdir -p /app/outputs

# 평가 실행
echo "Running evaluation..."
python -m agent_system.src.evaluation.evaluator

# 결과 확인
echo "Checking results..."
if [ -f "/app/outputs/evaluation_metrics.csv" ]; then
    echo "Evaluation completed"
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