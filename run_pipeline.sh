#!/usr/bin/env bash
# Runs the full EduPro data pipeline end-to-end, in order.
# Usage: bash run_pipeline.sh
set -e

echo "==> [1/6] Generating synthetic dataset..."
python src/01_generate_data.py

echo "==> [2/6] Engineering learner-level features..."
python src/02_feature_engineering.py

echo "==> [3/6] Running clustering (K-Means + Hierarchical validation)..."
python src/03_clustering.py

echo "==> [4/6] Building & evaluating recommendation engine..."
python src/04_recommendation_engine.py

echo "==> [5/6] Computing evaluation metrics..."
python src/05_evaluation.py

echo "==> [6/6] Generating EDA visuals..."
python src/06_eda_visuals.py

echo ""
echo "Pipeline complete. Launch the dashboard with:"
echo "    streamlit run app.py"
