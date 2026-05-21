#!/bin/bash
# Test model training and weight export
# Usage: bash tests/test_model.sh

set -e
cd "$(dirname "$0")/.."

PASS=0
FAIL=0

echo "============================================"
echo "  Romberger Model Training Tests"
echo "============================================"
echo ""

# Test 1: Run train_model_comparison.py
echo "Test 1: Running train_model_comparison.py..."
python3 pipeline/train_model_comparison.py > /dev/null 2>&1
if [ -f "results/model_comparison.csv" ]; then
    if grep -q "KNN (k=7)" results/model_comparison.csv; then
        ACCURACY=$(grep "KNN (k=7)" results/model_comparison.csv | cut -d',' -f2)
        echo "  PASS - model_comparison.csv exists, KNN (k=7) accuracy: $ACCURACY"
        PASS=$((PASS+1))
    else
        echo "  FAIL - model_comparison.csv missing KNN (k=7) row"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - results/model_comparison.csv not created"
    FAIL=$((FAIL+1))
fi

# Test 2: Verify cv_fold_results.csv has folds for KNN
echo "Test 2: Checking cross-validation fold results..."
if [ -f "results/cv_fold_results.csv" ]; then
    FOLD_COUNT=$(grep "KNN (k=7)" results/cv_fold_results.csv | wc -l | tr -d ' ')
    if [ "$FOLD_COUNT" -ge 20 ]; then
        echo "  PASS - cv_fold_results.csv has $FOLD_COUNT folds for KNN (k=7)"
        PASS=$((PASS+1))
    else
        echo "  FAIL - expected >=20 folds, got $FOLD_COUNT"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - results/cv_fold_results.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 3: Run train_final_model.py
echo "Test 3: Running train_final_model.py..."
python3 pipeline/train_final_model.py > /dev/null 2>&1
if [ -f "model/romberg_model_weights.json" ]; then
    echo "  PASS - model/romberg_model_weights.json created"
    PASS=$((PASS+1))
else
    echo "  FAIL - model/romberg_model_weights.json not created"
    FAIL=$((FAIL+1))
fi

# Test 4: Verify model metadata
echo "Test 4: Checking model metadata..."
SAMPLES=$(python3 -c "import json; m=json.load(open('model/romberg_model_weights.json')); print(m['metadata']['training_samples'])")
SUBJECTS=$(python3 -c "import json; m=json.load(open('model/romberg_model_weights.json')); print(m['metadata']['training_subjects'])")
N_FEATURES=$(python3 -c "import json; m=json.load(open('model/romberg_model_weights.json')); print(m['metadata']['n_features'])")

if [ "$SAMPLES" -eq 111 ] && [ "$SUBJECTS" -eq 22 ] && [ "$N_FEATURES" -eq 12 ]; then
    echo "  PASS - training_samples=$SAMPLES, training_subjects=$SUBJECTS, n_features=$N_FEATURES"
    PASS=$((PASS+1))
else
    echo "  FAIL - expected 111 samples, 22 subjects, 12 features; got $SAMPLES, $SUBJECTS, $N_FEATURES"
    FAIL=$((FAIL+1))
fi

# Test 5: Verify training data has correct count
echo "Test 5: Checking training data points..."
TP_COUNT=$(python3 -c "import json; m=json.load(open('model/romberg_model_weights.json')); print(len(m['training_data']))")
if [ "$TP_COUNT" -eq 111 ]; then
    echo "  PASS - training_data has $TP_COUNT points"
    PASS=$((PASS+1))
else
    echo "  FAIL - training_data has $TP_COUNT points (expected 111)"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
