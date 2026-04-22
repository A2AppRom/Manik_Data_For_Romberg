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
python3 train_model_comparison.py > /dev/null 2>&1
if [ -f "model_comparison.csv" ]; then
    if grep -q "SVM (Linear)" model_comparison.csv; then
        ACCURACY=$(grep "SVM (Linear)" model_comparison.csv | cut -d',' -f2)
        echo "  PASS - model_comparison.csv exists, SVM Linear accuracy: $ACCURACY"
        PASS=$((PASS+1))
    else
        echo "  FAIL - model_comparison.csv missing SVM (Linear) row"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - model_comparison.csv not created"
    FAIL=$((FAIL+1))
fi

# Test 2: Verify cv_fold_results.csv has 9 folds for SVM Linear
echo "Test 2: Checking cross-validation fold results..."
if [ -f "cv_fold_results.csv" ]; then
    FOLD_COUNT=$(grep "SVM (Linear)" cv_fold_results.csv | wc -l | tr -d ' ')
    if [ "$FOLD_COUNT" -eq 9 ]; then
        echo "  PASS - cv_fold_results.csv has $FOLD_COUNT folds for SVM Linear"
        PASS=$((PASS+1))
    else
        echo "  FAIL - expected 9 folds, got $FOLD_COUNT"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - cv_fold_results.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 3: Run train_final_model.py
echo "Test 3: Running train_final_model.py..."
python3 train_final_model.py > /dev/null 2>&1
if [ -f "romberg_model_weights.json" ]; then
    echo "  PASS - romberg_model_weights.json created"
    PASS=$((PASS+1))
else
    echo "  FAIL - romberg_model_weights.json not created"
    FAIL=$((FAIL+1))
fi

# Test 4: Verify model metadata
echo "Test 4: Checking model metadata..."
SAMPLES=$(python3 -c "import json; m=json.load(open('romberg_model_weights.json')); print(m['metadata']['training_samples'])")
SUBJECTS=$(python3 -c "import json; m=json.load(open('romberg_model_weights.json')); print(m['metadata']['training_subjects'])")
N_FEATURES=$(python3 -c "import json; m=json.load(open('romberg_model_weights.json')); print(m['metadata']['n_features'])")

if [ "$SAMPLES" -eq 142 ] && [ "$SUBJECTS" -eq 9 ] && [ "$N_FEATURES" -eq 6 ]; then
    echo "  PASS - training_samples=$SAMPLES, training_subjects=$SUBJECTS, n_features=$N_FEATURES"
    PASS=$((PASS+1))
else
    echo "  FAIL - expected 142 samples, 9 subjects, 6 features; got $SAMPLES, $SUBJECTS, $N_FEATURES"
    FAIL=$((FAIL+1))
fi

# Test 5: Verify weight vector has 6 entries
echo "Test 5: Checking weight vector..."
W_COUNT=$(python3 -c "import json; m=json.load(open('romberg_model_weights.json')); print(len(m['weights']))")
if [ "$W_COUNT" -eq 6 ]; then
    echo "  PASS - weight vector has $W_COUNT entries"
    PASS=$((PASS+1))
else
    echo "  FAIL - weight vector has $W_COUNT entries (expected 6)"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
