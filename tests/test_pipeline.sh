#!/bin/bash
# Test the data pipeline (consolidation, cleaning, chunking, feature extraction)
# Usage: bash tests/test_pipeline.sh

set -e
cd "$(dirname "$0")/.."

PASS=0
FAIL=0

echo "============================================"
echo "  Romberger Data Pipeline Tests"
echo "============================================"
echo ""

# Test 1: Run consolidate_data.py
echo "Test 1: Running consolidate_data.py..."
python3 consolidate_data.py > /dev/null 2>&1
if [ -f "romberg_data/manifest.csv" ]; then
    echo "  PASS - manifest.csv created"
    PASS=$((PASS+1))
else
    echo "  FAIL - manifest.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 2: Run clean_data.py
echo "Test 2: Running clean_data.py..."
python3 clean_data.py > /dev/null 2>&1
CLEAN_LOG="romberg_data_cleaned/cleaning_log.csv"
if [ -f "$CLEAN_LOG" ]; then
    LINE_COUNT=$(wc -l < "$CLEAN_LOG" | tr -d ' ')
    if [ "$LINE_COUNT" -gt 50 ]; then
        echo "  PASS - cleaning_log.csv has $LINE_COUNT lines"
        PASS=$((PASS+1))
    else
        echo "  FAIL - cleaning_log.csv only has $LINE_COUNT lines (expected >50)"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - cleaning_log.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 3: Run chunk_data.py
echo "Test 3: Running chunk_data.py..."
python3 chunk_data.py > /dev/null 2>&1
CHUNK_COUNT=$(find romberg_data_final -name "*.csv" -not -name "manifest.csv" | wc -l | tr -d ' ')
if [ "$CHUNK_COUNT" -gt 200 ]; then
    echo "  PASS - $CHUNK_COUNT chunk files created in romberg_data_final/"
    PASS=$((PASS+1))
else
    echo "  FAIL - only $CHUNK_COUNT chunk files (expected >200)"
    FAIL=$((FAIL+1))
fi

# Test 4: Run extract_features.py
echo "Test 4: Running extract_features.py..."
python3 extract_features.py > /dev/null 2>&1
if [ -f "features_dataset.csv" ]; then
    FEAT_LINES=$(wc -l < "features_dataset.csv" | tr -d ' ')
    if [ "$FEAT_LINES" -eq 143 ]; then
        echo "  PASS - features_dataset.csv has $FEAT_LINES lines (142 samples + header)"
        PASS=$((PASS+1))
    else
        echo "  FAIL - features_dataset.csv has $FEAT_LINES lines (expected 143)"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - features_dataset.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 5: Verify no subject_08 (Sophia's corrupted data)
echo "Test 5: Verifying Sophia (subject_08) is excluded..."
if [ ! -d "romberg_data_final/subject_08" ]; then
    echo "  PASS - subject_08 correctly excluded from final data"
    PASS=$((PASS+1))
else
    echo "  FAIL - subject_08 still exists in romberg_data_final/"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
