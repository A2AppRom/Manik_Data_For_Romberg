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
python3 pipeline/consolidate_data.py > /dev/null 2>&1
if [ -f "data/raw/manifest.csv" ]; then
    echo "  PASS - manifest.csv created"
    PASS=$((PASS+1))
else
    echo "  FAIL - manifest.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 2: Run clean_data.py
echo "Test 2: Running clean_data.py..."
python3 pipeline/clean_data.py > /dev/null 2>&1
CLEAN_LOG="data/cleaned/cleaning_log.csv"
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
python3 pipeline/chunk_data.py > /dev/null 2>&1
CHUNK_COUNT=$(find data/final -name "*.csv" -not -name "manifest.csv" | wc -l | tr -d ' ')
if [ "$CHUNK_COUNT" -gt 100 ]; then
    echo "  PASS - $CHUNK_COUNT chunk files created in data/final/"
    PASS=$((PASS+1))
else
    echo "  FAIL - only $CHUNK_COUNT chunk files (expected >100)"
    FAIL=$((FAIL+1))
fi

# Test 4: Run extract_features.py
echo "Test 4: Running extract_features.py..."
python3 pipeline/extract_features.py > /dev/null 2>&1
if [ -f "results/features_dataset.csv" ]; then
    FEAT_LINES=$(wc -l < "results/features_dataset.csv" | tr -d ' ')
    if [ "$FEAT_LINES" -gt 100 ]; then
        echo "  PASS - features_dataset.csv has $FEAT_LINES lines"
        PASS=$((PASS+1))
    else
        echo "  FAIL - features_dataset.csv has $FEAT_LINES lines (expected >100)"
        FAIL=$((FAIL+1))
    fi
else
    echo "  FAIL - features_dataset.csv not found"
    FAIL=$((FAIL+1))
fi

# Test 5: Verify correct subject count (22 subjects after Sophia exclusion + renumbering)
echo "Test 5: Verifying subject count..."
SUBJ_COUNT=$(cut -d',' -f1 results/features_dataset.csv | tail -n +2 | sort -u | wc -l | tr -d ' ')
if [ "$SUBJ_COUNT" -eq 22 ]; then
    echo "  PASS - 22 unique subjects in features_dataset.csv"
    PASS=$((PASS+1))
else
    echo "  FAIL - expected 22 subjects, got $SUBJ_COUNT"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
