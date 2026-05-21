#!/bin/bash
# Test the website content and predictions
# Usage: bash tests/test_website.sh

cd "$(dirname "$0")/.."

PASS=0
FAIL=0

echo "============================================"
echo "  Romberger Website Tests"
echo "============================================"
echo ""

# Test 1: No MotionSense references in HTML
echo "Test 1: Checking for MotionSense references..."
MS_COUNT=$(grep -rci "MotionSense\|standing-color\|sitting-color\|posture classification\|badge-standing\|badge-sitting" ui/index.html ui/learn.html 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
if [ "$MS_COUNT" -eq 0 ]; then
    echo "  PASS - zero MotionSense/standing/sitting references found"
    PASS=$((PASS+1))
else
    echo "  FAIL - found $MS_COUNT old references"
    FAIL=$((FAIL+1))
fi

# Test 2: Verify KNN model is referenced in index.html
echo "Test 2: Checking model type in index.html..."
if grep -q "KNN" ui/index.html && grep -q "k=7\|k: 7" ui/index.html; then
    echo "  PASS - KNN (k=7) model referenced in index.html"
    PASS=$((PASS+1))
else
    echo "  FAIL - KNN model reference not found in index.html"
    FAIL=$((FAIL+1))
fi

# Test 3: Verify 22 subjects referenced
echo "Test 3: Checking subject count in HTML..."
if grep -q "22 subjects\|22 unique" ui/index.html; then
    echo "  PASS - index.html references 22 subjects"
    PASS=$((PASS+1))
else
    echo "  FAIL - index.html does not reference 22 subjects"
    FAIL=$((FAIL+1))
fi

# Test 4: Verify Sophia not in per-subject chart data
echo "Test 4: Checking Sophia is not in chart data..."
if grep -q "Sophia" ui/index.html; then
    echo "  FAIL - Sophia still referenced in index.html"
    FAIL=$((FAIL+1))
else
    echo "  PASS - Sophia correctly removed from index.html"
    PASS=$((PASS+1))
fi

# Test 5: Start localhost and verify page loads
echo "Test 5: Testing localhost..."
python3 -m http.server 8000 --directory ui &>/dev/null &
SERVER_PID=$!
sleep 2

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null || echo "000")
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "  PASS - localhost:8000 returned HTTP 200"
    PASS=$((PASS+1))
else
    echo "  FAIL - localhost:8000 returned HTTP $HTTP_CODE (expected 200)"
    FAIL=$((FAIL+1))
fi

# Test 6: Verify learn.html exists and has content
echo "Test 6: Checking learn.html..."
if [ -f "ui/learn.html" ] && grep -q "Romberger" ui/learn.html; then
    echo "  PASS - learn.html exists with Romberger content"
    PASS=$((PASS+1))
else
    echo "  FAIL - learn.html missing or empty"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
