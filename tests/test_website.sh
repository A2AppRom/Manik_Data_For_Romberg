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
MS_COUNT=$(grep -ci "MotionSense\|standing-color\|sitting-color\|posture classification\|badge-standing\|badge-sitting" index.html learn.html 2>/dev/null || echo "0")
if [ "$MS_COUNT" -eq 0 ]; then
    echo "  PASS - zero MotionSense/standing/sitting references found"
    PASS=$((PASS+1))
else
    echo "  FAIL - found $MS_COUNT old references"
    FAIL=$((FAIL+1))
fi

# Test 2: Verify model weights are embedded in index.html
echo "Test 2: Checking model weights in index.html..."
if grep -q "\-1.630074510121072" index.html && grep -q "\-0.21889522234118264" index.html; then
    echo "  PASS - current SVM Linear weights found in index.html"
    PASS=$((PASS+1))
else
    echo "  FAIL - expected model weights not found in index.html"
    FAIL=$((FAIL+1))
fi

# Test 3: Verify 9 subjects referenced (not 10)
echo "Test 3: Checking subject count in HTML..."
if grep -q "9 subjects" index.html || grep -q '"9"' index.html; then
    echo "  PASS - index.html references 9 subjects"
    PASS=$((PASS+1))
else
    echo "  FAIL - index.html does not reference 9 subjects"
    FAIL=$((FAIL+1))
fi

# Test 4: Verify Sophia not in per-subject chart data
echo "Test 4: Checking Sophia is not in chart data..."
if grep -q "Sophia" index.html; then
    echo "  FAIL - Sophia still referenced in index.html"
    FAIL=$((FAIL+1))
else
    echo "  PASS - Sophia correctly removed from index.html"
    PASS=$((PASS+1))
fi

# Test 5: Start localhost and verify page loads
echo "Test 5: Testing localhost..."
python3 -m http.server 8000 &>/dev/null &
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

# Test 6: Verify learn.html has updated stats
echo "Test 6: Checking learn.html stats..."
if grep -q "9 subjects" learn.html && grep -q "142" learn.html; then
    echo "  PASS - learn.html references 9 subjects and 142 samples"
    PASS=$((PASS+1))
else
    echo "  FAIL - learn.html has outdated stats"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
