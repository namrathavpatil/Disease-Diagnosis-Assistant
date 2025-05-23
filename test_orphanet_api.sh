#!/bin/bash

# Make sure the ORPHANET_API_KEY is set
if [[ -z "${ORPHANET_API_KEY}" ]]; then
  echo "❌ ORPHANET_API_KEY not set. Use: export ORPHANET_API_KEY=your_key_here"
  exit 1
fi

# Base API URL
BASE_URL="https://api.orphadata.com"

# Helper function for pretty output
function test_endpoint() {
  local description="$1"
  local url="$2"
  echo -e "\n➡️  $description"
  curl -s -X GET "$url" \
    -H "Accept: application/json" \
    -H "Ocp-Apim-Subscription-Key: ${ORPHANET_API_KEY}" \
    | jq . || echo "⚠️ Failed to parse response"
  sleep 1
}

echo "🔍 Testing Orphanet API endpoints..."

# Test 1: Search by disease name (Marfan syndrome)
test_endpoint "Test 1: Search by disease name (Marfan syndrome)" "$BASE_URL/rd-cross-referencing/orphacodes/names/marfan%20syndrome?lang=en"

# Test 2: Search by ICD-11 code (1A00 for Cholera)
test_endpoint "Test 2: Search by ICD-11 code (1A00 for Cholera)" "$BASE_URL/rd-cross-referencing/icd-11s/1A00?lang=en"

# Test 3: Search by ICD-11 code (4A44.5 for Kawasaki disease)
test_endpoint "Test 3: Search by ICD-11 code (4A44.5 for Kawasaki disease)" "$BASE_URL/rd-cross-referencing/icd-11s/4A44.5?lang=en"

# Test 4: Search by OMIM code (154700 for Marfan syndrome)
test_endpoint "Test 4: Search by OMIM code (154700)" "$BASE_URL/rd-cross-referencing/omim-codes/154700?lang=en"

# Test 5: Get phenotypes for Marfan syndrome (ORPHA:558)
test_endpoint "Test 5: Get phenotypes for Marfan syndrome (ORPHA:558)" "$BASE_URL/rd-phenotypes/orphacodes/558?lang=en"

# Test 6: Get medical specialties for Marfan syndrome (ORPHA:558)
test_endpoint "Test 6: Get medical specialties for Marfan syndrome (ORPHA:558)" "$BASE_URL/rd-medical-specialties/orphacodes/558?lang=en" 