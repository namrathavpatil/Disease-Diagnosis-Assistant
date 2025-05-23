#!/bin/bash

# Check if ORPHANET_API_KEY is set
if [ -z "$ORPHANET_API_KEY" ]; then
    echo "Error: ORPHANET_API_KEY environment variable is not set"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base API URL
BASE_URL="https://api.orphadata.com"

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    
    echo -e "\n${GREEN}Testing $name...${NC}"
    echo "URL: $url"
    
    response=$(curl -s -w "\n%{http_code}" \
        -H "Accept: application/json" \
        -H "Ocp-Apim-Subscription-Key: ${ORPHANET_API_KEY}" \
        "$url")
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" -eq 200 ]; then
        echo -e "${GREEN}Success! Status code: $status_code${NC}"
        echo "Response preview:"
        echo "$body" | head -n 5
    else
        echo -e "${RED}Failed! Status code: $status_code${NC}"
        echo "Error response:"
        echo "$body"
    fi
}

# Test Orphanet API endpoints
echo -e "\n${GREEN}Testing Orphanet API endpoints...${NC}"

# 1. Search by ICD-11 code (Cholera)
test_endpoint "Orphanet ICD-11 Search (Cholera)" \
    "$BASE_URL/rd-cross-referencing/icd-11s/1A00?lang=en"

# 2. Search by ICD-11 code (Kawasaki disease)
test_endpoint "Orphanet ICD-11 Search (Kawasaki)" \
    "$BASE_URL/rd-cross-referencing/icd-11s/4A44.5?lang=en"

# 3. Get disease phenotypes (Marfan syndrome)
test_endpoint "Orphanet Disease Phenotypes" \
    "$BASE_URL/rd-phenotypes/orphacodes/558?lang=en"

# 4. Get disease specialties (Marfan syndrome)
test_endpoint "Orphanet Disease Specialties" \
    "$BASE_URL/rd-medical-specialties/orphacodes/558?lang=en"

# Test PubMed API endpoints
echo -e "\n${GREEN}Testing PubMed API endpoints...${NC}"

# 1. Search for PMIDs
test_endpoint "PubMed Search" \
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cholera%20treatment&retmode=json&retmax=10&tool=RAG_Medical_Assistant&email=nvpatil@usc.edu"

# 2. Fetch article details (using a known PMID)
test_endpoint "PubMed Article Details" \
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=12345678&retmode=xml&tool=RAG_Medical_Assistant&email=nvpatil@usc.edu" 