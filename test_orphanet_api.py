import os
import json
from app.data.orphanet_collector import OrphanetCollector
from app.config import settings

def test_orphanet_api():
    """Test the Orphanet API endpoints."""
    # Initialize collector
    collector = OrphanetCollector(api_key=settings.orphanet_api_key)
    
    # Test ICD-11 search (Cholera)
    print("\nTesting ICD-11 search for Cholera (1A00)...")
    cholera_diseases = collector.search_by_icd11("1A00")
    print(f"Found {len(cholera_diseases)} diseases")
    if cholera_diseases:
        print("First disease details:")
        print(json.dumps(cholera_diseases[0], indent=2))
    
    # Test disease phenotypes
    if cholera_diseases:
        orpha_code = cholera_diseases[0]["orpha_code"]
        print(f"\nTesting phenotypes for disease {orpha_code}...")
        phenotypes = collector.get_disease_phenotypes(orpha_code)
        print(f"Found {len(phenotypes)} phenotypes")
        if phenotypes:
            print("First phenotype details:")
            print(json.dumps(phenotypes[0], indent=2))
    
    # Test disease specialties
    if cholera_diseases:
        print(f"\nTesting specialties for disease {orpha_code}...")
        specialties = collector.get_disease_specialties(orpha_code)
        print(f"Found {len(specialties)} specialties")
        if specialties:
            print("First specialty details:")
            print(json.dumps(specialties[0], indent=2))

if __name__ == "__main__":
    test_orphanet_api() 