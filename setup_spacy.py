#!/usr/bin/env python3
"""
Setup script to install required spaCy language models.
Run this after installing the requirements.txt dependencies.
"""

import subprocess
import sys

def install_spacy_models():
    """Install required spaCy language models."""
    models = [
        "en_core_web_sm",
        "en_core_sci_sm",  # Scientific/medical model
    ]
    
    for model in models:
        print(f"Installing spaCy model: {model}")
        try:
            subprocess.run([
                sys.executable, "-m", "spacy", "download", model
            ], check=True)
            print(f"✓ Successfully installed {model}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {model}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("Installing spaCy language models...")
    if install_spacy_models():
        print("✓ All spaCy models installed successfully!")
    else:
        print("✗ Some models failed to install. Please check the errors above.")
        sys.exit(1) 