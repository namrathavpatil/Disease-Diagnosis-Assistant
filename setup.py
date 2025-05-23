from setuptools import setup, find_packages

setup(
    name="medical-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pydantic",
        "python-dotenv",
        "networkx",
        "biopython",
    ],
    python_requires=">=3.8",
) 