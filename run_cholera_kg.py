from app.data.pipeline import DataPipeline

if __name__ == "__main__":
    pipeline = DataPipeline()
    # Build knowledge graph for cholera
    kg = pipeline.run_pipeline(
        query="cholera",
        max_diseases=1,
        max_articles_per_disease=5,
        days_back=3650,  # 10 years for more articles
        force_refresh=True
    )
    print("Cholera knowledge graph built and saved.") 