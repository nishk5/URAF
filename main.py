from uraf.core import URAF

# Initialize URAF with any LLM backend
uraf_pipeline = URAF(model="qwen2.5-7b-instruct-1m", api_url="http://localhost:1234/v1/completions")

# Run a reasoning evaluation task
result = uraf_pipeline.run("Explain the impact of inflation on global markets.")

# Display structured response & evaluation
print("Structured Response:", result["processed_response"])
print("Evaluation:", result["evaluation_result"])