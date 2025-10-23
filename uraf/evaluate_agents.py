import asyncio

from loguru import logger

from uraf.benchmark import Benchmark
from uraf.benchmark_tracker import BenchmarkTracker
from uraf.config_loader import Config
from uraf.evaluator import LLMResponseEvaluator
from uraf.llm_client import LLMClient
from uraf.prompt_manager import PromptManager


async def run_evaluation(config):
    """Runs the evaluation pipeline asynchronously with proper tracking."""

    # Initialize Components
    llm_settings = config.get_llm_settings()
    llm = LLMClient(model=llm_settings["model"], api_url=llm_settings["api_url"])
    evaluator = LLMResponseEvaluator()
    tracker = BenchmarkTracker()

    # Select an agent type
    agent_types = list(Benchmark.get_agent_types())

    print("\n🔹 Available Agent Types:")
    for idx, agent in enumerate(agent_types, 1):
        print(f"{idx}. {agent}")

    while True:
        try:
            agent_choice = int(input("\n➡️ Select an agent type (number): ")) - 1
            if 0 <= agent_choice < len(agent_types):
                agent_type = agent_types[agent_choice]
                break
            else:
                print("❌ Invalid selection. Please choose a valid number.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    # 🔹 Generate a **dynamic benchmark question using LLM**
    benchmark_question = Benchmark.generate(agent_type)

    # 🔹 Apply structured reasoning techniques
    technique = Benchmark.get_technique_for_agent(agent_type)
    formatted_prompt = PromptManager.get_prompt_with_technique(benchmark_question, technique)

    # 🔹 Query the LLM
    response = await llm.query(formatted_prompt)
    logger.info(f"🔍 Raw LLM Response: {response}")

    # 🔹 Evaluate response
    evaluation = await evaluator.evaluate_response(response["summary"])

    # 🔹 Store results in BenchmarkTracker
    tracker.save_result(llm_settings["model"], agent_type, evaluation)

    print(f"\n✅ Benchmark Question: {benchmark_question}\n🔍 Evaluation: {evaluation}\n")


if __name__ == "__main__":
    config = Config()
    asyncio.run(run_evaluation(config))
