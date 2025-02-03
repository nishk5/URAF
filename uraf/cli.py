import argparse
import asyncio
from uraf.evaluate_agents import run_evaluation
from uraf.benchmark_tracker import BenchmarkTracker
from uraf.config_loader import Config

def main():
    parser = argparse.ArgumentParser(description="URAF Command-Line Interface")
    parser.add_argument("--run", action="store_true", help="Run an agent evaluation")
    parser.add_argument("--config", type=str, help="Specify a model-specific config file")
    parser.add_argument("--history", action="store_true", help="Show evaluation history")
    parser.add_argument("--compare", action="store_true", help="Compare model performances")
    parser.add_argument("--export", action="store_true", help="Export results to CSV")

    args = parser.parse_args()
    tracker = BenchmarkTracker()

    if args.config:
        config = Config(config_path=f"{args.config}")  # Load model-specific config
    else:
        config = Config()  # Load default config.yaml

    if args.run:
        asyncio.run(run_evaluation(config))  # ✅ Properly await async function
    elif args.history:
        history = tracker.load_results()
        for record in history:
            print(f"Model: {record['model']}, Agent: {record['agent_type']}, Score: {record['evaluation']['URAF Score']}")
    elif args.compare:
        print("Performance Summary:", tracker.compare_models())
    elif args.export:
        tracker.export_results()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
