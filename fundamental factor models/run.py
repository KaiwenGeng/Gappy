import argparse
from experiment import Experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_of_interest", nargs="+", required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_factors", type=int, required=True)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0)


    args = parser.parse_args()
    print("start experiment")
    experiment = Experiment(args)
    experiment.run()

# python run.py --assets_of_interest 600096 600428 300015 --start_date 20000101 --end_date 20220523 --target_col "daily excess return" --batch_size 8 --num_workers 4 --epochs 10 --num_factors 104
