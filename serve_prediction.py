from model.model import load_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Session purchase probability predictor")
    parser.add_argument('-m', type=str, help="Saved model path")
    parser.add_argument('-p', type=int, help='Products file')
    parser.add_argument('-s', type=int, help='Sessions file')
    parser.add_argument('-u', type=int, help='Users file')
    args = parser.parse_args()
    model = load_model(args.m)

