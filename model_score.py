import argparse
import csv
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model score")
    parser.add_argument('-p', type=str, help='Model predictions')
    parser.add_argument('-s', type=str, help='Sessions status')
    args = parser.parse_args()

    sessions_data = {}
    scores = []

    if args.p is None or args.s is None:
        print('Give all args!')
    else:
        with open(args.p) as predictions_file:
            csv_reader = csv.reader(predictions_file, delimiter=',')
            for row in csv_reader:
                sessions_data[row[0]] = {'prediction': float(row[1])}

        with open(args.s) as status_file:
            csv_reader = csv.reader(status_file, delimiter=',')
            for row in csv_reader:
                sessions_data[row[0]]['status'] = float(row[1])

        for key in sessions_data.keys():
            if sessions_data[key]['status'] == 1:
                scores.append(sessions_data[key]['prediction'] - 0.5)
            else:
                scores.append(0.5 - sessions_data[key]['prediction'])

    print(np.mean(scores))
