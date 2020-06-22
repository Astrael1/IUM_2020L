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
    good = 0
    bad = 0

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
            score = 0
            if sessions_data[key]['status'] == 1:
                score = sessions_data[key]['prediction'] - 0.5
            else:
                score = 0.5 - sessions_data[key]['prediction']
            if score > 0:
                good += 1
            else:
                bad += 1
            scores.append(score)
            print(f'session_id: {key} score: {score}')

    print(f'Good predictions: {good}, bad predictions: {bad}')
    print(f'Mean score {np.mean(scores)}')
