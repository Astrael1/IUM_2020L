import argparse
import csv

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Session purchase probability predictor")
    parser.add_argument('-f', type=str, help='sessions json file')
    args = parser.parse_args()

    if args.f is None:
        print('Give all args!')
    else:
        sessions = pd.read_json(args.f, lines=True)
        grouped_sessions = sessions.groupby('session_id')

        with open('sessions_status.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for name, group in grouped_sessions:
                id_s = group.iloc[-1].session_id
                status = 1 if group.iloc[-1].event_type == 'BUY_PRODUCT' else 0
                writer.writerow([id_s, status])
