# IUM_2020L
https://docs.google.com/document/d/1LPkhbCjRHQf_2KyUcv2wjFnacKGk9z0T2Z5rZLjenQQ/edit?usp=sharing
## Uruchomienie

### Preprocess basic data
`python data_processing/preprocess_basic.py`
### Preprocess mature data
`python data_processing/preprocess_mature.py`
### Train basic model
`python train_basic_model.py`
### Train mature model
`python train_mature_model.py`
### Serve prediction (basic) with dummy data
`python serve_prediction.py -b -s resources/to_predict_data.jsonl`
### Serve prediction (mature) with dummy data
`python serve_prediction.py -s resources/to_predict_data.jsonl`
### Model score (AB) 
`python model_score.py -p prediction_file_name.csv -s session_status_file_name.csv`
### Session status
`python sessions_status.py -f session_log_file.jsonl`