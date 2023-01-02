import re
import os
import pandas as pd
from pathlib import Path
import librosa
import json


def write_pred(model_id_or_path, results, wer, cols=None, output_dir="./results", split="dev"):
    """
    Write model predictions to file
    :param cols: List[str]
    :param output_dir: str
    :param model_id_or_path: str
    :param results: Dataset instance
    :param wer: float
    :return: DataFrame
    """
    model_id_or_path = model_id_or_path.replace("/", "-")
    if cols is None:
        cols = ["audio_paths", "text", "reference", "predictions", "wer", "accent"]
    predictions_data = {col: results[col] for col in cols}
    predictions_df = pd.DataFrame(data=predictions_data)

    output_path = f"{output_dir}/intron-open-{split}-{model_id_or_path}-wer-{round(wer, 4)}-{len(predictions_df)}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(output_path)
    return predictions_df


def get_s3_file(s3_file_name,
                s3_prefix="http://bucket-name.s3.amazonaws.com/",
                local_prefix="s3",
                bucket_name=None, s3=None):
    """
    download file from s3 bucket
    :param s3_file_name:
    :param s3_prefix:
    :param local_prefix:
    :param bucket_name:
    :param s3:
    :return:
    """
    local_file_name = s3_file_name.replace(s3_prefix, local_prefix)
    if not os.path.isfile(local_file_name):
        Path(os.path.dirname(local_file_name)).mkdir(parents=True, exist_ok=True)
        s3_key = s3_file_name[54:]
        s3.Bucket(bucket_name).download_file(Key=s3_key, Filename=local_file_name)
    return local_file_name


def get_json_result(local_file_name):
    with open(local_file_name, 'r') as f:
        result = json.load(f)
    pred = result['results']['transcripts'][0]['transcript']
    return pred
