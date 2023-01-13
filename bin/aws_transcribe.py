import boto3
import os
import pandas as pd
from tqdm import tqdm
import time
from datasets import load_metric
import numpy as np
import botocore

from src.utils.utils import parse_argument, get_s3_file, get_json_result
from src.utils.text_processing import clean_text

wer_metric = load_metric("wer")

transcribe = boto3.client(
    'transcribe',
    region_name='eu-west-2'
)

bucket_name = "intron-open-source"

s3 = boto3.resource(
    service_name='s3',
    region_name='eu-west-2',
)

UNIQUE_JOB_NUM=40

def get_aws_transcript_medical(job_name, job_uri, service):
    """
    Make calls to transcribe service
    :param service: str [transcribe-medical or transcribe]
    :param job_name: str, unique job name
    :param job_uri: str, s3 path
    :return:
    """
    transcribe.start_medical_transcription_job(
        MedicalTranscriptionJobName=job_name,
        Media={
            'MediaFileUri': job_uri
        },
        OutputBucketName=bucket_name,
        OutputKey=f'aws-{service}-output-files/',
        LanguageCode='en-US',
        Specialty='PRIMARYCARE',
        Type='DICTATION'
    )
    status = transcribe.get_medical_transcription_job(MedicalTranscriptionJobName=job_name)


def get_aws_transcript(job_name, job_uri, service):
    """
    Make calls to transcribe medical service
    :param service: str [transcribe-medical or transcribe]
    :param job_name: str, unique job name
    :param job_uri: str, s3 path
    :return:
    """
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={
            'MediaFileUri': job_uri
        },
        MediaFormat='wav',
        LanguageCode='en-US',
        OutputBucketName=bucket_name,
        OutputKey=f'aws-{service}-output-files/',
    )
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)


def aws_transcribe(data_df, service, split, audio_dir):
    """
    Send requests to AWS transcribe medical service
    :param service: str [transcribe-medical or transcribe]
    :param data_df: DataFrame
    :return:
    """
    s3_paths = ["http://speech-app.s3.amazonaws.com/static/audio/uploads/",
                "https://speech-app.s3.eu-west-2.amazonaws.com/static/audio/uploads/"
               ]
    for idx, row in tqdm(data_df.iterrows(), total=data_df.shape[0], desc=f"call {service} endpoint"):
        audio_path = row['audio_paths']

        # print(idx)
        if not os.path.isfile(audio_path.replace(f"/AfriSpeech-100/{split}/", audio_dir)):
            print(audio_path)
            continue
        if split in ['train', 'dev']:
            s3_uri = f"s3://intron-open-source{audio_path}"
        else:
            audio_path = audio_path.replace(f"/AfriSpeech-100/{split}/", "")
            s3_uri = f"{s3_paths[0]}{audio_path}"
              
        s3_job_name = f"{split}-transcription-job-{UNIQUE_JOB_NUM}-{service}-{idx}"
        # print(s3_job_name)
        try:
            if "medical" in service:
                get_aws_transcript_medical(s3_job_name, s3_uri, service)
            else:
                get_aws_transcript(s3_job_name, s3_uri, service)
        except Exception as e:
            print(idx, str(e))
            
        if idx % 100 == 0:
            # avoid RateLimitExceeded error
            time.sleep(60)


def get_aws_results_from_s3(data_df, service, audio_dir, split="dev"):
    """
    Get transcription results written to s3 directory
    :param data_df: DataFrame
    :param service: [transcribe-medical or transcribe]
    :return: List[str], List[float]
    """
    preds = []
    preds_clean = []
    wers = []
    for idx, row in tqdm(data_df.iterrows(), total=data_df.shape[0], desc="get results from s3"):
        if not os.path.isfile(row['audio_paths'].replace(f"/AfriSpeech-100/{split}/", audio_dir)):
            preds.append("")
            preds_clean.append("") 
            wers.append(None)
            continue
        pred = ""
        pred_clean = ""
        s3_job_name = f"{split}-transcription-job-{UNIQUE_JOB_NUM}-{service}-{idx}"
        predicted_transcript_file = f'https://s3.eu-west-2.amazonaws.com/' \
                                    f'intron-open-source/aws-{service}-output-files/{"medical/" if "medical" in service else ""}{s3_job_name}.json'
        s3_prefix = f'https://s3.eu-west-2.amazonaws.com/intron-open-source/aws-{service}-output-files{"/medical" if "medical" in service else ""}'
        try:
            local_dev_fname = get_s3_file(predicted_transcript_file,
                                          s3_prefix=s3_prefix,
                                          local_prefix="/data/saved_models/predictions/aws/dev",
                                          bucket_name=bucket_name,
                                          s3=s3)
            pred = get_json_result(local_dev_fname)
            # print(s3_prefix, predicted_transcript_file, local_dev_fname, pred)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ClientError':
                print(idx, "Object does not exists")
            else:
                print(idx, "Unexpected error: %s" % e)
            
        except botocore.exceptions.ParamValidationError as error:
            raise ValueError('The parameters you provided are incorrect: {}'.format(error))

        pred_clean = clean_text(pred)
        preds_clean.append(pred_clean)
        preds.append(pred)
        wers.append(wer_metric.compute(predictions=[pred_clean], references=[clean_text(row['transcript'])]))

    return preds, wers, pred_clean


def write_aws_results(data_df, predictions, preds_clean, wer_list, split="dev",
                      model_id_or_path='aws-transcribe', output_dir="./results"):
    """
    Write predictions and wer to disk
    :param output_dir: str
    :param data_df: DataFrame
    :param predictions: List[str]
    :param wer_list: List[float]
    :param model_id_or_path: str
    :return:
    """
    data_df['predictions'] = predictions
    data_df['predictions_clean'] = preds_clean
    data_df['wer'] = wer_list
    all_wer = np.mean(data['wer'])

    out_path = f'{output_dir}/intron-open-{split}-{model_id_or_path}-wer-{round(all_wer, 4)}-{len(data_df)}.csv'
    data_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == '__main__':
    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    split = args.data_csv_path.split("-")[1]
    
    data = pd.read_csv(args.data_csv_path)
    data = data[data.duration < 17]

    # aws_service = 'transcribe-medical'
    assert 'aws' in args.model_id_or_path
    aws_service = "-".join(args.model_id_or_path.split('-')[1:])
    print(aws_service)
    
    #UNIQUE_JOB_NUM+=1
    aws_transcribe(data, aws_service, split, args.audio_dir)

    prediction_list, all_wer_list, pred_clean_list = get_aws_results_from_s3(data, aws_service, args.audio_dir, split)

    write_aws_results(data, prediction_list, pred_clean_list, all_wer_list, split, args.model_id_or_path)
