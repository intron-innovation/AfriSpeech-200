import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from google.cloud import speech
import azure.cognitiveservices.speech as speechsdk
from datasets import load_metric

from src.utils.utils import parse_argument
from src.utils.text_processing import clean_text

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""  # credentials file
wer_metric = load_metric("wer")

google_client = speech.SpeechClient()


def get_config(service):
    if "gcp" in service:
        speech_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US",
            sample_rate_hertz=44100,
            audio_channel_count=2,
            model='medical_dictation' if 'medical' in service else 'default'
        )
    elif "azure" in service:
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                               region=os.environ.get('SPEECH_REGION'))
        speech_config.speech_recognition_language="en-US"
    return speech_config


def azure_asr(fname, speech_config):
    done = False
    result = ""
    speech_recognition_result = None
    
    def stop_cb(evt):
        # print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        nonlocal speech_recognition_result
        done = True
        speech_recognition_result = evt.result

    # audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    audio_config = speechsdk.audio.AudioConfig(filename=fname)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # speech_recognition_result = speech_recognizer.recognize_once_async().get()
    
    # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    # speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    # speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)
    
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        result = speech_recognition_result.text
        # print("Recognized: {}".format(result))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return result


def gcp_transcribe_file(speech_file, recognition_config):
    """
    Transcribe the given audio file asynchronously.
    Note that transcription is limited to a 60 seconds audio file.
    Use a GCS file for audio longer than 1 minute.
    """

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    response = google_client.recognize(config=recognition_config, audio=audio)
    if len(response.results) > 0:
        return response.results[0].alternatives[0].transcript
    return ""


def azure_cleanup(text):
    return text.lower() \
        .replace(" comma,", ",") \
        .replace(" comma", ",") \
        .replace(" full stop.", ".") \
        .replace(" full stop", ".") \
        .replace(",.", ".") \
        .replace(",,", ",")


def gcp_cleanup(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    text = text.lower()
    text = text.replace(" [ comma ] ", ", ") \
        .replace(" [ hyphen ] ", "-") \
        .replace(" [ full stop ] ", ".") \
        .replace(" [ full stop", ".") \
        .replace(" [ full", ".") \
        .replace(" [ question mark ]", "?") \
        .replace(" [ question mark", "?") \
        .replace(" [ question", "?") \
        .replace("[ next line ]", "next line") \
        .strip()
    text = " ".join(text.split())
    return text


def main_transcribe_medical(data, service):
    speech_config = get_config(service)
    preds_raw = []
    preds_clean = []
    wers = []

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        if not os.path.isfile(row['audio_paths']):
            preds_raw.append("")
            preds_clean.append("") 
            wers.append(0)
            continue
        
        if 'gcp' in service:
            pred = gcp_transcribe_file(row['audio_paths'], speech_config)
            if 'medical' in service:
                pred_clean = clean_text(gcp_cleanup(pred))
            else:
                pred_clean = clean_text(pred)
        elif 'azure' in service:
            pred = azure_asr(row['audio_paths'], speech_config)
            pred_clean = clean_text(azure_cleanup(pred))
        else:
            raise NotImplementedError
            
        preds_raw.append(pred)
        preds_clean.append(pred_clean)
        
        wers.append(wer_metric.compute(predictions=[pred_clean], references=[clean_text(row['transcript'])]))
        if idx % 400 == 0:
            # avoid RateLimitExceeded error
            time.sleep(60)

    assert len(data) == len(preds_raw) == len(wers) == len(preds_clean)
    data['predictions_raw'] = preds_raw
    data['predictions'] = preds_clean
    data['wer'] = wers

    return data


def write_gcp_results(data, model_id_or_path='gcp-transcribe',
                      output_dir="./results", split='dev'):
    """
    Write predictions and wer to disk
    :param split: str, train/test/dev
    :param output_dir: str
    :param data: DataFrame
    :param model_id_or_path: str
    :return:
    """
    all_wer = np.mean(data['wer'])
    out_path = f'{output_dir}/intron-open-{split}-{model_id_or_path}-wer-{round(all_wer, 4)}-{len(data)}.csv'
    data_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == '__main__':
    
    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    split = args.data_csv_path.split("-")[1]
    
    data_df = pd.read_csv(args.data_csv_path)
    data_df = data_df[data_df.duration < 17]
    data_df["audio_paths"] = data_df["audio_paths"].apply(
        lambda x: x.replace(f"/AfriSpeech-100/{split}/", args.audio_dir)
    )

    assert 'gcp' in args.model_id_or_path or 'azure' in args.model_id_or_path
    data_df = main_transcribe_medical(data_df, args.model_id_or_path)
    
    write_gcp_results(data_df, args.model_id_or_path, split=split)
