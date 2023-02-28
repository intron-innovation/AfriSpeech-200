import nemo.collections.asr as nemo_asr
import glob
import os
import sys
import jiwer
import pandas as pd
from tqdm.auto import tqdm
from src.utils.text_processing import clean_text

ABS = "/home/mila/c/chris.emezue/scratch/AfriSpeech-100/test/" # this is the path where the audio files exist
NEW_ABS = "/home/mila/c/chris.emezue/scratch/AfriSpeech-100/ffmpeg_test/" # this is the path where the converted audio files will stay
MODEL_DIR = sys.argv[1]
#MODEL_DIR = "/home/mila/c/chris.emezue/scratch/AfriSpeech-100/output/nemo_experiments/nemo_unfrozen/nemo_all/all/"
MODEL_PATH = os.path.join(MODEL_DIR,'Model-en.nemo')
print(f'Working on {MODEL_DIR}')
wavfiles =  glob.glob(ABS+"*/"+"*.wav")
uttids = []
predictions = []
references = []

#asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('nvidia/stt_en_conformer_ctc_large')
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(MODEL_PATH)


test_df = pd.read_csv('/home/mila/c/chris.emezue/AfriSpeech-Dataset-Paper/data/intron-test-public-5077.csv')

test_df['audio_paths'] = test_df['audio_paths'].apply(lambda x: x.replace('/AfriSpeech-100/test/',ABS))

audio_files = test_df['audio_paths'].values.tolist()
true_texts = test_df['transcript'].values.tolist()

#for audio_file, text in zip(audio_files,true_texts):
#import pdb;pdb.set_trace()
#for wavfile in tqdm(wavfiles):
#    import pdb;pdb.set_trace()


    #os.system(cmd + wavfile + cmd1 + os.path.join(NEW_ABS,wavfile.split("/")[-1]))
transcription = asr_model.transcribe(audio_files,batch_size=2)
#predictions.append(transcription)
#references.append(text)

data = pd.DataFrame(dict(predictions=transcription, reference=true_texts,
                            audio_paths=audio_files))


test_df["pred_clean"] = [clean_text(text) for text in transcription]
test_df["ref_clean"] = [clean_text(text) for text in true_texts]

all_wer = jiwer.wer(list(test_df["ref_clean"]), list(test_df["pred_clean"]))
print(f"Cleanup WER: {all_wer * 100:.2f} %")

df_path = os.path.join(MODEL_DIR,'test_df.csv')
test_df.to_csv(df_path)