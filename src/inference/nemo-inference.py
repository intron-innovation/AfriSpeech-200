import nemo.collections.asr as nemo_asr
import glob
import os
import jiwer
import pandas as pd
from src.utils.text_processing import clean_text

ABS = "/scratch/pbsjobs/axy327/dev/" # this is the path where the audio files exist
wavfiles =  glob.glob(ABS+"*/"+"*.wav")
uttids = []
predictions = []
references = []
cmd = "ffmpeg -i "
cmd1 = " -ac 1 -ar 16000 "
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
for wavfile in wavfiles:
    os.system(cmd + wavfile + cmd1 + wavfile.split("/")[-1])
    transcription = asr_model.transcribe([wavfile.split("/")[-1]])
    predictions.append(transcription)


    data = pd.DataFrame(dict(predictions=predictions, reference=references,
                             audio_paths=wavfile))


    data["pred_clean"] = [clean_text(text) for text in data["predictions"]]
    data["ref_clean"] = [clean_text(text) for text in data["reference"]]

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    print(f"Cleanup WER: {all_wer * 100:.2f} %")



    data.to_csv('nemo')





#with open("../../results/african-nlp-nemo-transducer-predictons", "w") as f:
#    for i in range(len(wavfiles)):
#        f.write(f"{wavfiles[i]}\t{predictions[i]}\n")