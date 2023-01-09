import os
from src.inference.inference import run_benchmarks
from src.utils.prepare_dataset import load_afri_speech_data
from src.utils.utils import parse_argument


if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    split = args.data_csv_path.split("-")[1]

    # Load dataset
    if "whisper" in args.model_id_or_path:
        test_dataset = load_afri_speech_data(
            data_path=args.data_csv_path,
            max_audio_len_secs=args.max_audio_len_secs,
            audio_dir=args.audio_dir,
            split=split,
            return_dataset=False, gpu=args.gpu
        )
    else:
        test_dataset = load_afri_speech_data(
            data_path=args.data_csv_path,
            max_audio_len_secs=args.max_audio_len_secs,
            audio_dir=args.audio_dir, gpu=args.gpu,
            split=split,
        )

    run_benchmarks(
        model_id_or_path=args.model_id_or_path,
        test_dataset=test_dataset,
        output_dir=args.output_dir,
        gpu=args.gpu,
        batchsize=args.batchsize
    )
