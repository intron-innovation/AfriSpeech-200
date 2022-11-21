from src.inference import run_benchmarks, load_data


def main(model_list, dataset):
    for model_id in model_list:
        print(f"starting: {model_id}")
        run_benchmarks(model_id, dataset)


if __name__ == '__main__':
    dataset_path = "./data/intron-dev-public-3232.csv"
    intron_dataset = load_data(dataset_path)

    models_list = [
        'jonatasgrosman/wav2vec2-large-xlsr-53-english',
        "facebook/wav2vec2-large-960h",
        "jonatasgrosman/wav2vec2-xls-r-1b-english",
        "facebook/wav2vec2-large-960h-lv60-self",
        "facebook/hubert-large-ls960-ft",
        "facebook/wav2vec2-large-robust-ft-swbd-300h",
    ]

    main(models_list, intron_dataset)
