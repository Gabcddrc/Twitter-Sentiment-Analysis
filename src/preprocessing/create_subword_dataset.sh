# Prepare data for subword-nmt
python extract_text_from_dataset.py


# Extract subword units
subword-nmt learn-bpe -s 64000 < training_set_text_only.csv > subword-training-units.txt

subword-nmt apply-bpe -c subword-training-units.txt < training_set_text_only.csv > training_set_subwords_text_only.csv
subword-nmt apply-bpe -c subword-training-units.txt < validation_set_text_only.csv > validation_set_subwords_text_only.csv
subword-nmt apply-bpe -c subword-training-units.txt < test_set_text_only.csv > test_set_subwords_text_only.csv


# Reconstruct data set
mkdir dataset-subwords
python reconstruct_dataset.py