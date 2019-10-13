model_name=$1
N=$2
train_conullu=$3
tune_conullu=$4
save_model=$5
hp_args="${*:6}"

./venv/bin/python3 ./language_models.py train "$model_name" "$N" "$train_conullu" "$tune_conullu" "$save_model" "$hp_args"