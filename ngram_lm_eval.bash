save_model=$1
eval_conullu=$2

./venv/bin/python3 ./language_models.py eval "$save_model" "$eval_conullu"