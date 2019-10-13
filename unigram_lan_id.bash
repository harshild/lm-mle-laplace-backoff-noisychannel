sentence=$1
actual_lid=$2
dataset_list="${*:3}"

./venv/bin/python3 ./language_identification.py "$sentence" "$dataset_list" "$actual_lid"