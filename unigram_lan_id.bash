sentence=$1
dataset_list="${*:2}"

python3 ./language_identification.py "$sentence" "$dataset_list"