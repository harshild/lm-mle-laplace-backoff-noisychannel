bash ngram_lm_train.bash mle 1 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./mle-1.pkl
bash ngram_lm_train.bash mle 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./mle-2.pkl
bash ngram_lm_train.bash laplace 1 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1.pkl
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2.pkl
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1-4.pkl k=4
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2-4.pkl k=4
bash ngram_lm_train.bash backoff 1 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-1-1.pkl e1=1
bash ngram_lm_train.bash backoff 1 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-1-1.pkl e1=1 e2=1

bash ngram_lm_eval.bash ./laplace-2-4.pkl /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu

bash ngram_lm_eval.bash ./backoff-1-1.pkl /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-dev.conllu


bash unigram_lan_id.bash "Ich" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
bash unigram_lan_id.bash "Best" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
bash unigram_lan_id.bash "blind" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"