bash ngram_lm_train.bash mle 1 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./mle-1.pkl
bash ngram_lm_train.bash mle 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./mle-2.pkl
bash ngram_lm_train.bash laplace 1 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1.pkl
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2.pkl
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1-4.pkl k=4
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2-4.pkl k=4

bash ngram_lm_eval.bash ./laplace-2-4.pkl /home/harshild/Documents/NLP/UD_English-EWT/en_ewt-ud-test.conllu

