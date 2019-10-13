bash ngram_lm_train.bash mle 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./mle-1.pkl
bash ngram_lm_train.bash mle 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./mle-2.pkl
bash ngram_lm_train.bash laplace 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1.pkl
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2.pkl
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1-4.pkl k=4
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2-4.pkl k=4
bash ngram_lm_train.bash backoff 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-1-1.pkl e1=1
bash ngram_lm_train.bash backoff 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-1-1.pkl e1=1 e2=1

bash ngram_lm_eval.bash ./mle-1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu
bash ngram_lm_eval.bash ./laplace-2-4.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu
bash ngram_lm_eval.bash ./backoff-1-1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu


bash unigram_lan_id.bash "Best" "[German,English]" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
bash unigram_lan_id.bash "blind" "[German,English]" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
bash unigram_lan_id.bash "Ich" "[German,English]" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"

bash unigram_lan_id.bash "Ich great" "[German,English]" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
bash unigram_lan_id.bash "Ich great score" "[German,English]" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"
bash unigram_lan_id.bash "Ich great bin" "[German,English]" "English=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" "German=/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu"


#Assignment3

bash ngram_lm_train.bash mle 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./mle-1-1.pkl unk_f=1
bash ngram_lm_train.bash mle 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./mle-1-11.pkl unk_f=11
bash ngram_lm_train.bash mle 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./mle-1-11111.pkl unk_f=11111
bash ngram_lm_train.bash mle 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./mle-2.pkl

bash ngram_lm_train.bash laplace 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1-0.pkl gamma=0
bash ngram_lm_train.bash laplace 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1-1.pkl gamma=1
bash ngram_lm_train.bash laplace 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-1-10.pkl gamma=10
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2-1.pkl gamma=1
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2-10.pkl gamma=10
bash ngram_lm_train.bash laplace 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./laplace-2-100.pkl gamma=100

bash ngram_lm_train.bash backoff 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-1-1-0.1.pkl epsilon1=1 delta1=0.1
bash ngram_lm_train.bash backoff 1 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-1-2-0.2.pkl epsilon1=2 delta1=0.2
bash ngram_lm_train.bash backoff 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-2-1-0.1-1-0.1.pkl epsilon1=1 epsilon2=1 delta1=0.1 delta2=0.1
bash ngram_lm_train.bash backoff 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-2-2-0.1-1-0.2.pkl epsilon1=2 epsilon2=1 delta1=0.1 delta2=0.2
bash ngram_lm_train.bash backoff 2 /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu ./backoff-2-2-0.2-2-0.2.pkl epsilon1=2 epsilon2=2 delta1=0.2 delta2=0.2

bash ngram_lm_eval.bash ./mle-1-1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./mle-1-11.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./mle-1-11111.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./mle-2.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu

bash ngram_lm_eval.bash ./laplace-1-0.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./laplace-1-1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./laplace-1-10.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./laplace-2-1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./laplace-2-10.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./laplace-2-100.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu

bash ngram_lm_eval.bash ./backoff-1-1-0.1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./backoff-1-2-0.2.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./backoff-2-1-0.1-1-0.1.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./backoff-2-2-0.1-1-0.2.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu
bash ngram_lm_eval.bash ./backoff-2-2-0.2-2-0.2.pkl /home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu