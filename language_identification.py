import operator

import numpy

from dataset_parser import parse_conllu_dataset
from language_models import unigram_mle, unigram_laplace


def main():
    language_datasets_path = {
        "English": "/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu",
        "German": "/home/harshild/Documents/NLP/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu",
    }
    sentence = "Ich".split(" ")

    sentence_count = {}
    sum_sentence_count = 0
    for lang_name, language_dataset_path in language_datasets_path.items():
        lang_sentence_list = parse_conllu_dataset(language_dataset_path)
        sentence_count[lang_name] = len(lang_sentence_list)
        sum_sentence_count = sum_sentence_count + sentence_count[lang_name]

    ll_p_sentence_l = {}
    for lang_name, language_dataset_path in language_datasets_path.items():
        lang_sentence_list = parse_conllu_dataset(language_dataset_path)
        lm = unigram_laplace(lang_sentence_list, k=1)

        log_p_sentence = 0
        for w in sentence:
            if lm.keys().__contains__(w):
                log_p_sentence = log_p_sentence + numpy.log(lm[w])
            else:
                log_p_sentence = log_p_sentence + numpy.log(0.00000000001)

        p_l = sentence_count[lang_name] / sum_sentence_count

        ll_p_sentence_l[lang_name] = log_p_sentence + numpy.log(p_l)

    print(max(ll_p_sentence_l.items(), key=operator.itemgetter(1))[0])


if __name__ == '__main__':
    main()
