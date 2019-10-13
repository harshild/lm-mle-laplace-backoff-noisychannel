import operator
import sys
# import sklearn.metrics.precisions_score
# import sklearn.metrics.recall_score
import numpy
from sklearn.metrics import precision_score, recall_score

from dataset_parser import parse_conllu_dataset
from language_models import unigram_mle, unigram_laplace, UNKNOWN_OOV


def main():
    sentence = sys.argv[1].split(" ")
    language_datasets_path = {}

    for param in sys.argv[2].split(" "):
        param_split = param.split("=")
        language_datasets_path[param_split[0]] = param_split[1]

    sentence_count = {}
    sum_sentence_count = 0
    for lang_name, language_dataset_path in language_datasets_path.items():
        lang_sentence_list = parse_conllu_dataset(language_dataset_path)
        sentence_count[lang_name] = len(lang_sentence_list)
        sum_sentence_count = sum_sentence_count + sentence_count[lang_name]

    ll_p_sentence_l = {}
    for lang_name, language_dataset_path in language_datasets_path.items():
        lang_sentence_list = parse_conllu_dataset(language_dataset_path)
        lm = unigram_laplace(lang_sentence_list, gamma=1)

        log_p_sentence = 0
        for w in sentence:
            if lm.keys().__contains__(w):
                log_p_sentence = log_p_sentence + numpy.log(lm[w])
            else:
                log_p_sentence = log_p_sentence + numpy.log(lm[UNKNOWN_OOV])

        p_l = sentence_count[lang_name] / sum_sentence_count

        ll_p_sentence_l[lang_name] = log_p_sentence + numpy.log(p_l)

    print(ll_p_sentence_l)
    predicted = list(dict(sorted(ll_p_sentence_l.items(), key=operator.itemgetter(1), reverse=True)).keys())
    print(predicted)
    actual = sys.argv[3].strip('[]').split(',')
    print("Predicted", predicted)
    print("Actual", actual)

    print("Precision(Macro)", precision_score(actual, predicted, average="macro"))
    print("Precision(Micro)", precision_score(actual, predicted, average="micro"))
    print("Recall(Macro)", recall_score(actual, predicted, average="macro"))
    print("Recall(Micro)", recall_score(actual, predicted, average="micro"))

    print('Language Name , \u00EE = ', max(ll_p_sentence_l.items(), key=operator.itemgetter(1)))


if __name__ == '__main__':
    main()
