import sys
from collections import Counter
import pickle

import numpy

from dataset_parser import parse_conllu_dataset, findtokens, make_asymetric_pairs


def prepare_bigram_input(sentence_list):
    new_sentence_list = []
    for sentence in sentence_list:
        sentence.insert(0, '<BOS>')
        sentence.append('<EOS>')
        new_sentence_list.append(sentence)
    pairs = make_asymetric_pairs(new_sentence_list)
    tokens = findtokens(sentence_list)
    token_count = Counter(tokens)
    pair_count = Counter(pairs)
    return pair_count, token_count


def prepare_unigram_input(sentence_list):
    tokens = findtokens(sentence_list)
    sum_cw_i = len(tokens)
    token_count = Counter(tokens)
    return sum_cw_i, token_count


def unigram_mle(sentence_list):
    # p(w_j)=c(w_j) / ∑i=1=>n c(w_i)
    sum_cw_i, token_count = prepare_unigram_input(sentence_list)
    p = {}
    for w_j, cw_j in token_count.items():
        p[w_j] = cw_j / sum_cw_i
    return p


def bigram_mle(sentence_list):
    # p(w_k|w_i)=c(w_i,w_k) / c(w_i)
    pair_count, token_count = prepare_bigram_input(sentence_list)
    p = {}
    for (w_i, w_k), c_wi_wk in pair_count.items():
        # this is probability p(w_k|w_i)
        i = w_k + "|" + w_i
        p[i] = c_wi_wk / token_count[w_i]
    return p


def unigram_laplace(sentence_list, k):
    # p(w_j)=c(w_j) + k / ∑i=1=>n c(w_i) + k*V
    sum_cw_i, token_count = prepare_unigram_input(sentence_list)
    p = {}
    for w_j, cw_j in token_count.items():
        p[w_j] = (cw_j + k) / (sum_cw_i + (k * len(token_count)))
    return p


def bigram_laplace(sentence_list, k):
    # p(w_k|w_i)=c(w_i,w_k) + k / c(w_i) + k*V*V
    pair_count, token_count = prepare_bigram_input(sentence_list)
    p = {}
    for (w_i, w_k), c_wi_wk in pair_count.items():
        # this is probability p(w_k|w_i)
        i = w_k + "|" + w_i
        p[i] = (c_wi_wk + 1) / (token_count[w_i] + (k * (len(token_count) ^ 2)))
    return p


def process_hyperparamters(params):
    hyperparams = {}
    for param in params.split(" "):
        hyperparams[param.split("=")[0]] = param.split("=")[1]
    return hyperparams


def calculate_perplexity(N, model, conllu_data_tune):
    sentence_list = parse_conllu_dataset(conllu_data_tune)
    tokens = findtokens(sentence_list)
    perplexity = 0
    if N == 1:
        types = Counter(tokens)
        sum = 0
        for w in types:
            if model.keys().__contains__(w):
                sum = sum + (types[w] * (numpy.log(model[w])))
        perplexity = sum / len(tokens)
    if N == 2:
        pairs = make_asymetric_pairs(sentence_list)
        unique_pairs = Counter(pairs)
        sum = 0
        for w in unique_pairs:
            if model.keys().__contains__(w[0] + "|" + w[1]):
                sum = sum + (unique_pairs[w] * (numpy.log(model[w[0] + "|" + w[1]])))
        perplexity = sum / len(pairs)

    return perplexity


def main():
    mode = sys.argv[1]
    model_name = None
    hyperparameters = None
    model = None
    conllu_data_tune = None
    N = None
    if mode == "train":
        model_name = sys.argv[2]
        N = int(sys.argv[3])
        conllu_data_train = sys.argv[4]
        conllu_data_tune = sys.argv[5]
        save_file = sys.argv[6]

        if sys.argv[7]:
            hyperparameters = process_hyperparamters(sys.argv[7])


        sentence_list_train = parse_conllu_dataset(conllu_data_train)

        model = {}
        if model_name == "mle":
            if N == 1:
                model = unigram_mle(sentence_list_train)
            if N == 2:
                model = bigram_mle(sentence_list_train)

        if model_name == 'laplace':
            k = 1
            if hyperparameters.keys().__contains__("k"):
                k = int(hyperparameters["k"])

            if N == 1:
                model = unigram_laplace(sentence_list_train, k)
            if N == 2:
                model = bigram_laplace(sentence_list_train, k)

        pickle.dump(model, open(save_file, "wb"))

    if mode == "eval":
        model = pickle.load(open(sys.argv[2], "rb"))
        conllu_data_tune=sys.argv[3]
        N = 2
    print(model_name, N, hyperparameters, calculate_perplexity(N, model, conllu_data_tune))


if __name__ == '__main__':
    main()
