import sys
from collections import Counter
import pickle
import numpy
from dataset_parser import parse_conllu_dataset, findtokens, make_asymetric_pairs

# todo apply format t=for 10 decimal precision
BEGIN_OF_SENTENCE = '<BOS>'
UNKNOWN_OOV = "<UNK>"
END_OF_SENTENCE = "<EOS>"


class LanguageModel:
    def __init__(self, model_name, N, hyper_parameters, model_data):
        self.model_name = model_name
        self.N = N
        self.hyper_parameters = hyper_parameters
        self.model_data = model_data

    @property
    def model_name(self):
        return self._model_name

    @property
    def N(self):
        return self._N

    @property
    def hyper_parameters(self):
        return self._hyper_parameters

    @property
    def model_data(self):
        return self._model_data

    @hyper_parameters.setter
    def hyper_parameters(self, value):
        self._hyper_parameters = value

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @N.setter
    def N(self, value):
        self._N = value

    @model_data.setter
    def model_data(self, value):
        self._model_data = value


def validate_probability(p):
    if 0 < p <= 1:
        return p
    else:
        if p == 0:
            print("Optimization needed")
        else:
            if float("{0:.10f}".format(p)) == p:
                print("Wrong probability")
                return None
            else:
                validate_probability(float("{0:.10f}".format(p)))


def prepare_bigram_input(sentence_list, unk_f):
    for sentence in sentence_list:
        sentence.insert(0, BEGIN_OF_SENTENCE)
        sentence.append(END_OF_SENTENCE)
    token_len, type_count = prepare_unigram_input(sentence_list, 2 * unk_f)
    pairs = make_asymetric_pairs(sentence_list)
    pair_count = Counter(pairs)
    pair_count[(UNKNOWN_OOV, UNKNOWN_OOV)] = unk_f
    return pair_count, len(pairs) + unk_f, type_count, token_len


def prepare_unigram_input(sentence_list, unk_frequency):
    tokens = findtokens(sentence_list)
    type_count = Counter(tokens)
    type_count[UNKNOWN_OOV] = unk_frequency
    sum_cw_i = len(tokens) - type_count[BEGIN_OF_SENTENCE] + unk_frequency
    type_count.__delitem__(type_count[BEGIN_OF_SENTENCE])

    return sum_cw_i, type_count


def unigram_mle(sentence_list, unk_f):
    # p(w_j)=c(w_j) / ∑i=1=>n c(w_i)
    sum_cw_i, type_count = prepare_unigram_input(sentence_list, unk_f)
    p = {}
    for w_j, cw_j in type_count.items():
        p[w_j] = validate_probability(cw_j / sum_cw_i)
    return p


def bigram_mle(sentence_list):
    # p(w_k|w_i)=c(w_i,w_k) / c(w_i)
    pair_count, len_pair, type_count, token_len = prepare_bigram_input(sentence_list, 1)
    p = {}
    for (w_i, w_k), c_wi_wk in pair_count.items():
        # this is probability p(w_k|w_i)
        i = w_k + "|" + w_i
        p[i] = validate_probability(c_wi_wk / type_count[w_i])
    return p


def unigram_laplace(sentence_list, l):
    # p(w_j)=c(w_j) + k / ∑i=1=>n c(w_i) + k*V
    sum_cw_i, type_count = prepare_unigram_input(sentence_list, 1)
    p = {}
    for w_j, cw_j in type_count.items():
        p[w_j] = validate_probability((cw_j + l) / (sum_cw_i + (l * len(type_count))))
    return p


def bigram_laplace(sentence_list, l):
    # p(w_k|w_i)=c(w_i,w_k) + k / c(w_i) + k*V*V
    pair_count, _, type_count, _ = prepare_bigram_input(sentence_list, 1)
    p = {}
    for (w_i, w_k), c_wi_wk in pair_count.items():
        # this is probability p(w_k|w_i)
        i = w_k + "|" + w_i
        p[i] = validate_probability((c_wi_wk + 1) / (type_count[w_i] + (l * (len(type_count) ^ 2))))
    return p


def process_hyperparamters(params):
    hyperparams = {}
    for param in params.split(" "):
        hyperparams[param.split("=")[0]] = param.split("=")[1]
    return hyperparams


def query_unigram_train_model(train_model, w):
    if train_model.keys().__contains__(w):
        return train_model[w]
    else:
        return train_model[UNKNOWN_OOV]


def query_bigram_train_model(train_model, pair):
    if train_model.keys().__contains__(pair[1] + "|" + pair[0]):
        return train_model[pair[1] + "|" + pair[0]]
    else:
        return train_model[UNKNOWN_OOV + "|" + UNKNOWN_OOV]


def calculate_perplexity(N, train_model, conllu_data_dev):
    dev_dataset = parse_conllu_dataset(conllu_data_dev)
    sum_p = 0
    M = 0
    if N == 1:
        dev_token_len, dev_type_count = prepare_unigram_input(dev_dataset, 0)
        M = len(dev_type_count)
        for w, count in dev_type_count.items():
            p = query_unigram_train_model(train_model, w)
            sum_p = sum_p + (count * (numpy.log(p)))
    if N == 2:
        dev_pair_count, dev_len_pair, dev_type_count, dev_token_len = prepare_bigram_input(dev_dataset, 0)
        M = len(dev_pair_count)
        for pair, count in dev_pair_count.items():
            p = query_bigram_train_model(train_model, pair)
            sum_p = sum_p + (count * (numpy.log(p)))

    perplexity = numpy.exp(-(sum_p / M))

    return perplexity


def unigram_backoff(sentence_list_train, sentence_list_tune, delta1, eta1):
    sum_cw_i, type_count = prepare_unigram_input(sentence_list_train, 1)
    tune_sum_cw_i, tune_type_count = prepare_unigram_input(sentence_list_tune, 1)

    pb_sum = 0
    p = {}
    not_observed = []
    for y, _ in tune_type_count.items():
        if type_count[y] > eta1:
            p[y] = validate_probability((type_count[y] - delta1) / sum_cw_i)
            pb_sum = pb_sum + p[y]
        else:
            not_observed.append(y)

    constant_backoff = (1 - validate_probability(pb_sum))

    for y in not_observed:
        p[y] = constant_backoff / len(not_observed)
        pb_sum = pb_sum + p[y]

    return p


def bigram_backoff(sentence_list_train, sentence_list_tune, delta1, delta2, eta1=0, eta2=0):
    pair_count, _, type_count,_ = prepare_bigram_input(sentence_list_train, 1)
    pair_count_tune, _, type_count_tune, _ = prepare_bigram_input(sentence_list_tune, 1)

    q = unigram_backoff(sentence_list_train, sentence_list_tune, delta1, eta1)
    p = {}

    for x, _ in type_count_tune.items():
        pb_sum = 0
        pb_sum_not_observed = 0
        not_observed = []
        d = dict(filter(lambda elem: elem[0][0] == x, pair_count_tune.items()))
        for pair, count in d.items():
            if pair_count[pair] > eta2:
                p[pair[1] + "|" + pair[0]] = validate_probability((pair_count[pair] - delta2) / type_count[pair[0]])
                pb_sum = pb_sum + p[pair[1] + "|" + pair[0]]
            else:
                p[pair[1] + "|" + pair[0]] = q[pair[1]]
                pb_sum_not_observed = validate_probability(pb_sum_not_observed + p[pair[1] + "|" + pair[0]])
                not_observed.append(pair)

        if len(not_observed) > 0:
            alpha = (1 - pb_sum) / pb_sum_not_observed

            for pair in not_observed:
                p[pair[1] + "|" + pair[0]] = alpha * q[pair[1]]
                pb_sum = validate_probability(pb_sum + p[pair[1] + "|" + pair[0]])

    return p


def main():
    mode = sys.argv[1]
    if mode == "train":
        hyper_parameters = None
        model_name = sys.argv[2]
        N = int(sys.argv[3])
        conllu_data_train = sys.argv[4]
        conllu_data_tune = sys.argv[5]
        save_file = sys.argv[6]

        if sys.argv[7]:
            hyper_parameters = process_hyperparamters(sys.argv[7])

        sentence_list_train = parse_conllu_dataset(conllu_data_train)

        model_data = {}
        if model_name == "mle":
            if N == 1:
                unk_f = 1
                if hyper_parameters is not None and hyper_parameters.keys().__contains__("unk_f"):
                    unk_f = float(hyper_parameters["unk_f"])
                model_data = unigram_mle(sentence_list_train, unk_f)
            if N == 2:
                model_data = bigram_mle(sentence_list_train)

        if model_name == 'laplace':
            l = 1
            if hyper_parameters is not None and hyper_parameters.keys().__contains__("l"):
                l = float(hyper_parameters["l"])

            if N == 1:
                model_data = unigram_laplace(sentence_list_train, l)
            if N == 2:
                model_data = bigram_laplace(sentence_list_train, l)
        if model_name == 'backoff':
            eta1 = 1
            eta2 = 1
            delta1 = 0.1
            delta2 = 0.1
            if hyper_parameters is not None and hyper_parameters.keys().__contains__("eta1"):
                eta1 = float(hyper_parameters["eta1"])
            if hyper_parameters is not None and hyper_parameters.keys().__contains__("eta2"):
                eta2 = float(hyper_parameters["eta2"])
            if hyper_parameters is not None and hyper_parameters.keys().__contains__("delta1"):
                delta1 = float(hyper_parameters["delta1"])
            if hyper_parameters is not None and hyper_parameters.keys().__contains__("delta2"):
                delta2 = float(hyper_parameters["delta2"])
            sentence_list_tune = parse_conllu_dataset(conllu_data_tune)
            if N == 1:
                model_data = unigram_backoff(sentence_list_train, sentence_list_tune, delta1, eta1)
            if N == 2:
                model_data = bigram_backoff(sentence_list_train, sentence_list_tune, delta1, delta2, eta1, eta2)

        language_model = LanguageModel(model_name, N, hyper_parameters, model_data)
        pickle.dump(language_model, open(save_file, "wb"))
        print(language_model.model_name, language_model.N, language_model.hyper_parameters,
              calculate_perplexity(language_model.N, language_model.model_data, conllu_data_tune)
              )
    if mode == "eval":
        language_model = pickle.load(open(sys.argv[2], "rb"))
        conllu_data_eval = sys.argv[3]
        print(language_model.model_name, language_model.N, language_model.hyper_parameters,
              calculate_perplexity(language_model.N, language_model.model_data, conllu_data_eval)
              )


if __name__ == '__main__':
    main()
