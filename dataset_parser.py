import csv
import sys


def parse_conllu_dataset(filepath):
    """

    :rtype: list
    """
    sentence_list = []
    word_list = []

    csv.field_size_limit(sys.maxsize)

    with open(filepath, encoding="utf8") as file_tsv:
        for line in csv.reader(file_tsv, delimiter="\t"):
            # ignore commented lines
            if len(line) > 0 and not str.startswith(line[0], "#"):
                word_list.append(line[1])
            elif len(word_list) > 0:
                sentence_list.append(word_list)
                word_list = []

        if len(word_list) > 0:
            sentence_list.append(word_list)

    return sentence_list


def wordlist_to_sentence(word_list):
    return " ".join(word_list)


def average_wordcount(sentence_list):
    total_word_count = 0
    for sentence in sentence_list:
        total_word_count += len(sentence)

    return total_word_count / len(sentence_list)


def findtypes(sentence_list):
    types = set(findtokens(sentence_list))
    return types


def findtokens(sentence_list):
    tokens = [word for word_list in sentence_list for word in word_list]
    return tokens


def make_asymetric_pairs(sentence_list):
    pairs = []
    for i in range(0, len(sentence_list)):
        for index in range(0, len(sentence_list[i]) - 1):
            pairs.append((sentence_list[i][index], sentence_list[i][index + 1]))
    return pairs
