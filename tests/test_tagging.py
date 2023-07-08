"""Test cases for the tagging module."""
import pytest
import pathlib
import conllu

from nlpmodels import tagging

CORPUS_TRAIN_PATH = pathlib.Path('/home/joselier/UD_Spanish-AnCora/es_ancora-ud-train.conllu')
CORPUS_TEST_PATH = pathlib.Path('/home/joselier/UD_Spanish-AnCora/es_ancora-ud-test.conllu')
TAGTYPE = 'upos'

@pytest.fixture
def spanish_train_corpus():
    """Spanish AnCora Corpus for training tagging models"""
    with open(CORPUS_TRAIN_PATH, "r", encoding="utf-8") as corpusFile:
        ancoraCorpus = list(conllu.parse_incr(corpusFile))
    return ancoraCorpus

@pytest.fixture
def spanish_sample_tokenlist():
    """Sample token list for tag method"""
    with open(CORPUS_TEST_PATH, "r", encoding="utf-8") as corpusFile:
        ancoraCorpus = list(conllu.parse_incr(corpusFile))
    sampleTokenlist = [token['form'] for token in ancoraCorpus[0]]
    return sampleTokenlist

@pytest.fixture
def spanish_test_corpus():
    """Test token list for evaluate method"""
    with open(CORPUS_TEST_PATH, "r", encoding="utf-8") as corpusFile:
        ancoraCorpus = list(conllu.parse_incr(corpusFile))
    return ancoraCorpus


def test_hidden_markov_model_spanish(spanish_train_corpus, spanish_test_corpus):
    """Hidden Markov Model achieves test accuracy greater than 90%"""
    model = tagging.HiddenMarkovModel()
    model.train(TAGTYPE, spanish_train_corpus)
    accuracy = model.evaluate(spanish_test_corpus)
    assert accuracy > 0.9

def test_maximum_entropy_markov_model_spanish(spanish_train_corpus, spanish_test_corpus):
    """MEMM achieves test accuracy greater than 90%"""
    model = tagging.MaximumEntropyMarkovModel()
    model.train(TAGTYPE, spanish_train_corpus)
    accuracy = model.evaluateTokenlist(spanish_test_corpus[0])
    assert accuracy > 0.9