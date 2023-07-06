"""Test cases for the tagging module."""
import pytest
from click.testing import CliRunner
import pathlib
import nltk
nltk.download('punkt')
import conllu

from nlpmodels import tagging

CORPUS_PATH = pathlib.Path('/home/joselier/UD_Spanish-AnCora/es_ancora-ud-dev.conllu')
TAGTYPE = 'upos'

@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()

@pytest.fixture
def spanish_corpus():
    """Spanish AnCora Corpus for testing tagging models"""
    with open(CORPUS_PATH, "r", encoding="utf-8") as corpusFile:
        ancoraCorpus = list(conllu.parse_incr(corpusFile))
    return ancoraCorpus

@pytest.fixture
def spanish_token_list() -> list:
    """Sample Spanish Token List"""
    sequence = 'Hola, ayer me compre un helado muy grande y rico'
    tokenList = nltk.tokenize.word_tokenize(sequence, 'spanish')
    return tokenList

def test_hidden_markov_model_spanish(spanish_corpus, spanish_token_list):
    """Tag method returns a list"""
    model = tagging.HiddenMarkovModel()
    model.train(TAGTYPE, spanish_corpus)
    predictedTags = model.tag(spanish_token_list)
    print(predictedTags)
    assert type(predictedTags) == list
