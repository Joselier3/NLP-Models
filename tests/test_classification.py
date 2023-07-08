"""
Tests for nlpmodules.classification module
"""

import pytest
import pathlib
import os
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sklearn.model_selection import train_test_split

from nlpmodels import classification

CORPUS1_PATH = pathlib.Path('/home/joselier/corpus1')
SPAM_PATH = CORPUS1_PATH.joinpath('spam')
HAM_PATH = CORPUS1_PATH.joinpath('ham') 

@pytest.fixture
def spam_corpus():
  """Spam and Non-spam (ham) emails corpus"""
  emails = []
  categories = []
  lang = English()
  tokenizer = Tokenizer(lang.vocab)

  for file in os.listdir(SPAM_PATH):
    with open(SPAM_PATH.joinpath(file), encoding='latin-1') as f:
      email = f.read()
      emails.append(tokenizer(email))
      categories.append('spam')
      
  for file in os.listdir(HAM_PATH):
    with open(HAM_PATH.joinpath(file), encoding='latin-1') as f:
      emails.append(f.read())
      categories.append('ham')

  train_emails, test_emails, train_categories, test_categories = train_test_split(emails, categories, train_size=0.75)

  return (train_emails, test_emails, train_categories, test_categories)

def test_naive_bayes_classifier(spam_corpus):
  """Asserts an accuracy greater than or equal to 90%"""
  (train_emails, test_emails, train_categories, test_categories) = spam_corpus
  model = classification.NaiveBayesClassifier()
  model.train(train_emails, train_categories)
  accuracy = model.evaluate(test_emails, test_categories)
  assert accuracy >= 0.9