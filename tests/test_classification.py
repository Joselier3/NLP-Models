"""
Tests for nlpmodules.classification module
"""

import pytest
import pathlib
import os

from nlpmodels import classification

CORPUS1_PATH = pathlib.Path('/home/joselier/corpus1')
SPAM_PATH = CORPUS1_PATH.joinpath('spam')
HAM_PATH = CORPUS1_PATH.joinpath('ham') 

@pytest.fixture
def spam_corpus():
  """Spam and Non-spam (ham) emails corpus"""
  emails = []
  categories = []
  #lectura de spam data
  for file in os.listdir(SPAM_PATH):
    with open(SPAM_PATH.joinpath(file), encoding='latin-1') as f:
      emails.append(f.read())
      categories.append('spam')
  #lectura de ham data
  for file in os.listdir(HAM_PATH):
    with open(HAM_PATH.joinpath(file), encoding='latin-1') as f:
      emails.append(f.read())
      categories.append('ham')

  return (emails, categories)

def test_naive_bayes_classifier(spam_corpus):
  (emails, categories) = spam_corpus
  model = classification.NaiveBayesClassifier()
  model.train(emails[:-1], categories[:-1])
  predict = model.classify(emails[-1])
  assert predict == categories[-1]