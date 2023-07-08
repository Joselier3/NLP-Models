import math
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

class NaiveBayesClassifier:
	def __init__(self):
		# argmax_k log P(Ck) + \sum log P(Wi|Ck)
		self.wordGivenCategoryProbs = {} # P(Wi|Ck) = P(Wi,Ck) / P(Ck)
		self.categoryProbs = {} # P(Ck) = C(Ck) / docsLen
		self.wordUnionCategoryProbs = {} # P(Wi,Ck) = C(Wi,Ck) / corpusLen

	def _propertyCount(self):
		categoryCount = {}
		wordUnionCategoryCount = {}

		for i in range(len(self.corpus)):
			try:
				categoryCount[self.categories[i]] += 1
			except KeyError:
				categoryCount[self.categories[i]] = 1
				
			for word in self.corpus[i]:
				try:
					wordUnionCategoryCount[f'{word},{self.categories[i]}'] += 1
				except KeyError:
					wordUnionCategoryCount[f'{word},{self.categories[i]}'] = 1