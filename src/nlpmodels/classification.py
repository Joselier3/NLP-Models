import math
import numpy as np
from sklearn.metrics import accuracy_score

class NaiveBayesClassifier:
	def __init__(self):
		# argmax_k log P(Ck) + \sum log P(Wi|Ck)
		self.wordGivenCategoryProbs = {} # P(Wi|Ck) = P(Wi,Ck) / P(Ck)
		self.categoryProbs = {} # P(Ck) = C(Ck) / docsLen
		self.wordUnionCategoryProbs = {} # P(Wi,Ck) = C(Wi,Ck) / corpusLen

		self.corpusLen = 0
		self.docsLen = 0
		self.uniqueCategories = set()
		self.uniqueWords = set()

	def _propertyCount(self):
		categoryCount = {}
		wordUnionCategoryCount = {}

		for i in range(len(self.corpus)):
			self.docsLen += 1
			self.uniqueCategories.add(self.categories[i])
			try:
				categoryCount[self.categories[i]] += 1
			except KeyError:
				categoryCount[self.categories[i]] = 1
				
			for word in self.corpus[i]:
				self.corpusLen += 1
				self.uniqueWords.add(word)
				try:
					wordUnionCategoryCount[f'{word},{self.categories[i]}'] += 1
				except KeyError:
					wordUnionCategoryCount[f'{word},{self.categories[i]}'] = 1

		return categoryCount, wordUnionCategoryCount
	
	def train(self, corpus, categories):
		self.corpus = corpus
		self.categories = categories

		categoryCount, wordUnionCategoryCount = self._propertyCount()

		# Category Probability Calculation
		for cat in self.uniqueCategories:
			self.categoryProbs[cat] = categoryCount[cat] / self.docsLen

			# Word Union Category Probability
			for word in self.uniqueWords:
				try:
					key = f'{word},{cat}'
					self.wordUnionCategoryProbs[key] = wordUnionCategoryCount[key] / self.corpusLen
				except KeyError:
					# Laplace smoothing to prevent assigning 0 probability to unseen events 
					# as probability 0 causes problems when using logarithms
					self.wordUnionCategoryProbs[key] = 1 / self.corpusLen

	def classify(self, doc):
		categories = list(self.uniqueCategories)
		categoryProbPaths = []
		for category in categories:
			categoryProb = math.log10(self.categoryProbs[category])
			for word in doc:
				try:
					categoryProb += math.log10(self.wordUnionCategoryProbs[f'{word},{category}'] / self.categoryProbs[category])
				except KeyError:
					categoryProb += 0 	
			categoryProbPaths.append(categoryProb)

		predictedCategory = categories[np.argmax(categoryProbPaths)]
		return predictedCategory
	
	def evaluate(self, test_docs, test_categories):
		predicted_categories = []
		for doc in test_docs:
			predicted_categories.append(self.classify(doc))
		
		accuracy = accuracy_score(predicted_categories, test_categories)
		print(f'NaiveBayesClassifier Accuracy: {accuracy}')
		return accuracy