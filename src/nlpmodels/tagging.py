import numpy as np
import statistics

class HiddenMarkovModel():
  def __init__(self) -> None:
    self.tagProbabilities = {} # P(Ti) = C(Ti) / corpusLength
    self.tagUnionProbabilities = {} # P(Ti, Ti-1) = C(Ti,Ti-1) / corpusLength-1
    self.wordUnionTagProbabilities = {} # P(Wi, Ti) = C(Wi,Ti) / corpusLength
    self.initialTagProbabilities = {}

    self.emissionProbabilities = {} # P(Wi|Ti) =  P(Wi, Ti) / P(Ti)
    self.transitionProbabilities = {} # P(Ti|Ti-1) = P(Ti, Ti-1) / P(Ti-1)

    self.corpusLength = 0
    self.corpusTokenlistLength = 0
    self.uniqueTags = []

  def _propertyCount(self):
    # TOKEN STRUCTURE
    # Token([('id', 2),
    #        ('form', 'cierto'),
    #        ('lemma', 'cierto'),
    #        ('upos', 'ADJ'),
    #        ('xpos', 'ADJ'),
    #        ('feats', OrderedDict([('Gender', 'Masc'), ('Number', 'Sing')])),
    #        ('head', 3),
    #        ('deprel', 'nsubj'),
    #        ('deps', None),
    #        ('misc', None)])

    tagCount = {} # C(Ti)
    tagUnionCount = {} # C(Ti,Ti-1)
    wordUnionTagCount = {} # C(Wi,Ti)
    initialTagCount = {} # C(Ti^(0))

    previousTag = None
    wordList = []

    for tokenList in self.corpus:
      self.corpusTokenlistLength += 1
      initialTag = tokenList[0][self.tagtype]
      try:
        initialTagCount[initialTag] += 1
      except KeyError:
        initialTagCount[initialTag] = 1
      
      for token in tokenList:
        self.corpusLength += 1
        word = token['form'].lower()
        tag = token[self.tagtype]

        # Tag Count
        try:
          tagCount[tag] += 1
        except KeyError:
          tagCount[tag] = 1

        # TagUnionTag Count
        if previousTag:
          try:
            tagUnionCount[f'{tag},{previousTag}'] += 1
          except KeyError:
            tagUnionCount[f'{tag},{previousTag}'] = 1

        # WordUnionTag Count
        try:
          wordUnionTagCount[f'{word},{tag}'] += 1
        except KeyError:
          wordUnionTagCount[f'{word},{tag}'] = 1

        wordList.append(word)
        previousTag = tag

    self.uniqueWords = list(set(wordList))
    self.uniqueTags = list(tagCount.keys())

    return tagCount, tagUnionCount, wordUnionTagCount, initialTagCount

  def train(self, tagtype, corpus):
    self.corpus = corpus
    self.tagtype = tagtype

    tagCount, tagUnionCount, wordUnionTagCount, initialTagCount = self._propertyCount()

    # PROBABILITY CALCULATION
    for tag in self.uniqueTags:
      self.tagProbabilities[tag] = tagCount[tag] / self.corpusLength
      try:
        self.initialTagProbabilities[tag] = initialTagCount[tag] / self.corpusTokenlistLength
      except KeyError:
        self.initialTagProbabilities[tag] = 0

      for previousTag in self.uniqueTags:
        key = f'{tag},{previousTag}'
        try:
          self.tagUnionProbabilities[key] = tagUnionCount[key] / (self.corpusLength-1)
        except KeyError:
          self.tagUnionProbabilities[key] = 0

      for word in self.uniqueWords:
        key = f'{word},{tag}'
        try:
          self.wordUnionTagProbabilities[key] = wordUnionTagCount[key] / self.corpusLength
        except KeyError:
          self.wordUnionTagProbabilities[key] = 0

    # EMISSION AND TRANSITION PROBABILITIES CALCULATION
    for tag in self.uniqueTags:
      for word in self.uniqueWords:
        self.emissionProbabilities[f'{word}|{tag}'] = self.wordUnionTagProbabilities[f'{word},{tag}'] / self.tagProbabilities[tag]

      for previousTag in self.uniqueTags:
        self.transitionProbabilities[f'{tag}|{previousTag}'] = self.tagUnionProbabilities[f'{tag},{previousTag}'] / self.tagProbabilities[previousTag]

  @staticmethod 
  def _isWordSeen(viterbiProbs):
    return any(viterbiProbs)

  def tag(self, tokenList):
    viterbiInitialProbs = []
    wordList = tokenList.copy()
    wordList = [word.lower() for word in wordList]

    # First word probability calculation
    for tag in self.uniqueTags:
      try:
        viterbiProb = self.initialTagProbabilities[tag]*self.emissionProbabilities[f'{wordList[0]}|{tag}']
      except KeyError:
        viterbiProb = 0
      viterbiInitialProbs.append(viterbiProb)

    predictedTags = []
    
    initialViterbiProb = max(viterbiInitialProbs)
    initialTag = self.uniqueTags[np.argmax(viterbiInitialProbs)]
    predictedTags.append(initialTag)

    wordList.pop(0)
    previousTag = initialTag
    previousViterbiProb = initialViterbiProb
    for word in wordList:
      viterbiProbs = []

      for tag in self.uniqueTags:
        # If word given tag(emission probability) not seen (KeyError), assign probability of viterbi path to 0
        try:
          # If previous word not seen (previousViterbiProb == 0), use emission probability instead
          if previousViterbiProb != 0:
            viterbiProb = previousViterbiProb * self.transitionProbabilities[f'{tag}|{previousTag}'] * self.emissionProbabilities[f'{word}|{tag}']
          else:
            viterbiProb = self.emissionProbabilities[f'{word}|{tag}']
        except KeyError:
          viterbiProb = 0
        viterbiProbs.append(viterbiProb)

      # If word seen, select the most probable tag, if not default to PROPN ()
      if self._isWordSeen(viterbiProbs):
        tag = self.uniqueTags[np.argmax(viterbiProbs)]
      else:
        tag = self.uniqueTags[4] #(PROPN)
      prob = max(viterbiProbs)
      predictedTags.append(tag)
      
      previousTag = tag
      previousViterbiProb = prob

    tagsDict = [(tokenList[i], predictedTags[i]) for i in range(len(tokenList))]

    return tagsDict
  
  @staticmethod
  def _accuracy(predictedTags, realTags):
    try:
      correctTags = 0
      for i in range(len(predictedTags)):
        if predictedTags[i] == realTags[i]:
          correctTags += 1

      accuracy = correctTags / len(predictedTags)
      return accuracy
    except IndexError:
      print('Tag arrays are not the same size!')

  def _getTagWordLists(self, corpus):
    corpusTags = []
    corpusWords = []

    for tokenList in corpus:
      tokenlistTags = []
      tokenlistWords = []
      for token in tokenList:
        tokenlistTags.append(token[self.tagtype])
        tokenlistWords.append(token['form'])

      corpusTags.append(tokenlistTags)
      corpusWords.append(tokenlistWords)

    return corpusWords, corpusTags

  
  def evaluate(self, testCorpus):
    corpusWords, corpusTags = self._getTagWordLists(testCorpus)

    predictedCorpusTags = []
    for words in corpusWords:
      predictedTags = [result[1] for result in self.tag(words)]
      predictedCorpusTags.append(predictedTags)

    accuracies = []
    for i in range(len(predictedCorpusTags)):
      accuracy = self._accuracy(predictedCorpusTags[i], corpusTags[i])
      accuracies.append(accuracy)
    mean_accuracy = statistics.mean(accuracies)

    print(f"Model Accuracy: {mean_accuracy}")

    return mean_accuracy