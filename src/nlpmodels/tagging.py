from conllu import parse_incr
import numpy as np

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

    for tokenList in parse_incr(self.corpus):
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
      self.initialTagProbabilities[tag] = initialTagCount[tag] / self.corpusTokenlistLength

      for previousTag in self.uniqueTags:
        key = f'{tag},{previousTag}'
        try:
          self.tagUnionProbabilities[key] = tagUnionCount[key] / self.corpusLength-1
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

  def tag(self, tokenList):
    viterbiInitialProbs = []
    for tag in self.uniqueTags:
      viterbiProb = self.initialTagProbabilities[tag]*self.emissionProbabilities[f'{tokenList[0]}|{tag}']
      viterbiInitialProbs.append(viterbiProb)

    predictedTags = []
    
    initialViterbiProb = max(viterbiInitialProbs)
    initialTag = self.uniqueTags[np.argmax(viterbiInitialProbs)]
    predictedTags.append(initialTag)

    tokenList.pop(0)
    previousTag = initialTag
    previousViterbiProb = initialViterbiProb
    for word in tokenList:
      viterbiProbs = []

      for tag in self.uniqueTags:
        viterbiProb = previousViterbiProb * self.transitionProbabilities[f'{tag}|{previousTag}'] * self.emissionProbabilities[f'{word}|{tag}']
        viterbiProbs.append(viterbiProb)

      tag = self.uniqueTags[np.argmax(viterbiProbs)]
      prob = max(viterbiProbs)
      predictedTags.append(tag)
      
      previousTag = tag
      previousViterbiProb = prob

    return predictedTags