from conllu import parse_incr

class HiddenMarkovModel():
  def __init__(self) -> None:
    self.tagProbabilities = {} # P(Ti) = C(Ti) / corpusLength
    self.tagUnionProbabilities = {} # P(Ti, Ti-1) = C(Ti,Ti-1) / corpusLength-1
    self.wordUnionTagProbabilities = {} # P(Wi, Ti) = C(Wi,Ti) / corpusLength

    self.emissionProbabilities = {} # P(Wi|Ti) =  P(Wi, Ti) / P(Ti)
    self.transitionProbabilities = {} # P(Ti|Ti-1) = P(Ti, Ti-1) / P(Ti-1)
    self.corpusLength = 0

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

    lastTag = None
    for tokenList in parse_incr(self.corpus):
      for token in tokenList:
        self.corpusLength += 1
        tag = token[self.tagtype]

        # Tag Count
        try:
          tagCount[tag] += 1
        except KeyError:
          tagCount[tag] = 1

        # TagUnionTag Count
        if lastTag:
          try:
            tagUnionCount[f'{tag},{lastTag}'] += 1
          except KeyError:
            tagUnionCount[f'{tag},{lastTag}'] = 1

        # WordUnionTag Count
        try:
          wordUnionTagCount[f'{token},{tag}'] += 1
        except KeyError:
          wordUnionTagCount[f'{token},{tag}'] = 1

        lastTag = tag

    return tagCount, tagUnionCount, wordUnionTagCount

  def train(self, tagtype, corpus):
    self.corpus = corpus
    self.tagtype = tagtype

    tagCount, tagUnionCount, wordUnionTagCount = self._propertyCount()

    # Probability Calculation
    self.uniqueTags = tagCount.keys()

    for tag, count in tagCount.items():
      self.tagProbabilities[tag] = count / self.corpusLength
