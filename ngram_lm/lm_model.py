from collections import Counter
import numpy as np
import math

"""
CS6120 Homework 2 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"

tdata='training_files\iamsam2.txt'
tdata2='training_files\berp-training.txt'
#content=read_file(tdata)
# UTILITY FUNCTIONS
def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  # STUDENTS IMPLEMENT 
  ret=[]
  for i in range(len(tokens)):
      if i+n <= len(tokens):
          a=tuple(tokens[i:i+n])
          ret.append(a)
  return ret

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents
print('12')
def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  # PROVIDED
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total


def create_cumulative_distribution(d):
    """
    Convert a dictionary into a cumulative distribution.
    
    Args:
    d (dict): A dictionary where keys are objects (e.g., words) and counts 
    
    Returns:
    list of tuples: Each tuple contains an object and its cumulative probability.
    """
    n=sum(d.values())
    new_d = {key: value / n for key, value in d.items()}
    items = sorted(new_d.items(), key=lambda x: x[1])
    cumulative_distribution = []
    cumulative = 0
    for item, prob in items:
        cumulative += prob
        cumulative_distribution.append((item, cumulative))
    return cumulative_distribution


#tokenize_line('a',2)
class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    # STUDENTS IMPLEMENT
    self.n = n_gram
    #gram 
    self.C = None
    self.Vocabulary = None
  
  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # STUDENTS IMPLEMENT
    self.Vocabulary = Counter(tokens)
    adjusted_tokens = [token if self.Vocabulary[token] > 1 else '<UNK>' for token in tokens]
    self.Vocabulary = Counter(adjusted_tokens)
    if verbose:
      print('function start')
    #token=tokenize(tokens,self.n,by_char=False)
    if verbose:
      print(tokens)
    n_gram=create_ngrams(adjusted_tokens,self.n)
    if verbose:
      print(n_gram)
    self.C=Counter(n_gram)
    

  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # STUDENTS IMPLEMENT
    #print('start')
    adjusted_tokens = [token if self.Vocabulary[token] > 1 else '<UNK>' for token in sentence_tokens]
    score_gram=create_ngrams(adjusted_tokens,self.n)
    c = self.C
    p=1
    V=len(self.Vocabulary.keys())
    #print(len(score_gram))
    for i in range(len(score_gram)):
      #print('inside')
      numerator = self.C[score_gram[i]]
      if self.n != 1:
        de = sum([c[t] for t in c.keys() if t[0:self.n-1]==score_gram[i][0:self.n-1]])
      else:
        de = sum([c[t] for t in c.keys()])
      #print(numerator,'numerator')
      #print(de,'de')
      #print(V,'v')
      #laplace smoothing
      a=(numerator+1)/(de+V)
      p=p*a
    return p


  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # STUDENTS IMPLEMENT
    n=self.n
    beginning = [SENTENCE_BEGIN]*(n-1)
    ret=[]
    p=np.random.uniform(0,1,1)
    if n == 1:
      ret.append(SENTENCE_BEGIN)
      cumulative_distribution = create_cumulative_distribution(self.Vocabulary)
      print(cumulative_distribution)
      while True:
        p=np.random.uniform(0,1,1)
        for item, cumulative_prob in cumulative_distribution:
            #print('item','cum',item,cumulative_prob)
            if p[0] <= cumulative_prob:
                  if item != '<s>':
                    ret.append(item)
                  break
        if ret[-1] == SENTENCE_END:
          break
        print(ret)
    if n>1:
        #sentence begin
        for i in range(n-1):
          ret.append(SENTENCE_BEGIN)
        beginning_tuple = tuple(beginning)
        #create dict with same prefix of n-1 elements
        d= {k:self.C[k] for k in self.C if k[0:n-1] == beginning_tuple}
        #print('d',d)
        cumulative_distribution =  create_cumulative_distribution(d)
        while True:
          p=np.random.uniform(0,1,1)
          for item, cumulative_prob in cumulative_distribution:
              if p[0] <= cumulative_prob:
                    #print('item','cum',item,cumulative_prob)
                    ret.append(item[-1])
                    break
          if ret[-1] == SENTENCE_END:
             break
          #update prefix 
          beginning = ret[-(n-1):]
          beginning_tuple = tuple(beginning)
          beginning_count = {k:self.C[k] for k in self.C if k[0:n-1] == beginning_tuple}
          cumulative_distribution = create_cumulative_distribution(beginning_count)
          #print('ret',ret)
    #ret.append(cumulative_distribution[-1][0])
    return ret



  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for i in range(n)]


  def perplexity(self, sequence: list) -> float:
    """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    n=len(self.Vocabulary)
    score=self.score(sequence)
    return score**(-1/n)
    
  
# not required
if __name__ == '__main__':
  print()
  print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")
  print("call a function")
  print(tokenize_line("tokenize this sentence!", 2, by_char=False))
  print(tokenize(["apples are fruit", "bananas are too"], 2, by_char=False))
