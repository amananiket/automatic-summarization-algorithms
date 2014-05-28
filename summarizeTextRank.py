import nltk
from collections import Counter
import sklearn.feature_extraction.text as fe
import networkx

sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw = r"Narendra Modi will visit Rajghat on Monday morning to pay homage to Mahatma Gandhi ahead of being sworn in as the 15th Prime Minister of India. Mr Modi, who is scheduled to take the oath of office and secrecy at 6 pm on Wednesday, would visit Rajghat at 7 am, official sources said. Mr Modi's swearing-in will be attended by nearly 3,000 guests, including top leaders from SAARC countries like Pakistan Prime Minister Nawaz Sharif and Sri Lankan President Mahinda Rajapaksa. (Before Narendra Modi's Swearing-in, Speculation on his Cabinet) Outgoing Prime Minister Manmohan Singh, Congress President Sonia Gandhi and party vice-president Rahul Gandhi, besides leaders of various other parties and chief ministers of a number of states will also be attending the function in the forecourt of Rashtrapati Bhavan."

sentTokens = sentenceTokenizer.tokenize(raw)

#def wordsCounter(sentence):
#    wordTokens = nltk.wordpunct_tokenize(sentence)
#    return Counter(word.lower() for word in wordTokens)

featureMatrix = fe.CountVectorizer().fit_transform(sentTokens)
normalizedMatrix = fe.TfidfTransformer().fit_transform(featureMatrix)

similarityGraph = normalizedMatrix*normalizedMatrix.T

networkGraph = networkx.from_scipy_sparse_matrix(similarityGraph)
scores = networkx.pagerank(networkGraph)

print scores


