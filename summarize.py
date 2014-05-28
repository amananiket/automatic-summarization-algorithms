import nltk
import re
from nltk.corpus import wordnet as wn
import copy
from cPickle import load
from time import time
import numpy

input = open('tag.pkl','rb')
tagger = load(input)
input.close()
start = time()

count = 0

def sentDistance(index1,index2):
    sentIndex1Set = False
    sentIndex2Set = False
    
    for i in range(0,len(sentWordIndex)):
        if i == 0:
            startIndex = 0
        else:
            startIndex = sentWordIndex[i-1]

        if index1>=startIndex and index1<sentWordIndex[i]:
            sentIndex1Set = True
            sentIndex1 = i-1

        if index2>=startIndex and index2<sentWordIndex[i]:
            sentIndex2Set = True
            sentIndex2 = i-1

        if sentIndex1Set and sentIndex2Set:
            break

    return abs(sentIndex1 - sentIndex2)

def calculateScore(index1,index2,relationType,similarityIndex):
    if relationType == 0 or relationType==1:
        return 10*similarityIndex

    elif relationType == 2:
        return 4*similarityIndex
        

class metaChain:
    """Class for the creation of a meta-chain"""

    score = 0

    def __init__(self,baseWord,baseIndex,baseSense):
        self.connectionsList = []
        
        base = {}
        base['word'] = baseWord
        base['wordIndex'] = baseIndex
        base['sense'] = baseSense
        self.wordsList = [base]
        
    def getTokenContribution(self,nounTuple):
        contribution = 0
        
        for connection in self.connectionsList:
            if connection['sIndex']==nounTuple[1] or connection['eIndex']==nounTuple[1]:
                contribution = contribution + calculateScore(connection['sIndex'],connection['eIndex'],connection['type'],connection['similarityIndex'])

        return contribution

    def removeWord(self,nounTuple):

        for word in self.wordsList:
            if word['word'] == nounTuple[0]:
                self.wordsList.remove(word)
                break

        for connection in self.connectionsList:
            if connection['sIndex']==nounTuple[1] or connection['eIndex']==nounTuple[1]:
                self.score = self.score - calculateScore(connection['sIndex'],connection['eIndex'],connection['type'],connection['similarityIndex'])
                self.connectionsList.remove(connection)

    def addWord(self,word,wordSynsets,index):
        global count
        for chain in metaChainsList:
                
            if chain.wordsList[0] == self.wordsList[0]:

                chainAddData = []
                relatedFlag = False
                wordMatched = False
                
                for wordObj in chain.wordsList:
                    if wordObj['word'] == word and wordObj['wordIndex'] != index:

                        connection = {}
                        connection['sIndex'] = wordObj['wordIndex']
                        connection['eIndex'] = index
                        connection['type'] = 0
                        connection['similarityIndex'] = 1
                        chain.connectionsList.append(connection)
                        self.score = self.score + calculateScore(wordObj['wordIndex'],index,0,1)
                        
                        newWord = {}
                        newWord['word'] = word
                        newWord['wordIndex'] = index
                        newWord['sense'] = wordObj['sense']

                        chain.wordsList.append(newWord)
                        

                        wordMatched = True
                        break

                    elif wordObj['word'] == word and wordObj['wordIndex'] == index:
                        wordMatched = True
                        break
                        
                    else:
                        for syn in wordSynsets:

                            
                            synExistsFlag = False
                            
                            for chainAdd in chainAddData:
                                if chainAdd['syn'] == syn:
                                    synExistsFlag = True
                                    currentSyn = chainAdd
                                    break

                            if synExistsFlag == False:
                                synAddData = {}
                                synAddData['syn'] = syn
                                synAddData['score'] = 0
                                synAddData['connections'] = []
                                currentSyn = synAddData
                                chainAddData.append(synAddData)

                            if syn.pos != 'n' or wordObj['sense'].pos != 'n':
                                continue
                            
                            if wordObj['sense'].wup_similarity(syn) < 0.5:
                                continue
                            
                            if wordObj['sense'].wup_similarity(syn) == 1 and sentDistance(wordObj['wordIndex'],index) <= 7:
                                
                                connection = {}
                                connection['sIndex'] = wordObj['wordIndex']
                                connection['eIndex'] = index
                                connection['type'] = 1
                                connection['similarityIndex'] = wordObj['sense'].wup_similarity(syn)
                                relatedFlag = True
                                currentSyn['connections'].append(connection)
                                currentSyn['score'] = currentSyn['score'] + calculateScore(wordObj['wordIndex'],index,1,connection['similarityIndex'])
                                
                            elif wordObj['sense'].wup_similarity(syn) > 0.8 and sentDistance(wordObj['wordIndex'],index) <= 3:
                                #print wordObj['sense'].wup_similarity(syn)
                                connection = {}
                                connection['sIndex'] = wordObj['wordIndex']
                                connection['eIndex'] = index
                                connection['type'] = 2
                                connection['similarityIndex'] = wordObj['sense'].wup_similarity(syn)
                                relatedFlag = True
                                currentSyn['connections'].append(connection)
                                currentSyn['score'] = currentSyn['score'] + calculateScore(wordObj['wordIndex'],index,2,connection['similarityIndex'])

                if wordMatched == False and relatedFlag:
                    for newChain in chainAddData:
                        if newChain['score'] != 0:
                            chainCopy = metaChain(chain.wordsList[0]['word'],chain.wordsList[0]['wordIndex'],chain.wordsList[0]['sense'])
                            chainCopy.wordsList = list(chain.wordsList)
                            chainCopy.connectionsList = list(chain.connectionsList)
                            chainCopy.score = chain.score
                            
                            newWord = {}
                            newWord['word'] = word
                            newWord['wordIndex'] = index
                            newWord['sense'] = newChain['syn']
                            chainCopy.wordsList.append(newWord)
                            chainCopy.score = chainCopy.score + newChain['score']
                            chainCopy.connectionsList.extend(newChain['connections'])
                            metaChainsList.append(chainCopy)
                            count = count + 1
                            print "chain added", count, time()-start
                      
        

raw = r"Narendra Modi will visit Rajghat on Monday morning to pay homage to Mahatma Gandhi ahead of being sworn in as the 15th Prime Minister of India. Mr Modi, who is scheduled to take the oath of office and secrecy at 6 pm on Wednesday, would visit Rajghat at 7 am, official sources said. Mr Modi's swearing-in will be attended by nearly 3,000 guests, including top leaders from SAARC countries like Pakistan Prime Minister Nawaz Sharif and Sri Lankan President Mahinda Rajapaksa. (Before Narendra Modi's Swearing-in, Speculation on his Cabinet) Outgoing Prime Minister Manmohan Singh, Congress President Sonia Gandhi and party vice-president Rahul Gandhi, besides leaders of various other parties and chief ministers of a number of states will also be attending the function in the forecourt of Rashtrapati Bhavan."
sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

wordTokens = nltk.wordpunct_tokenize(raw)

taggedTokens = tagger.tag(wordTokens)
sentTokens = sentenceTokenizer.tokenize(raw)
sentWordIndex = []
wordIndex = 0

for sent in sentTokens:
    wordIndex = wordIndex + len(sent)
    sentWordIndex.append(wordIndex)

nounsList = []
metaChainsList = []

for w,tag in taggedTokens:
    if tag != None and tag.startswith('N'):
        wordTuple = (w,taggedTokens.index((w,tag)))
        nounsList.append(wordTuple)


print len(nounsList)


for nounTuple in nounsList:
    senses = wn.synsets(nounTuple[0],pos='n')

    for syn in senses:
        print syn
        chain = metaChain(nounTuple[0],nounTuple[1],syn)
        metaChainsList.append(chain)

        for noun in nounsList:
            if noun[1]>nounTuple[1]:
                chain.addWord(noun[0],wn.synsets(noun[0],pos='n'),noun[1])



print "meta chains formed"
        
for nounTuple in nounsList:
    contributionsList = []
    for metaChain in metaChainsList:
        contributionsList.append(metaChain.getTokenContribution(nounTuple))

    maxContriIndex = contributionsList.index(max(contributionsList))

    for i in range(0,len(metaChainsList)):
        if i != maxContriIndex:
            metaChainsList[i].removeWord(nounTuple)


end = time()           

print end-start

chainScores = []
wordSets = []

for chain in metaChainsList:
    length = len(chain.wordsList)
    words = []
    for wordObj in chain.wordsList:
        words.append(wordObj['sense'])

    distinctWords = set(words)
    wordSets.append(distinctWords)
    
    if length != 0:
        homogeneityIndex = 1-len(set(words))/length
    else:
        homogeneityIndex = 0

    chainScores.append(length*homogeneityIndex)


criteria = numpy.mean(chainScores)+ 2*numpy.std(chainScores)

strongChains = []

chosenSents = []

for i in range(0,len(chainScores)):
    if chainScores[i] > criteria:
        for sent in sentTokens:
            wordTokens = nltk.wordpunct_tokenize(sent)
            if len(wordSets[i].intersection(wordTokens)) > 0:
                chosenSents.append(i)
                break



for sent in chosenSents:
    print sentTokens[sent]

