# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 00:02:41 2019

@author: Venky
"""

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk import RegexpParser

text = "God is Great! I won a lottery."
#Word Tokeization
word_token = word_tokenize(text)
print(word_token)
sentance_token = sent_tokenize(text)
print(sentance_token)

#Tagging & Chunking with NLTK
split_text = text.split()
#POS Tagging
tokens_tag = pos_tag(split_text)
print(tokens_tag)

#Chunking
patterns= """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
chunker = RegexpParser(patterns)
output = chunker.parse(tokens_tag)
print(output)

#POS in Graph
tag = pos_tag(word_token)
print(tag)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp  = RegexpParser(grammar)
result = cp.parse(tag)
print(result)
result.draw()    # It will draw the pattern graphically which can be seen in Noun Phrase chunking 

#Stemming
ps = PorterStemmer()
for w in word_token:
	rootWord=ps.stem(w)
	print(rootWord)
    
#Lemmatization
import nltk
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()	
for w in word_token:
	print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))  
    
#WordNet with NLTK: Finding Synonyms for words
from nltk.corpus import wordnet
syns = wordnet.synsets("dog")
print(syns)

synonyms = []
antonyms = []

for syn in wordnet.synsets("happy"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			 antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


#Frequency
a = "Guru99 is the site where you can find the best tutorials for Software Testing     Tutorial, SAP Course for Beginners. Java Tutorial for Beginners and much more. Please     visit the site guru99.com and much more."
words = word_tokenize(a)
fd = nltk.FreqDist(words)
fd.plot()

#Bigrams
text = "Guru99 is a totally new kind of learning experience."
Tokens = word_tokenize(text)
output = list(nltk.bigrams(Tokens))
print(output)

#Trigrams
output2 = list(nltk.trigrams(Tokens))
print(output2)