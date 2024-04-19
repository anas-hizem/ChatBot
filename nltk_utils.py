import nltk 
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import  PorterStemmer
stemmmer = PorterStemmer()

def tokenize(sentance) :
    return nltk.word_tokenize(sentance)

def stem(word): #Stemming is the process of reducing a word to its root or base form, typically by removing suffixes and prefixes.
    return stemmmer.stem(word.lower())

def bag_of_words(tokenize_sentance , all_words):
    tokenize_sentance = [stem(w) for w in tokenize_sentance]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenize_sentance:
            bag[idx]+=1
    return bag

# sentance = ["hello" , "how" , "are" , "you"]
# words = ["hi" , "hello" , "I" , "you", "thank" , "cool"]
# bog = bag_of_words(sentance,  words)
# print(f'bag of words is {bog}')


# ch = "Hello my friends !!"
# print(ch)
# ch = tokenize(ch)
# print(ch)


# words = ["organize" , "organizing" , "organizes"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)