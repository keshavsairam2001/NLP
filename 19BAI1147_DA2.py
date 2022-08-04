#!/usr/bin/env python
# coding: utf-8

# # **CSE4022 Natural Language Processing**
# ## ***Digital Assignments 1 & 2***
# ## Name:Keshav Sairam
# ## Reg no:19BAI1147

# ## **Digital Assignment 2**
# 
# 
# ---
# 
# 

# # **Question 1**
# ### Create a text corpus with minimum 200 words (unique contents). Implement the following text processing   
#                                                                                 
# *   Word segmentation
# *   Sentence segmentation
# *   Convert to Lowercase
# *   Stop words removal
# *   Stemming
# *   Lemmatization
# *   Part of speech tagger
#  

# ### **Solution:**

# In[24]:


# Creating the text corpus with 297 words
text = "All of us, at least once in our lives, have experienced our parents hollering at us for sitting idle. When they see us roaming around unnecessarily or sitting without any work, they seemingly ask don't we have better things to do? We always take it as unneeded screaming and fail to realize the deeper meaning it holds. Life spent doing constructive work, is life well spent. The great dramatist Shakespeare rightly observed that life should be measured by deeds, not years. Age is no criterion for the meaning of life. It is the actions, good deeds which give meaning to life and make man immortal. Long life is desired by all, but if one does not do any noble deeds, then such a life has no worth. Great leaders like Mahatma Gandhi, Lal Bahadur Shastri, Abraham Lincoln, Swami Vivekananda, Mother Teresa and many others are remembered even after so many years after their passing. People still take inspiration from their lifestyles and preaching. It is only the great deeds of these leaders that have inspired many generations. A lily flower lives just for a day, but it is remembered for its fragrance and sweetness. Respect is earned by actions and not acquired by years. Existence becomes exciting when it is lived for others or when it does something beneficial to mankind. We generally work for the whole day, earn money and spend it on our needs. These are some common things that all do, but we should all do some noble deeds as well. We should share our smile, advice, cheer, and help with our fellow beings. These things may give them happiness and they may remember us when we are not around. Hence the saying, ""We live in deeds, not in years,"" proves to be true."
text


# ### *Word Segmentation*

# In[46]:


# Word Segmentation (Word Tokenization)
from nltk.tokenize import word_tokenize
wordsList = word_tokenize(text)
print(wordsList)


# ### *Sentence Segmentation*

# In[ ]:


# Sentence Segmentation where each sentence is seperated by a comma (,).
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sent_tokenizer.tokenize(text)
sents


# In[27]:


# METHOD 2 for Sentence Segmentation
from nltk.tokenize import sent_tokenize
print(sent_tokenize(text))


# ### *Convert to Lowercase*

# In[30]:


# We want our model to not get confused by seeing the same word with different cases like one starting with capital and one without and interpret both differently. 
# So we convert all words into the lower case to avoid redundancy in the token list
import re
text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
words = text.split()
print(words)


# ### *Stop Words Removal*

# In[35]:


# When we use the features from a text to model, we will encounter a lot of noise. 
# These are the stop words like the, he, her, etc… which don’t help us and, just be removed before processing for cleaner processing inside the model. 
# With NLTK we can see all the stop words available in the English language.

from nltk.corpus import stopwords
# Remove stop words
words = [w for w in words if w not in stopwords.words("english")]
print(words)


# ### *Stemming*

# In[38]:


# In our text we may find many words like playing, played, playfully, etc… which have a root word, play all of these convey the same meaning. 
# So we can just extract the root word and remove the rest. 
# Here the root word formed is called ‘stem’ and it is not necessarily that stem needs to exist and have a meaning. 
# Just by committing the suffix and prefix, we generate the stems.

# NLTK provides us with PorterStemmer LancasterStemmer and SnowballStemmer packages.

from nltk.stem.porter import PorterStemmer
# Reduce words to their stems
stemmed = [PorterStemmer().stem(w) for w in words]
print(stemmed)


# ### *Lemmatization*

# In[39]:


# We want to extract the base form of the word here. 
# The word extracted here is called Lemma and it is available in the dictionary. 
# We have the WordNet corpus and the lemma generated will be available in this corpus. 
# NLTK provides us with the WordNet Lemmatizer that makes use of the WordNet Database to lookup lemmas of words.

from nltk.stem.wordnet import WordNetLemmatizer
# Reduce words to their root form
lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
print(lemmed)

## NOTE: Stemming is much faster than lemmatization as it doesn’t need to lookup in the dictionary and just follows the algorithm to generate the root words.


# ### *Part of Speech Tagger*

# In[49]:


# Part of Speech tagging is used in text processing to avoid confusion between two same words that have different meanings. 
# With respect to the definition and context, we give each word a particular tag and process them. 
# Two Steps are used here:

# Tokenize text (word_tokenize).
# Apply the pos_tag from NLTK to the above step.

# removing stop words from wordList.
stop_words = set(stopwords.words('english'))


# Using a Tagger. Which is part-of-speech
# tagger or POS-tagger.
wordsList = [w for w in wordsList if not w in stop_words]
tagged = nltk.pos_tag(wordsList)
print(tagged)


# 
# 
# ---
# 
# 
