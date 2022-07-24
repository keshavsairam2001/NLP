#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[5]:


import nltk
nltk.download()


# In[ ]:


#Better Word Tokenizers


# In[6]:


text = """" Well, the government is not what it used to be like.It is far more Democratic."""


# In[11]:


import regex
regex.split("[\s\.\,]",text)


# In[12]:


nltk.word_tokenize(text)


# In[ ]:


#Stemming
#Porter Stemmer


# In[14]:


from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
plurals=['caresses','flies','dies','denied','agreed','owned','humbled','sized','meeting','stating']
for word in plurals:
    print(f"{word} >>> {stemmer.stem(word)}")


# In[ ]:


#Snowball Stemmer


# In[15]:


from nltk.stem.snowball import SnowballStemmer
SnowballStemmer.languages


# In[16]:


sn_stemmer=SnowballStemmer("english")


# In[17]:


sn_stemmer.stem("generously")


# In[18]:


stemmer.stem("generously")


# In[ ]:


Lemmatization


# In[19]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[20]:


for word in plurals:
    print(f"{word}>>>{lemmatizer.lemmatize(word)}")


# In[39]:


#tokenizing a sentence
#getting its size
import sys
from nltk.tokenize import word_tokenize
text1=word_tokenize("Hi there!")
sys. getsizeof(text1) 


# In[40]:


#length of tokens
en = text.translate(str.maketrans('', '', '!,.?'))


# In[41]:


len(en)


# In[ ]:




