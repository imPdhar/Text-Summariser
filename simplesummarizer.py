#!/usr/bin/env python
# coding: utf-8

# In[28]:


#Bapu Pruthvidhar Email:talktobpd1@gmail.com 
#Abstract Summarisation
import nltk                              #Imported natural language toolkit as project is based on text
import pandas as pd                      #Imported Pandas for getting file from path and dataframes


from nltk.tokenize import sent_tokenize, word_tokenize #Tokenizer for tokenizing
from nltk.corpus import stopwords #Filtering stop words which affect abstraction 
from string import punctuation #For removing Unintended Punctuation 
from nltk.probability import FreqDist #Calculating frequency 

doc1=open("data.txt","r")     #Acquiring File
doc1txt=doc1.read()
print(doc1txt) #Printing File 


# In[29]:


from string import punctuation                           #Import Filter
txt=''.join(c for c in doc1txt if not c.isdigit())       #Remove Digits
txt=''.join(c for c in txt if c not in punctuation).lower()  #Remove Punctuations and lower case the sentences
print(txt)


# In[30]:


words=nltk.tokenize.word_tokenize(txt)        #Tokenize
fdist=FreqDist(words)   
count_frame=pd.DataFrame(fdist, index=[0]).T           #Form Dataframe
count_frame.columns=['Count']                      
print(count_frame)                            #Print Dataframe


# In[38]:


txt=' '.join([word for word in txt.split() if word not in (stopwords.words('english'))])   #Remove Stopwords using NLTK
words=nltk.tokenize.word_tokenize(txt)                      #Re-Tokenize
fdist=FreqDist(words)   
count_frame=pd.DataFrame(fdist, index=[0]).T             #Create Dataframe 
#print(count_frame)
print (' '.join(words))                                  #Join words based on Sequential Summary 


# 

# In[37]:


import heapq                                   #Import heapq to use values in an array 
summary_sentences = heapq.nlargest(100, fdist, key=fdist.get)   #Use highest frequency for Unordered Summarisation
summary = ' '.join(summary_sentences)                           #Join list 
print(summary)                                                  #Print Unordered Summary


# In[ ]:





# In[ ]:




