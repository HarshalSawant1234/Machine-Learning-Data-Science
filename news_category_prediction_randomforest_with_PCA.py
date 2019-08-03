#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

dataset = pd.read_excel('Data_Train.xlsx',delimiter='\t',quoting=3)
testing_data = pd.read_excel('Data_Test.xlsx',delimiter='\t',quoting=3)
#quoting = 3 for ignoring the double quotes


# In[2]:


import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords # corpus implies collection of words of same type
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,7628):
    review = re.sub('[^a-zA-Z]',' ',dataset['STORY'][i])
    # re.sub function with ^ only keeps character and removes all the punctuation
    #' '= space so that two words dont joint
    review = review.lower()
    # lowercase character
    review = review.split()
    # to convert string into list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # set for faster execution
    # checking irrelevant words in stopwords e.g this
    # PorterStemmer converts 'loved' into 'love' into their original stem

    review = ' '.join(review) 
    # convert the list into string again. # ' ' - so that words dont join
    corpus.append(review) # to append the review to the corpus list


# In[3]:


#Creating Bag of Words Model for training set
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 14141) # include most frequent 1500 words 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


# In[12]:


#Reducing the dimentionalilty of training set by PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2748)
X = pca.fit_transform(X)


# In[15]:


#Randon Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)


# In[6]:


# Creatign Bag of Models for Test Set
corpus1 = []
for i in range(0,2748):
    review1 = re.sub('[^a-zA-Z]',' ',testing_data['STORY'][i])
    review1 = review1.lower()
    review1 = review1.split()
    ps1 = PorterStemmer()
    review1 = [ps1.stem(word1) for word1 in review1 if not word1 in set(stopwords.words('english'))]
    review1 = ' '.join(review1) 
    corpus1.append(review1) # to append the review to the corpus list
    

cv1 = CountVectorizer() # include most frequent 1500 words 
X_test = cv1.fit_transform(corpus1).toarray()
# Naive Bayes , Decision Tree Classification , Random Forest Classification frequently used
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test , y_pred)
#accuracy = classifier.score(X_test, y_test)
#print("accuracy",accuracy)


# In[8]:


#Dimentionality Reduction of test set
#Reducing the dimentionalilty by PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_test = pca.fit_transform(X_test)


# In[16]:


y_pred = classifier.predict(X_test)


# In[18]:


submission = pd.DataFrame({'SECTION':y_pred})
filename = 'News_prediction_3.xlsx'
submission.to_excel(filename)
print('Saved file: ' + filename)


# In[14]:


X.shape


# In[11]:





# In[ ]:




