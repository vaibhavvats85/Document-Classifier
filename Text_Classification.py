
# coding: utf-8

# In[1]:


import pickle

filename = 'C:/Users/500040117/desktop/Models/finalized_model.sav'
classifier=pickle.load( open(filename, 'rb'))

filename_1 = 'C:/Users/500040117/desktop/Models/finalized_model_1.sav'
transformer=pickle.load( open(filename_1, 'rb'))

filename_2 = 'C:/Users/500040117/desktop/Models/finalized_model_2.sav'
tfidf_transformer=pickle.load( open(filename_2, 'rb'))




# In[8]:


text=[input(">")]

X_new_counts = transformer.transform(text)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = classifier.predict(X_new_tfidf)
print('You maybe talking about '+''.join(predicted))

