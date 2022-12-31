#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
import pickle
import numpy as np
import nltk
import pandas as pd, numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


# In[4]:


class ItemBasedRecommender():
    def __init__(self, ratings_dataframe):
        #Read supplied dataframe into an instance variable called reviews
        self.reviews=ratings_dataframe[['name', 'reviews_rating','reviews_username']]
        self.reviews=self.reviews[-self.reviews.reviews_username.isna()]
        self.reviews['product_id']=self.reviews.groupby(['name']).ngroup()
        
        #Create a separate dataframe of user ids
        self.reviews['user_id']=self.reviews.groupby(['reviews_username']).ngroup()
        self.user_id_mapping=self.reviews[['user_id','reviews_username']].drop_duplicates()

        #Create a separate dataframe of product ids
        self.product_id_mapping=self.reviews[['product_id','name']].drop_duplicates()
        self.reviews[['user_id','product_id','reviews_rating']]
        self.reviews.drop_duplicates(inplace=True)
        self.reviews.rename({'reviews_rating':'rating'}, axis=1, inplace=True)
        
        #If the same user has given the same product 2 ratings - take the average
        for user, product in self.reviews.groupby(['user_id','product_id']).count()[self.reviews.groupby(['user_id','product_id']).count().rating>1].index:
            mean_rating=np.mean(self.reviews[((self.reviews.user_id==user)& (self.reviews.product_id==product))].rating)
            self.reviews=self.reviews[-((self.reviews.user_id==user)& (self.reviews.product_id==product))]
            self.reviews=self.reviews.append({'user_id':int(user),'product_id':int(product),'rating':mean_rating}, ignore_index=True)
        #Averaging logic end        

        #Create the item correlation matrix
        self.data_pivot=self.reviews.pivot(index='user_id', columns='product_id', values='rating').T
        self.mean=np.nanmean(self.data_pivot,axis=1)
        self.subtracted=(self.data_pivot.T-self.mean).T
        
        self.item_correlation=1-pairwise_distances(self.subtracted.fillna(0), metric='cosine')
        self.item_correlation[np.isnan(self.item_correlation)]=0
        self.item_correlation[self.item_correlation<0]=0
        
        #Predict ratings
        self.predicted_ratings=np.dot( (self.data_pivot.fillna(0)).T,self.item_correlation)
        
        self.predicted_ratings=pd.DataFrame(self.predicted_ratings.T)
        self.predicted_ratings['product_id']= self.data_pivot.T.columns
        self.predicted_ratings.set_index('product_id', inplace=True)
        self.predicted_ratings=self.predicted_ratings.T
        
        del(self.data_pivot, self.item_correlation,self.mean, self.subtracted)
        
    def recommend(self,user_name, num_of_items=20):
        'Parameters\n----------\nuser_name: string, required\n The name of the user to search for;\nnum_of_items: integer, optional\n Number of products to display, default is 20;\n\n'
        if(len(self.user_id_mapping[self.user_id_mapping.reviews_username==user_name].user_id)>0):
            user_id=self.user_id_mapping[self.user_id_mapping.reviews_username==user_name].user_id.item()
        else:
            raise Exception('User Not Found!')
        recommended_product_list=self.reviews[self.reviews.product_id.isin(self.predicted_ratings.loc[user_id].T.sort_values(ascending=False).index[:num_of_items])].product_id.unique().tolist()
        return self.product_id_mapping[self.product_id_mapping.product_id.isin(recommended_product_list)].name.tolist()


# In[5]:


#This function will replace the contractions with full text
def contractions(s):
    s = re.sub(r"won't", "will not",s)
    s = re.sub(r"would't", "would not",s)
    s = re.sub(r"wouldn't", "would not",s)
    s = re.sub(r"could't", "could not",s)
    s = re.sub(r"couldn't", "could not",s)
    s = re.sub(r"can\'t", "can not",s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    return s


# In[ ]:


##Please refer to doc string below for description of this function
def sentiment_analysis(prod_list,ratings_dataframe):
    """
    This function takes 20 products and original dataframe as input and does the following:

    filters the original dataframe for 20 products
    Text Pre-processing
    Lemmatization
    Creates TF-IDF vector
    Loads the logistic Regression Model and predicts the sentiment
    Calculates the % of positive sentiment for each product
    Selects the top 5 products based on % of positive sentiment and gives as output

    Parameters
    ----------
    prod_list         : list
        List of top 20 products obtained from Recommendation Engine.
    ratings_dataframe : Dataframe
        Original Dataframe which contains the review data

    Returns
    -------
    final_prod_list
        List with top 5 products filter based on % of positive sentiment.
    """
    prod_df=pd.DataFrame(data=prod_list,columns=['product_name'])
    #filter for top 20 products in the original dataframe and get the corresponding reviews
    df=pd.merge(ratings_dataframe,prod_df,left_on=['name'],right_on= ['product_name'],how='inner') 
    #remove the reviews with null values 
    df = df.dropna(subset=['reviews_title','reviews_text'])
    #create a single column using reviews_title and reviews_text
    df['reviews_final'] = df['reviews_title'] + ' ' + df['reviews_text']
    
    #Text Pre-processing
    df['reviews_final']=df['reviews_final'].apply(lambda x: x.lower()) #convert to lower
    df['reviews_final']=df['reviews_final'].apply(lambda x:  re.sub('[^\w\s]' ,'', x)  )#remove special characters
    df['reviews_final']=df['reviews_final'].apply(lambda x: re.sub( '([\w]*[A-Za-z]*[0-9]+[A-Za-z]+[\w]*)|([\w]*[A-Za-z]+[0-9]+[A-Za-z]*[\w]*)' ,'', x) )#remove alphnumeric characters 
    df['reviews_final']=df['reviews_final'].apply(lambda x:  re.sub(r'\d+' ,'', x)  ) #remove numbers
    df['reviews_final']=df['reviews_final'].apply(lambda x: re.sub(r'http\S+', '', x)) #remove hyperlinks
    df['reviews_final']=df['reviews_final'].apply(lambda x:contractions(x)) #remove contractions
    df['reviews_final']=df['reviews_final'].apply(lambda x: ' '.join([re.sub('[^A-Za-z]+','', x) for x in nltk.word_tokenize(x)])) #remove non-alphabetic characters
    lemmatizer = WordNetLemmatizer() #perform Lemmatization
    df['reviews_final']=df['reviews_final'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))
    X = df['reviews_final']
    vectorizer = pickle.load(open("models/vectorizer.pickle", "rb")) #load the TF-IDF vectorizer
    tfidf_test=vectorizer.transform(X)
    lg_model = pickle.load(open('models/logistic_regression.sav', 'rb')) #load the logistic regression model
    y_test_pred=lg_model.predict(tfidf_test) #Predict the sentiment
    df['predicted_sentiment'] = y_test_pred 
    #calculate % of positive sentiment for each product
    grouped_df=df.groupby('name',as_index=False).agg(positive_sentiment_perc = ("predicted_sentiment", "mean")) 
    grouped_df=grouped_df.nlargest(5,'positive_sentiment_perc') #get top 5 products based on % of positive sentiment
    final_prod_list=grouped_df['name'].to_list()
    return final_prod_list


def get_top5_recommendations(user_name):
    """
    This is a wrapper method for Item based Recommendation and Sentiment analysis. app.py will invoke this method.

    Parameters
    ----------
    user_name         : string
        User name to get top 5 recommended product

    Returns
    -------
    final_prod_list
        List with top 5 products filter based on % of positive sentiment.
    """
    ratings_dataframe = pd.read_csv('dataset/sample30.csv')
    reco_obj = ItemBasedRecommender(ratings_dataframe)
    try:
        prod_list=reco_obj.recommend(user_name,20)
        final_list=sentiment_analysis(prod_list,ratings_dataframe)
        return final_list
    except:
        return None