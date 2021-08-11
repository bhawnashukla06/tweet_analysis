#libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string

from textblob import TextBlob
#text processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


import warnings
warnings.filterwarnings("ignore")


#creating a stopword set
import nltk
#nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


#load dataset

def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

#delete useless columns
def delete_redundant_cols(df,cols):
    for col in cols:
        del df[col]
    return df


#preprocessing the tweet text
def preprocess_tweet_text(tweet):

    #convert all text lowercase
    tweet = tweet.lower()
    #remove any urls
    tweet = re.sub(r"http\S+|www\S+|https\S+","",tweet,flags=re.MULTILINE)
    #remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    #remove user @ references and # from tweet
    tweet = re.sub(r'\@\w+|\#',"",tweet)
    # remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if not word in stop_words]

    #stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]

    #lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w,pos='a') for w in stemmed_words]
    return " ".join(lemma_words)





#print(preprocess_tweet_text('hello i am bhawna shukla interacting with this prgrm '))
dataset = load_dataset("vaccination_all_tweets.csv.", ['id','user_name','user_location','user_description','user_created','user_followers','user_friends','user_favourites','user_verified','date','text','hashtags','source','retweets','favorites','is_retweet'])
# Remove unwanted columns from dataset
n_dataset = delete_redundant_cols(dataset, ['id','user_name','user_location','user_description','user_created','user_followers','user_friends','user_favourites','user_verified','date','hashtags','source','retweets','favorites','is_retweet'])
# Remove unwanted columns from dataset])
#Preprocess data
dataset['tweet'] = dataset['text'].apply(preprocess_tweet_text)

# adding polarity and subjectivty to the dataframe
dataset['polarity'] = dataset['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
dataset['subjectivity'] = dataset['tweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
neg = 0
pos = 0
neut = 0
#function to count number of positive tweet , negative tweet and neutral tweet
def represent(polarity):
    global neg,pos,neut
    if polarity < 0.000000:
        neg = neg + 1
    elif polarity> 0.000000:
        pos = pos + 1
    elif polarity == 0.000000:
        neut = neut + 1
dataset['polarity'].apply(represent)

# printing graphs
y = np.array([pos,neut,neg])
mylabels = ["Positive","Neutral","Negative"]
mycolors = ["#009933","#0080ff","#cc0000"]

# pie chart
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))


def func(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(y, autopct=lambda pct: func(pct, y),
                                  textprops=dict(color="w"))

ax.legend(wedges, mylabels,
          title="tweet analysis",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("tweet analysis")

plt.show()

# bar graphs
plt.bar(mylabels,y)
plt.show()
# polarity histogram
dataset['polarity'].hist()
plt.title("polarity")
plt.show()
# subjectivity histogram
dataset['subjectivity'].hist()
plt.title("subjectivity")
plt.show()
# storing negative tweet in one datafram
neg_tweets = dataset[dataset.polarity <0]
neg_tweets.to_csv("negative_tweets.csv")
neg_string = []
for t in neg_tweets.tweet:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
#neg_string contain all words that are in negative tweets
#print(neg_string)

# word cloud for negative words
from wordcloud import WordCloud
wordcloud1 = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud1, interpolation="bilinear")
plt.title("negative wordcloud")
plt.axis("off")
plt.show()

# storing positive tweets in one data frame
pos_tweets = dataset[dataset.polarity >0]
pos_tweets.to_csv("positive_tweet.csv")
pos_string = []
for t in pos_tweets.tweet:

    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
#pos_string contain all words that are in positive tweets
#print(pos_string)


#word cloud for positive words
from wordcloud import WordCloud
wordcloud2 = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.title("positive wordcloud")
plt.axis("off")
plt.show()


# csv file for negative words with their frequency
#neg_freq = neg_tweets.tweet.str.split(expand=True).stack().value_counts()
#neg_freq.to_csv('neg_words_csv')

# csv file with positive words with their frequency
#pos_freq=pos_tweets.tweet.str.split(expand=True).stack().value_counts()
#pos_freq.to_csv('pos_words_csv')
#dataset.to_csv('data.csv')