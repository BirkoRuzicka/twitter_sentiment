# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:43:42 2021

@author: Dr. Birko-Katarina Ruzicka
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import tweepy
import re
import string
from textblob import TextBlob
import stopwordsiso
from wordcloud import WordCloud
from nltk.probability import FreqDist
import datetime
import pickle


# =============================================================================
# Get the tweets
# =============================================================================

search_term = 'manufactured+diamonds'

# if you already saved the df as a pickle, use:
# with open(f'{search_term}.pkl', 'rb') as file:
#     df = pickle.load(file)

# load Twitter API credentials
log = pd.read_csv('Login.csv')
consumerKey = log["key"][0]
consumerSecret = log["key"][1]
accessToken = log["key"][2]
accessTokenSecret = log["key"][3]

# create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)

# create the API object while passing in the auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)


# search for tweets
i = input('confirm that you want to perform a twitter search (y/n). \n'
          'this will eat into your "free searches" count. \n'
          'remember to set THE CORRECT DATES for search! \n'
          '>> ')

if i == 'y':
    pass
elif i == 'n':
    raise Exception('ok, search aborted')
else:
    raise Exception('no answer given')

# search by search term:
    
# tweets = api.search_full_archive('diamonds', query=search_term,
#                                   fromDate=202105100000,
#                                   toDate=202106220000)
    
tweets = api.search_30_day('diamonds30', query=search_term,
                           fromDate=202105230000, toDate=202106230000)


# create dataframe and add columns
df = pd.DataFrame([tweet for tweet in tweets], columns=['tweets'])


def get_text(tweet):
    if tweet.truncated is True:
        text = tweet.extended_tweet['full_text']
    else:
        text = tweet.text
    return text
    

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9\_:]+', '', text)  # remove @mentions
    text = re.sub(r'RT[\s]+', '', text)  # remove retweet indicator plus space
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove hyperlinks
    text = text.replace('\n', ' ')
    return text


df['text'] = df['tweets'].apply(get_text)
df['text_cleaned'] = df['text'].apply(clean_text)
df['screen_name'] = [i.user.screen_name for i in df['tweets']]
df['date'] = [i.created_at for i in df['tweets']]
df['week'] = [i.week for i in df['date']]
df['doy'] = [int(i.strftime('%j')) for i in df['date']]
df['polarity'] = [TextBlob(i).sentiment.polarity for i in df['text']]
# note: the polarity analysis treats emojis like their alt text!

# make sure that tweets are in chronological order
df = df.sort_values(by=['date']).reset_index(drop=True)


with open(f'{search_term}.pkl', 'wb') as file:
    pickle.dump(df, file)

    
#%% clean the data and generate wordcloud
    
df = df.dropna(axis=0)
tweetlist = [twts.lower() for twts in df['text_cleaned']]  # list of tweets
wordstring = ' '.join(tweetlist)  # string of tweets
wordlist = [wrd for wrd in wordstring.split(' ')]  # list of words
while '' in wordlist: wordlist.remove('')

wordlist2 = []  # list of words without stopwords, mentions, or links
stopwords = stopwordsiso.stopwords("en")

for word in wordlist:
    if word not in stopwords \
       and not ('@' in word) \
       and not ('http' in word) \
       and not ('&amp;' in word):
        wordlist2.append(word)


wordlist_clean = []  # list of words, stripped of punctuation
for word in wordlist2:
    word_stripped = word.strip(string.punctuation + '…')
    if word_stripped not in stopwords and len(word_stripped) > 2:
        wordlist_clean.append(word_stripped)
    
    
wordstring_clean = ' '.join(wordlist_clean)


stopwords.update(['artificial', 'synthetic', 'diamond', 'diamonds', 'lab',
                  'grown'])

wordCloud = WordCloud(width=1000, height=600, max_font_size=119,
                      stopwords=stopwords, max_words=200,  # 60
                      collocations=False, colormap='cividis')  # cividis
wordCloud.generate(wordstring_clean)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.title(f'WordCloud for search term "{search_term}", May to June 2021')
plt.savefig(f'wordcloud_{search_term} may to june', dpi=150)
plt.show()


#%% word frequency plot

df_freq = pd.DataFrame(wordlist_clean)
df_freq = df_freq[0].value_counts()

freqdoctor = FreqDist()
for words in df_freq:
    freqdoctor[words] += 1

df_freq = df_freq[:15, ]

plt.figure(figsize=(10, 5))
sns.barplot(x=df_freq.values, y=df_freq.index, alpha=0.8, color='#31436d')
plt.title(f'Top words for search term "{search_term}", May to June 2021')
plt.xlabel('Count of Words', fontsize=12)
plt.savefig(f'frequentwords_{search_term} may to june', dpi=150)
plt.show()


#%% time series of polarity

days = list(df['doy'].unique())
dates = []
for day in range(days[0], days[-1] + 1, 5):
    x = datetime.datetime(2021, 1, 1) + datetime.timedelta(int(day) - 1)
    date = x.strftime('%d.%m.')
    dates.append(date)
    
polarity_pos = dict.fromkeys(days, 0)
polarity_neg = dict.fromkeys(days, 0)
count_pos = dict.fromkeys(days, 0)
count_neg = dict.fromkeys(days, 0)

for i in df.index:
    if df['polarity'][i] >= 0:
        polarity_pos[df['doy'][i]] += df['polarity'][i]
        count_pos[df['doy'][i]] += 1
    if df['polarity'][i] < 0:
        polarity_neg[df['doy'][i]] += df['polarity'][i]
        count_neg[df['doy'][i]] += 1

count_all = df['doy'].value_counts().to_dict()
count_n = list(count_neg.values())
count_p = list(count_pos.values())

# tweet count
fig, ax = plt.subplots(figsize=(8, 5))
plt.axhline(y=0, color='black', linewidth=0.2)
plt.bar(x=count_all.keys(), height=count_all.values(),
        edgecolor='black', color='none', alpha=0.8)
ax.set_xlabel('date (2021)')
ax.set_xticks(list(range(days[0], days[-1] + 1, 5)))
ax.set_xticklabels(dates)
ax.set_ylabel('tweets per day')
ax.set_title(f'count of tweets containing "{search_term}", May '
             'to June 2021')
plt.savefig(f'count {search_term} may to june', dpi=150)
plt.show()


# tweet count showing pos/neg distribution
fig, ax = plt.subplots(figsize=(8, 5))
plt.axhline(y=0, color='black', linewidth=0.2)
plt.bar(x=count_neg.keys(), height=count_n,
        color='#FF847C', label='negative tweets')
plt.bar(x=count_pos.keys(), height=count_p, bottom=count_n,
        color='#99B898', label='positive tweets')
ax.set_xlabel('date (2021)')
ax.set_xticks(list(range(days[0], days[-1] + 1, 5)))
ax.set_xticklabels(dates)
# ax.set_ylim([0, max(count_n) * 1.05])
ax.set_ylabel('tweets per day')
ax.set_title(f'count of tweets containing "{search_term}", May '
             'to June 2021')
ax.legend(loc=0)
plt.savefig(f'count pos-neg {search_term} may to june', dpi=150)
plt.show()
        
        
# cummulative polarity
fig, ax = plt.subplots(figsize=(8, 5))
plt.axhline(y=0, color='black', linewidth=0.2)
plt.bar(x=days, height=list(polarity_pos.values()),
        color='#99B898', label='positive tweets')
plt.bar(x=days, height=list(polarity_neg.values()),
        color='#FF847C', label='negative tweets')
ax.set_xlabel('date (2021)')
ax.set_xticks(list(range(days[0], days[-1] + 1, 5)))
ax.set_xticklabels(dates)
abs_max = max(max(polarity_pos.values()), abs(min(polarity_neg.values())))
ax.set_ylim([abs_max * -1.1, abs_max * 1.1])
ax.set_ylabel('cumulative polarity of tweets')
ax.set_title(f'polarity of tweets containing "{search_term}", May '
             'to June 2021')
ax.legend(loc=0)
plt.savefig(f'polarity {search_term} may to june', dpi=150)
plt.show()


#%% visualize trends across search terms

# load all pickles into dataframes, add column 'query'
with open('synthetic+diamonds.pkl', 'rb') as file:
    df1 = pickle.load(file)
    df1['query'] = pd.Series(
        ['synthetic+diamonds' for x in range(len(df1.index))])
with open('artificial+diamonds.pkl', 'rb') as file:
    df2 = pickle.load(file)
    df2['query'] = pd.Series(
        ['artificial+diamonds' for x in range(len(df2.index))])
with open('fake+diamonds.pkl', 'rb') as file:
    df3 = pickle.load(file)
    df3['query'] = pd.Series(
        ['fake+diamonds' for x in range(len(df3.index))])
with open('lab-grown+diamonds.pkl', 'rb') as file:
    df4 = pickle.load(file)
    df4['query'] = pd.Series(
        ['lab-grown+diamonds' for x in range(len(df4.index))])
with open('man-made+diamonds.pkl', 'rb') as file:
    df5 = pickle.load(file)
    df5['query'] = pd.Series(
        ['man-made+diamonds' for x in range(len(df5.index))])
with open('manufactured+diamonds.pkl', 'rb') as file:
    df6 = pickle.load(file)
    df6['query'] = pd.Series(
        ['manufactured+diamonds' for x in range(len(df6.index))])
    
# concatenate all dataframes into one comprehensive dataframe
df_all = df1.append(df2, ignore_index=True)\
    .append(df3, ignore_index=True)\
    .append(df4, ignore_index=True)\
    .append(df5, ignore_index=True)\
    .append(df6, ignore_index=True)
    
    
# create new column 'is_pos'
is_pos = []
for i in df_all.index:
    if df_all['polarity'][i] > 0:
        is_pos.append(1)
    elif df_all['polarity'][i] < 0:
        is_pos.append(-1)
    else:
        is_pos.append(0)
df_all['is_pos'] = is_pos

# create palette for polarity
BlGyGr = ['black', '#bbbbbb', 'limegreen']  # neg neu pos

# point or line plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df_all, x='doy', y='polarity', hue='is_pos',
                palette=BlGyGr, legend=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
ax.set_xlabel('date (2021)')
ax.set_ylabel('tweets')
fig.show()

# lines
fig, ax = plt.subplots(figsize=(9, 5))
g = sns.kdeplot(data=df_all, x="doy", hue="query",
                legend=False)
g.set(xlabel='day of 2021',
      title='Tweets per search term, May to June 2021')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
plt.legend(labels=list(df_all['query'].unique()),
           loc='upper left', title='search terms:')
plt.savefig('kdeplot_all_queries_tweets.png', dpi=200)
plt.show()

# filled
fig, ax = plt.subplots(figsize=(9, 5))
g = sns.kdeplot(data=df_all, x="doy", hue="query", multiple='fill',
                legend=False)
g.set(xlabel='date (2021)', ylabel='% density',
      title='Tweets per search term, May to June 2021')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
plt.legend(labels=list(df_all['query'].unique()),
           loc='upper left', title='search terms:')
plt.show()


# plot polarity over time, all queries
fig, ax = plt.subplots(figsize=(9, 5))
g = sns.kdeplot(data=df_all, x="doy", hue="is_pos", legend=False,
                palette=BlGyGr, linewidth=3)
g.set(xlabel='', title='Polarity of tweets (all search terms), May '
      'to June 2021')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
plt.legend(labels=['positive', 'neutral', 'negative'],
           loc='upper left')
plt.savefig('kdeplot_all_queries_polarity.png', dpi=200)
plt.show()

    
# facet grid of polarity trends for each query
plt.subplots(figsize=(12, 8))
g = sns.FacetGrid(df_all, col="query", col_wrap=3)
g.map_dataframe(sns.kdeplot, x="doy", hue="is_pos", palette=BlGyGr)
g.set_axis_labels("day of 2021", "Density")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Polarity of tweets, May to June 2021')
g.fig.tight_layout()
plt.legend(labels=['positive', 'neutral', 'negative'],
           loc='upper right', bbox_to_anchor=(1.5, 2.33))
plt.savefig('facet_grid_polarity.png', dpi=200,
            bbox_inches='tight')  # prevents cuting off legend (outside axes)
plt.show()


# composite count-polarity
sns.jointplot(data=df_all, x="doy", y="polarity", hue="query")
sns.jointplot(data=df_all, x="doy", y="polarity", kind='hex')
sns.jointplot(data=df_all, x="doy", y="polarity", kind='kde', hue='query')
sns.jointplot(data=df_all, x="doy", y="polarity")


pass
