# twitter_sentiment
### Finding sentiment trends in tweets about a specific topic: A webscraping/NLP data science project (Python)

In this little practice project, I show how tweets about 'synthetic diamonds' have changed in sentiment over a course of 30 days in the summer of 2021. I also show how using different search terms, which objectively all describe 'synthetic diamonds', affected the resulting sentiment.

Summary of steps taken:
* set up Twitter developer account to access the twitter API
* downloaded relevant tweets (using the tweepy package)
* created a Pandas dataframe from the tweets and added columns for
  + full tweet text
  + tweet text cleaned from superfluous punctuation and stopwords
  + user screen name
  + time of creation
  + polarity of tweet text (found using the TextBlob package)
 * visualized the tweets' contents through a wordcloud (using the WordCloud package)
 * visualized most frequent words in a frequency plot (using the FreqDist module of the NLTK package)
 * visualized sentiment development over time in a bar-plot
 * compared sentiment development for several search terms in a line plot 
 
 I'm making the code freely available under a CC0 license. If you have any questions or suggestions for improvement, please get in touch!
