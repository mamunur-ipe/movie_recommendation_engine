# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Importing the dataset
df = pd.read_csv('imdb_5000.csv')

# show column names
columns = list(df.columns)
print(columns)

# show categorical columns
numeric_columns = list(df.describe().columns)
categorical_columns = list( set(columns) - set(numeric_columns) )
print(categorical_columns)

# numeric and categorical dataframes
df_numeric = df[numeric_columns]
df_categorical = df[categorical_columns]

#-----------------------------------------------------------------------------------------
## create some initial plots

#01. histograms for numeric data
plt.style.use('seaborn')
df.hist(figsize=[10,8])
plt.show()

#02. pairplot using seaborn for the numeric data
import seaborn as sns
# sns.set(style="white", color_codes=True)
sns.pairplot(df)


#03. create correlation heat-map
plt.style.use('seaborn-ticks')
from matplotlib.pyplot import cm
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(8,8))
gs = GridSpec(1, 1, figure=fig)
X = df_numeric
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(X.corr(), cmap=cm.Reds)

ax1.set_xticks(range(len(X.columns)))
ax1.set_xticklabels(X.columns.values.tolist(), size=12)
for label in ax1.get_xmajorticklabels():
    label.set_rotation(90)
    label.set_horizontalalignment("center")  # 'center', right', 'left'
    
ax1.set_yticks(range(len(X.columns)))
ax1.set_yticklabels(X.columns.values.tolist(), size=12)

# Loop over data dimensions and create text annotations.
for i in range(len(X.columns)):
    for j in range(len(X.columns)):
        text = ax1.text(j, i, df.corr().values.round(2)[i, j],
                       ha="center", va="center", color="w")

# fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.2 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(im, cax=cax)

ax1.set_title('Correlation heatmap for numeric features', fontweight="bold", fontsize=14)

#-----------------------------------------------------------------------------------------
# drop some of the columns those I don't want to include
df.drop(columns=['num_critic_for_reviews','director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross', 'cast_total_facebook_likes', 'actor_3_name', 'facenumber_in_poster', 'movie_imdb_link', 'num_user_for_reviews', 'country', 'budget', 'actor_2_facebook_likes', 'aspect_ratio', 'movie_facebook_likes'], inplace=True)
df['color'].value_counts()
df.drop(columns=['color'], inplace=True)
# reset the index
df.reset_index(drop=True, inplace=True)

# show the percentage of missing values
df_missing_stat = df.isnull().sum().reset_index()
# set column names
df_missing_stat.columns = ['Column title', 'Missing value count']
# add a column for missing value in percentage
df_missing_stat['Missing value in percentage'] = df_missing_stat['Missing value count']/len(df)*100
print(df_missing_stat)
# show descriptive stats for the numerical data
print(df.describe().transpose())

# fill numeric column nan values with mean
mean = df[['duration', 'title_year']].mean().astype(int)
df[['duration', 'title_year']] = df[['duration', 'title_year']].fillna(mean)

# fill rest of the columns (categorical) nan values with 'others'
df.fillna('unknown', inplace=True)

#-----------------------------------------------------------------------------------------
## drop nan values
#df.dropna(inplace=True)
## reset the index
#df.reset_index(drop=True, inplace=True)

# check duplicate movie titles
df_duplicate = df[df.duplicated(['movie_title'], keep=False)].sort_values(by = ['movie_title'])
df_duplicate.head()

# remove the duplicate movie titles
df.drop_duplicates(subset=['movie_title'], keep='first', inplace=True)
# reset the index
df.reset_index(drop=True, inplace=True)

#-----------------------------------------------------------------------------------------

# create wordcloud for some of the important categorical columns
from wordcloud import WordCloud, STOPWORDS

#01. create a function which will accumulate all the words to form a very big sentence/text
def generate_text(df_column):
    '''
    parameters:
            df_column: particulat column of the df for which we want to perform one-hot coding

    return: 
            single sentence containing all the words
    '''
    column = list(df_column)
    
    # remove '|' from the elements of the columns
    column = [x.replace('|', ' ') for x in column]
    #create empty string
    text = ''
    for row in column:
        text = row + ' ' + text
    return text

#02. create wordcloud for the target columns
columns = ['genres', 'plot_keywords', 'actor_1_name', 'director_name', 'country', 'language', 'content_rating']
for column in columns:
    # create text for wordcloud
    text = generate_text(df[column])

    wordcloud = WordCloud(width = 1000, height = 600, 
                    background_color ='black',
                    max_words = 50,
#                     stopwords = STOPWORDS,
                    min_font_size = 10)

    wordcloud.generate(text)
    # plot the WordCloud image                        
    plt.figure(figsize = (10, 6), facecolor = None)
    plt.title(column, fontsize=20, fontweight='bold')
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()    
#-----------------------------------------------------------------------------------------
'''
# create weighted rank column
We will create a new column 'weighted rank' of a movie as per below formula. IMDB follows 
the below formula for ranking their movies
ref: https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV#

weighted rank (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C
 where:
  R = 'imdb_score'
  v = 'num_voted_users'
  m = 25,000 #minimum number of votes required to be listed
  C = the mean vote across the whole report (7.21 as per this dataset)
'''

r_column = df['imdb_score']
m = 25000
c = df['num_voted_users'].dot(df['imdb_score'])/df['num_voted_users'].sum()
for idx, value in enumerate(r_column):
    r = value
    v = df['num_voted_users'][idx]
    wr = (v/(v+m))*r + (m/(v+m))*c
    df.loc[idx, 'weighted_rank'] = wr

#drop 'imdb_score' and 'num_voted_users' columns
df.drop(columns = ['imdb_score', 'num_voted_users'], inplace= True)

#-----------------------------------------------------------------------------------------

'''THIS PORTION NOT REQUIRED'''
## create a vocabulary of unique keywords from pandas dataframe
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

def df_to_vocabulary(df):
    genres = list(df.values)    
    genres = [x.replace('|', ' ') for x in genres]   
    # break sentences into words and append as list
    i =0
    for sentence in genres:
        words = word_tokenize(sentence)
        # remove stopwards
        sw = stopwords.words('english')
        new_list = [w for w in words if not w in sw]
        # stem words
        stemmed_list = [PorterStemmer().stem(w) for w in new_list]
        genres[i] = stemmed_list
        i += 1        
    # create a vocabulary of all available words in genres
    genres_vocabulary = set()
    for x in genres:
        genres_vocabulary = genres_vocabulary.union(set(x))        
    genres_vocabulary = list(genres_vocabulary)
    genres_vocabulary.sort()
    return genres_vocabulary


vocabulary_genres = df_to_vocabulary(df['genres'])

vocabulary_plot_keywords = df_to_vocabulary(df['plot_keywords'])
# the first 115 values are removed from the list since those are numeric and non-meaningful
vocabulary_plot_keywords = vocabulary_plot_keywords[115:]
# remove entry if it's just a single letter 
for x in vocabulary_plot_keywords:
    if len(x) <= 1:
        vocabulary_plot_keywords.remove(x)

#-----------------------------------------------------------------------------------------
# one-hot encoding for the columns 'genres' and 'plot_keywords'
## create a function which will one-hot encode an input column (pandas series)
# SciKit one-hot encoder can not be applied for the columns 'genres' or 'plot_keywords' 
# because of their natures because, for example, a cell of 'genres' column contains 
# multiple entries, e.g, comedy, sci-fi, drama. Therefore, the following function is developed.
def one_hot_encode(df, df_column, threshold=10, plot=False):
    '''
    parameters:
            df: main data frame,
            df_column: particulat column of the df for which we want to perform one-hot coding
            vocabulary: a set of all words available in the column
            threshold: either in percentage factor (0-1) or number of top frequent elemnts(greater than 1)
            plot: if True, a bar plot of the top frequent words will be generated
    return: 
            no. of top frequent words
            high frequency words
            frequency percentage of the high frequency words
    '''
    column = list(df_column)
    # remove '|' from the elements of the columns
    column = [x.replace('|', ' ') for x in column]
    
    # import necessary libraries
    from nltk.tokenize import word_tokenize 
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer 
    i =0
    for row in column:
        # seperate the words and make lists
        words = word_tokenize(row)
        # remove stopwards
        sw = stopwords.words('english')
        new_list = [w for w in words if not w in sw]
        # stem words
        stemmed_list = [PorterStemmer().stem(w) for w in new_list]
        column[i] = stemmed_list
        i += 1
    
    ## get the top elements (pareto analysis)
    #01. create a mega list which will contain all the elements of the column
    mega_list = []
    for row in column:
        for word in row:
            mega_list.append(word)
    df1 = pd.DataFrame(mega_list, columns = ['mega_list'])
    
    #02. calculate frequency percentage of all the unique elements
    frequency_percent = df1['mega_list'].value_counts()/len(df1)*100
    
    #03. find the top frequent elements
    high_frequency_elements = []
    y = []
    summation = 0
    for x in list(frequency_percent.index):
        high_frequency_elements.append(x)
        y.append(frequency_percent[x])
        # if threshold in percentage
        if threshold <= 1:
            summation += (frequency_percent[x]/100)
        # if threshold not in percentage, top frequent element numbers
        else:
            summation += 1
        if summation >= threshold:
            break
    
    # plot the frequency percetage of the top elements   
    if plot == True:
        import matplotlib.pyplot as plt
        plt.bar(high_frequency_elements, y)
        plt.xticks(rotation=90)
        
    #04. perform one-hot encoding of the column
    i =0
    for row in column:
        for x in high_frequency_elements:
            col_name = f"{df_column.name}_{x}"
            if x in row:
                df.loc[i, col_name] = 1
            else:
                df.loc[i, col_name] = 0
        i += 1
    if threshold <= 1:
        return len(high_frequency_elements), high_frequency_elements, y
    else:
        return sum(y), high_frequency_elements, y

# one-hot code "genre" column
no_of_genres, genres_high_frequency_elements, genres_y = one_hot_encode(df, df['genres'], threshold=24, plot=False)
df.drop(columns=['genres'], inplace=True)
## one-hot code "plot_keywords" column
#cum_percentage, keywords_high_frequency_elements, keywords_y = one_hot_encode(df, df['plot_keywords'], threshold=10, plot=False)
#df.drop(columns=['plot_keywords'], inplace=True)
df.drop(columns=['plot_keywords'], inplace=True)

#-----------------------------------------------------------------------------------------
# perform one-hot encoding for rest of the categorical variables
def one_hot_encode(df, df_column, threshold=20, plot=False):
    '''
    parameters:
            df: main data frame,
            df_column: particulat column of the df for which we want to perform one-hot coding
            vocabulary: a set of all words available in the column
            threshold: either in percentage factor (0-1) or number of top frequent elemnts(greater than 1)
            plot: if True, a bar plot of the top frequent words will be generated
    return: 
            no. of unique words in that column
            high frequency words
            frequency percentage of the high frequency words
    '''
    
    #01. calculate frequency percentage of all the unique elements
    frequency_percent = df_column.value_counts()/len(df)*100
    
    #02. find the top frequent elements
    high_frequency_elements = []
    y = []
    summation = 0
    for x in list(frequency_percent.index):
        high_frequency_elements.append(x)
        y.append(frequency_percent[x])
        # threshold in percentage
        if threshold <= 1:
            summation += frequency_percent[x]/100
        # threshold not in percentage, top frequent element numbers
        else:
            summation += 1
        if summation >= threshold:
            break
      
    # plot the frequency percetage of the top elements   
    if plot == True:
        import matplotlib.pyplot as plt
        plt.bar(high_frequency_elements, y)
        plt.xticks(rotation=90)
        
    #03. perform one-hot encoding of the column
    i =0
    for row in df_column:
        for x in high_frequency_elements:
            col_name = f"{df_column.name}_{x}"
            if x in row:
                df.loc[i, col_name] = 1
            else:
                df.loc[i, col_name] = 0
        i += 1
    if threshold <= 1:
        return len(high_frequency_elements), high_frequency_elements, y
    else:
        return sum(y), high_frequency_elements, y

# show value counts for the below columns
target_columns = ['director_name', 'actor_1_name', 'actor_2_name', 'language', 'content_rating']
for column in target_columns:
    print(column)
    print(df[column].value_counts())
    print('\n')

# perform one-hot encoding
#cum_percentage, high_frequency_elements, percent_value = one_hot_encode(df, df['director_name'], threshold=20, plot=False)
df.drop(columns=['director_name'], inplace=True)
cum_percentage, high_frequency_elements, percent_value = one_hot_encode(df, df['actor_1_name'], threshold=20, plot=False)
df.drop(columns=['actor_1_name'], inplace=True)
cum_percentage, high_frequency_elements, percent_value = one_hot_encode(df, df['actor_2_name'], threshold=20, plot=False)
df.drop(columns=['actor_2_name'], inplace=True)
cum_percentage, high_frequency_elements, percent_value = one_hot_encode(df, df['language'], threshold=2, plot=False)
df.drop(columns=['language'], inplace=True)
# replace PG-13' with 'PG'
df['content_rating'].replace('PG-13', 'PG', inplace=True)
cum_percentage, high_frequency_elements, percent_value = one_hot_encode(df, df['content_rating'], threshold=2, plot=False)
df.drop(columns=['content_rating'], inplace=True)

#-----------------------------------------------------------------------------------------

# Save the unscaled dataframe using pickle
import pickle
pickle.dump(df, open('df_before_scaling.pkl','wb'))
df_before_scaling = pickle.load(open('df_before_scaling.pkl','rb'))

#-----------------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler
# scale data
x1 = df_before_scaling.drop(columns=['movie_title', 'weighted_rank']).values
scaler = MinMaxScaler().fit(x1)
# numeric columns
scaled_data = scaler.transform(x1)
#pickle scaled data
pickle.dump(scaled_data, open('scaled_data.pkl','wb'))

# create a movie list with lower case and no space between strings
movie_list=[]
for value in df_before_scaling['movie_title'].tolist():
    name = value.strip()
    name = name.lower().replace(' ', '').replace(':', '').replace('\'', '').replace('-', '')
    movie_list.append(name)

#pickle the movie_list
pickle.dump(movie_list, open('movie_list.pkl','wb'))

# create a dataframe with columns movie_title and weighted_rank
df_name_and_weighted_rank = df_before_scaling[['movie_title', 'weighted_rank']]
#pickle df_name_and_weighted_rank
pickle.dump(df_name_and_weighted_rank, open('df_name_and_weighted_rank.pkl','wb'))

# define a function which will recommend best 'n' movies for a specific user
def recommend_movie(original_user_input, n=5):
    user_input = original_user_input.strip()
    user_input = user_input.lower().replace(' ', '').replace(':', '').replace('\'', '').replace('-', '')
    # get index of the movie from the movie_list
    movie_list = pickle.load(open('movie_list.pkl','rb'))
    idx = movie_list.index(user_input)
    # unpickle scaled data
    scaled_data = pickle.load(open('scaled_data.pkl','rb'))
    # calculate cosine similarity for the user input movie
    similarity_matrix = cosine_similarity(scaled_data, scaled_data[idx].reshape(1, -1))
    # get the index of the top 10 movies similar to user_input
    # the index 0 contains the user input movie. So, we start the index from 1
    idx = list(np.argsort(-similarity_matrix.flatten())[1:10])
    # reload df_name_and_weighted_rank
    df_name_and_weighted_rank = pickle.load(open('df_name_and_weighted_rank.pkl','rb'))  
    df_top_10_by_type = df_name_and_weighted_rank.loc[idx]
    # sort the list by weighted rank and show the first 5 movies
    result = df_top_10_by_type.sort_values(by=['weighted_rank'], ascending=False)['movie_title'][:n].tolist()
    return result

## Run the recommendation engine for user input
# user input
original_user_input = 'mr bean'

try:
    try:
        recommendations = recommend_movie(original_user_input)
        print(f"Since you liked {original_user_input}, our recommendations are: \n ")
        for x in recommendations:
            print(x)
    except:
        import difflib
        close_match = difflib.get_close_matches(original_user_input, movie_list, n=1)[0]
        user_input = close_match
        idx = movie_list.index(user_input)
        df_name_and_weighted_rank = pickle.load(open('df_name_and_weighted_rank.pkl','rb'))
        movie_title = df_name_and_weighted_rank.loc[idx, 'movie_title']
        print(f"Did you mean- {movie_title.strip()}? \nIf yes, our recommendations are:\n")
        recommendations = recommend_movie(close_match)
        for x in recommendations:
            print(x)
        
except:
    print("Sorry!! The movie is not in our database. Please try another movie or keyword.")
    



