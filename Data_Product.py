import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px

# Import data
# Read pre-processed data.
@st.cache
def load_dataset(data):
    dataset = pd.read_csv(data)
    return dataset

sunway_df_processed = load_dataset('sunway_df_processed.csv')
taylor_df_processed = load_dataset('taylor_df_processed.csv')
ucsi_df_processed = load_dataset('ucsi_df_processed.csv')
um_df_processed = load_dataset('um_df_processed.csv')
upm_df_processed = load_dataset('upm_df_processed.csv')
ukm_df_processed = load_dataset('ukm_df_processed.csv')

# Split each dataset by sentiment.
sunway_positive = sunway_df_processed.loc[sunway_df_processed['Sentiment'] == 'positive']
sunway_negative = sunway_df_processed.loc[sunway_df_processed['Sentiment'] == 'negative']
sunway_neutral = sunway_df_processed.loc[sunway_df_processed['Sentiment'] == 'neutral']
taylor_positive = taylor_df_processed.loc[taylor_df_processed['Sentiment'] == 'positive']
taylor_negative = taylor_df_processed.loc[taylor_df_processed['Sentiment'] == 'negative']
taylor_neutral = taylor_df_processed.loc[taylor_df_processed['Sentiment'] == 'neutral']
ucsi_positive = ucsi_df_processed.loc[ucsi_df_processed['Sentiment'] == 'positive']
ucsi_negative = ucsi_df_processed.loc[ucsi_df_processed['Sentiment'] == 'negative']
ucsi_neutral = ucsi_df_processed.loc[ucsi_df_processed['Sentiment'] == 'neutral']
um_positive = um_df_processed.loc[um_df_processed['Sentiment'] == 'positive']
um_negative = um_df_processed.loc[um_df_processed['Sentiment'] == 'negative']
um_neutral = um_df_processed.loc[um_df_processed['Sentiment'] == 'neutral']
upm_positive = upm_df_processed.loc[upm_df_processed['Sentiment'] == 'positive']
upm_negative = upm_df_processed.loc[upm_df_processed['Sentiment'] == 'negative']
upm_neutral = upm_df_processed.loc[upm_df_processed['Sentiment'] == 'neutral']
ukm_positive = ukm_df_processed.loc[ukm_df_processed['Sentiment'] == 'positive']
ukm_negative = ukm_df_processed.loc[ukm_df_processed['Sentiment'] == 'negative']
ukm_neutral = ukm_df_processed.loc[ukm_df_processed['Sentiment'] == 'neutral']


# Define functions.
# Function for count_values in single column.
def count_values_in_column(df,columns):
    count = df.loc[:,columns].value_counts(dropna=False)
    percentage = round(df.loc[:,columns].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([count,percentage],axis=1,keys=['Count','Percentage'])


# Function for Word Cloud.
@st.cache
def create_wordcloud(text, imagelink):
    mask = np.array(Image.open('Image/cloud.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(['university', 'universiti', 'student', 'malaysia', 'students', 'sunway', 'taylor', 'will', 's', 'us', 'sunday',
                     'school', 'sunwayuniversity', 'taylorsuniversity', 'ucsiuniversity', 'universitimalaya', 'amp', 'u', 
                      'putra', 'upm', 'ucsi', 'um', 'embolus', 'malaya', 'one', 'lakeside', 'malay', 'put', 'universitytaylor',
                     'pushkin'])
    wc = WordCloud(background_color='white', mask = mask, max_words=3000, 
                   stopwords=stopwords, repeat=False)
    text = ''.join(word for word in text)
    wc.generate(text)
    wc.to_file(imagelink)

# Function for mean of number of likes filtered by sentiment.
def mean_number_of_likes_by_sentiment(df):
    df = df.loc[:,['Number_of_Likes', 'Sentiment']]
    meandf = df.groupby(['Sentiment']).mean()
    meandf = meandf.Number_of_Likes.apply(lambda x : round(x,2))
    meandf = pd.DataFrame(meandf)
    meandf.rename(columns={'Number_of_Likes':'Mean_of_Number_of_Likes'}, inplace = True)
    
    return meandf
    
    

    
    
# Streamlit 
# Add a title
st.title('Sentiment Analysis for Malaysian Universities')



# Sidebar

with st.sidebar:
    st.markdown('**University Selection**')
             
uni = st.sidebar.selectbox(
    'Which Malaysian university you would like to view?',
    ('','Sunway University', 'Taylor\'s University', 'UCSI University', 
    'University of Malaya (UM)', 'University of Putra Malaysia (UPM)',
    'National University of Malaysia (UKM)'))

if uni == '':
    st.subheader('This is the data product for Alan\'s MDS Research Project (WQD7002).:sunglasses:')
    st.write('Kindly select a university at the sidebar to review.')
    imageuni = Image.open('Image/university.png')
    st.image(imageuni, caption='Image collected from Freepik')

if uni == 'Sunway University':
    
    st.write('You selected:', uni)
    
    # Pie Chart for Percentages of Sentiments
    df_sen = count_values_in_column(sunway_df_processed,'Sentiment') 
    fig_sen = px.pie(df_sen, values='Percentage', names= df_sen.index,
             title='Percentage of Sentiment for Sunway University',
             hover_data=['Count'], labels={'Count':'Count of Sentiments'})
    fig_sen.update_traces(textposition='inside', textinfo='percent+label')
    
    
    # Word Cloud
    # Positive Sentiment
    create_wordcloud(sunway_positive.Tokenized_Tweet.values, 'Image/Sunway_Positive_wc.png')
    image_pos = Image.open('Image/Sunway_Positive_wc.png')
    # Negative Sentiment
    create_wordcloud(sunway_negative.Tokenized_Tweet.values, 'Image/Sunway_Negative_wc.png')
    image_neg = Image.open('Image/Sunway_Negative_wc.png')
    # Neutral Sentiment
    create_wordcloud(sunway_neutral.Tokenized_Tweet.values, 'Image/Sunway_Neutral_wc.png')
    image_neu = Image.open('Image/Sunway_Neutral_wc.png')
    
    
    # Pie Chart for Source of Tweets
    df_sot = count_values_in_column(sunway_df_processed,'Source_of_Tweet')
    fig_sot = px.pie(df_sot, values='Percentage', names= df_sot.index,
             title='Percentage of Source of Tweets for Sunway University',
             hover_data=['Count'], labels={'Count':'Count of Souce of Tweets'})
    fig_sot.update_traces(textposition='inside', textinfo='percent+label')
    
    
    
    # Bar Chart for Mean of Number of Likes of Tweets
    df_mnumlike = mean_number_of_likes_by_sentiment(sunway_df_processed)
    sentiment = df_mnumlike.index
    mean = df_mnumlike['Mean_of_Number_of_Likes']
    fig_mnumlike = px.bar(df_mnumlike, x=mean,y=sentiment,
                          title='Mean of Number of Likes of Tweets for Sunway University', color='Mean_of_Number_of_Likes',
                          labels={'Mean_of_Number_of_Likes':'Mean Number of Likes'}, height=500, width=900, orientation='h')
    

    
    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Percentages of Sentiments', 'Word Clouds', 'Source of Tweets', 'Mean of Number of Likes of Tweets'])
    with tab1:
        st.plotly_chart(fig_sen, theme="streamlit", use_conatiner_width=True)
    with tab2:
        st.image(image_pos, caption='Word cloud of Tweets with Positive Sentiment for Sunway University')
        st.image(image_neg, caption='Word cloud of Tweets with Negative Sentiment for Sunway University')
        st.image(image_neu, caption='Word cloud of Tweets with Neutral Sentiment for Sunway University')
    with tab3:
        st.plotly_chart(fig_sot, theme="streamlit", use_conatiner_width=True)
    with tab4:
        st.plotly_chart(fig_mnumlike, theme="streamlit", use_conatiner_width=True)
    

    
elif uni == 'Taylor\'s University':
    
    st.write('You selected:', uni)
    
    # Pie Chart for Percentages of Sentiments
    df_sen = count_values_in_column(taylor_df_processed,'Sentiment') 
    fig_sen = px.pie(df_sen, values='Percentage', names= df_sen.index,
             title='Percentage of Sentiment for Taylor\'s University',
             hover_data=['Count'], labels={'Count':'Count of Sentiments'})
    fig_sen.update_traces(textposition='inside', textinfo='percent+label')
    
    
    # Word Cloud
    # Positive Sentiment
    create_wordcloud(taylor_positive.Tokenized_Tweet.values, 'Image/Taylor_Positive_wc.png')
    image_pos = Image.open('Image/Taylor_Positive_wc.png')
    # Negative Sentiment
    create_wordcloud(taylor_negative.Tokenized_Tweet.values, 'Image/Taylor_Negative_wc.png')
    image_neg = Image.open('Image/Taylor_Negative_wc.png')
    # Neutral Sentiment
    create_wordcloud(taylor_neutral.Tokenized_Tweet.values, 'Image/Taylor_Neutral_wc.png')
    image_neu = Image.open('Image/Taylor_Neutral_wc.png')
    
    
    # Pie Chart for Source of Tweets
    df_sot = count_values_in_column(sunway_df_processed,'Source_of_Tweet')
    fig_sot = px.pie(df_sot, values='Percentage', names= df_sot.index,
             title='Percentage of Source of Tweets for Sunway University',
             hover_data=['Count'], labels={'Count':'Count of Souce of Tweets'})
    fig_sot.update_traces(textposition='inside', textinfo='percent+label')
    
    
    
    # Bar Chart for Mean of Number of Likes of Tweets
    df_mnumlike = mean_number_of_likes_by_sentiment(taylor_df_processed)
    sentiment = df_mnumlike.index
    mean = df_mnumlike['Mean_of_Number_of_Likes']
    fig_mnumlike = px.bar(df_mnumlike, x=mean,y=sentiment,
                          title='Mean of Number of Likes of Tweets for Taylor\'s University', color='Mean_of_Number_of_Likes',
                          labels={'Mean_of_Number_of_Likes':'Mean Number of Likes'}, height=500, width=900, orientation='h')


    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Percentages of Sentiments', 'Word Clouds', 'Source of Tweets', 'Mean of Number of Likes of Tweets'])
    with tab1:
        st.plotly_chart(fig_sen, theme="streamlit", use_conatiner_width=True)
    with tab2:
        st.image(image_pos, caption='Word cloud of Tweets with Positive Sentiment for Taylor\'s University')
        st.image(image_neg, caption='Word cloud of Tweets with Negative Sentiment for Taylor\'s University')
        st.image(image_neu, caption='Word cloud of Tweets with Neutral Sentiment for Taylor\'s University')
    with tab3:
        st.plotly_chart(fig_sot, theme="streamlit", use_conatiner_width=True)
    with tab4:
        st.plotly_chart(fig_mnumlike, theme="streamlit", use_conatiner_width=True)
    
    
    
    
elif uni == 'UCSI University':
    
    st.write('You selected:', uni)
    
    # Pie Chart for Percentages of Sentiments
    df_sen = count_values_in_column(ucsi_df_processed,'Sentiment') 
    fig_sen = px.pie(df_sen, values='Percentage', names= df_sen.index,
             title='Percentage of Sentiment for UCSI University',
             hover_data=['Count'], labels={'Count':'Count of Sentiments'})
    fig_sen.update_traces(textposition='inside', textinfo='percent+label')
    
    
    # Word Cloud
    # Positive Sentiment
    create_wordcloud(ucsi_positive.Tokenized_Tweet.values, 'Image/UCSI_Positive_wc.png')
    image_pos = Image.open('Image/UCSI_Positive_wc.png')
    # Negative Sentiment
    create_wordcloud(ucsi_negative.Tokenized_Tweet.values, 'Image/UCSI_Negative_wc.png')
    image_neg = Image.open('Image/UCSI_Negative_wc.png')
    # Neutral Sentiment
    create_wordcloud(ucsi_neutral.Tokenized_Tweet.values, 'Image/UCSI_Neutral_wc.png')
    image_neu = Image.open('Image/UCSI_Neutral_wc.png')
    
    
    # Pie Chart for Source of Tweets
    df_sot = count_values_in_column(ucsi_df_processed,'Source_of_Tweet')
    fig_sot = px.pie(df_sot, values='Percentage', names= df_sot.index,
             title='Percentage of Source of Tweets for UCSI University',
             hover_data=['Count'], labels={'Count':'Count of Souce of Tweets'})
    fig_sot.update_traces(textposition='inside', textinfo='percent+label')
    
    
    
    # Bar Chart for Mean of Number of Likes of Tweets
    df_mnumlike = mean_number_of_likes_by_sentiment(ucsi_df_processed)
    sentiment = df_mnumlike.index
    mean = df_mnumlike['Mean_of_Number_of_Likes']
    fig_mnumlike = px.bar(df_mnumlike, x=mean,y=sentiment,
                          title='Mean of Number of Likes of Tweets for UCSI University', color='Mean_of_Number_of_Likes',
                          labels={'Mean_of_Number_of_Likes':'Mean Number of Likes'}, height=500, width=900, orientation='h')


    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Percentages of Sentiments', 'Word Clouds', 'Source of Tweets', 'Mean of Number of Likes of Tweets'])
    with tab1:
        st.plotly_chart(fig_sen, theme="streamlit", use_conatiner_width=True)
    with tab2:
        st.image(image_pos, caption='Word cloud of Tweets with Positive Sentiment for UCSI University')
        st.image(image_neg, caption='Word cloud of Tweets with Negative Sentiment for UCSI University')
        st.image(image_neu, caption='Word cloud of Tweets with Neutral Sentiment for UCSI University')
    with tab3:
        st.plotly_chart(fig_sot, theme="streamlit", use_conatiner_width=True)
    with tab4:
        st.plotly_chart(fig_mnumlike, theme="streamlit", use_conatiner_width=True)

        
        

elif uni == 'University of Malaya (UM)':
    
    st.write('You selected:', uni)
    
    # Pie Chart for Percentages of Sentiments
    df_sen = count_values_in_column(um_df_processed,'Sentiment') 
    fig_sen = px.pie(df_sen, values='Percentage', names= df_sen.index,
             title='Percentage of Sentiment for University of Malaya (UM)',
             hover_data=['Count'], labels={'Count':'Count of Sentiments'})
    fig_sen.update_traces(textposition='inside', textinfo='percent+label')
    
    
    # Word Cloud
    # Positive Sentiment
    create_wordcloud(um_positive.Tokenized_Tweet.values, 'Image/UM_Positive_wc.png')
    image_pos = Image.open('Image/UM_Positive_wc.png')
    # Negative Sentiment
    create_wordcloud(um_negative.Tokenized_Tweet.values, 'Image/UM_Negative_wc.png')
    image_neg = Image.open('Image/UM_Negative_wc.png')
    # Neutral Sentiment
    create_wordcloud(um_neutral.Tokenized_Tweet.values, 'Image/UM_Neutral_wc.png')
    image_neu = Image.open('Image/UM_Neutral_wc.png')
    
    
    # Pie Chart for Source of Tweets
    df_sot = count_values_in_column(um_df_processed,'Source_of_Tweet')
    fig_sot = px.pie(df_sot, values='Percentage', names= df_sot.index,
             title='Percentage of Source of Tweets for University of Malaya (UM)',
             hover_data=['Count'], labels={'Count':'Count of Souce of Tweets'})
    fig_sot.update_traces(textposition='inside', textinfo='percent+label')
    
    
    
    # Bar Chart for Mean of Number of Likes of Tweets
    df_mnumlike = mean_number_of_likes_by_sentiment(um_df_processed)
    sentiment = df_mnumlike.index
    mean = df_mnumlike['Mean_of_Number_of_Likes']
    fig_mnumlike = px.bar(df_mnumlike, x=mean,y=sentiment,
                          title='Mean of Number of Likes of Tweets for University of Malaya (UM)', color='Mean_of_Number_of_Likes',
                          labels={'Mean_of_Number_of_Likes':'Mean Number of Likes'}, height=500, width=900, orientation='h')


    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Percentages of Sentiments', 'Word Clouds', 'Source of Tweets', 'Mean of Number of Likes of Tweets'])
    with tab1:
        st.plotly_chart(fig_sen, theme="streamlit", use_conatiner_width=True)
    with tab2:
        st.image(image_pos, caption='Word cloud of Tweets with Positive Sentiment for University of Malaya (UM)')
        st.image(image_neg, caption='Word cloud of Tweets with Negative Sentiment for University of Malaya (UM)')
        st.image(image_neu, caption='Word cloud of Tweets with Neutral Sentiment for University of Malaya (UM)')
    with tab3:
        st.plotly_chart(fig_sot, theme="streamlit", use_conatiner_width=True)
    with tab4:
        st.plotly_chart(fig_mnumlike, theme="streamlit", use_conatiner_width=True)
    
    
    
    
elif uni == 'University of Putra Malaysia (UPM)':
    
    st.write('You selected:', uni)
    
    # Pie Chart for Percentages of Sentiments
    df_sen = count_values_in_column(upm_df_processed,'Sentiment') 
    fig_sen = px.pie(df_sen, values='Percentage', names= df_sen.index,
             title='Percentage of Sentiment for University of Putra Malaysia (UPM)',
             hover_data=['Count'], labels={'Count':'Count of Sentiments'})
    fig_sen.update_traces(textposition='inside', textinfo='percent+label')
    
    
    # Word Cloud
    # Positive Sentiment
    create_wordcloud(upm_positive.Tokenized_Tweet.values, 'Image/UPM_Positive_wc.png')
    image_pos = Image.open('Image/UPM_Positive_wc.png')
    # Negative Sentiment
    create_wordcloud(upm_negative.Tokenized_Tweet.values, 'Image/UPM_Negative_wc.png')
    image_neg = Image.open('Image/UPM_Negative_wc.png')
    # Neutral Sentiment
    create_wordcloud(upm_neutral.Tokenized_Tweet.values, 'Image/UPM_Neutral_wc.png')
    image_neu = Image.open('Image/UPM_Neutral_wc.png')
    
    
    # Pie Chart for Source of Tweets
    df_sot = count_values_in_column(upm_df_processed,'Source_of_Tweet')
    fig_sot = px.pie(df_sot, values='Percentage', names= df_sot.index,
             title='Percentage of Source of Tweets for University of Putra Malaysia (UPM)',
             hover_data=['Count'], labels={'Count':'Count of Souce of Tweets'})
    fig_sot.update_traces(textposition='inside', textinfo='percent+label')
    
    
    
    # Bar Chart for Mean of Number of Likes of Tweets
    df_mnumlike = mean_number_of_likes_by_sentiment(upm_df_processed)
    sentiment = df_mnumlike.index
    mean = df_mnumlike['Mean_of_Number_of_Likes']
    fig_mnumlike = px.bar(df_mnumlike, x=mean,y=sentiment,
                          title='Mean of Number of Likes of Tweets for University of Putra Malaysia (UPM)',
                          color='Mean_of_Number_of_Likes',labels={'Mean_of_Number_of_Likes':'Mean Number of Likes'}, 
                          height=500, width=900, orientation='h')


    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Percentages of Sentiments', 'Word Clouds', 'Source of Tweets', 'Mean of Number of Likes of Tweets'])
    with tab1:
        st.plotly_chart(fig_sen, theme="streamlit", use_conatiner_width=True)
    with tab2:
        st.image(image_pos, caption='Word cloud of Tweets with Positive Sentiment for University of Putra Malaysia (UPM)')
        st.image(image_neg, caption='Word cloud of Tweets with Negative Sentiment for University of Putra Malaysia (UPM)')
        st.image(image_neu, caption='Word cloud of Tweets with Neutral Sentiment for University of Putra Malaysia (UPM)')
    with tab3:
        st.plotly_chart(fig_sot, theme="streamlit", use_conatiner_width=True)
    with tab4:
        st.plotly_chart(fig_mnumlike, theme="streamlit", use_conatiner_width=True)

    
    
    
elif uni == 'National University of Malaysia (UKM)':
    
    st.write('You selected:', uni)
    
    # Pie Chart for Percentages of Sentiments
    df_sen = count_values_in_column(ukm_df_processed,'Sentiment') 
    fig_sen = px.pie(df_sen, values='Percentage', names= df_sen.index,
             title='Percentage of Sentiment for National University of Malaysia (UKM)',
             hover_data=['Count'], labels={'Count':'Count of Sentiments'})
    fig_sen.update_traces(textposition='inside', textinfo='percent+label')
    
    
    # Word Cloud
    # Positive Sentiment
    create_wordcloud(ukm_positive.Tokenized_Tweet.values, 'Image/UKM_Positive_wc.png')
    image_pos = Image.open('Image/UKM_Positive_wc.png')
    # Negative Sentiment
    create_wordcloud(ukm_negative.Tokenized_Tweet.values, 'Image/UKM_Negative_wc.png')
    image_neg = Image.open('Image/UKM_Negative_wc.png')
    # Neutral Sentiment
    create_wordcloud(ukm_neutral.Tokenized_Tweet.values, 'Image/UKM_Neutral_wc.png')
    image_neu = Image.open('Image/UKM_Neutral_wc.png')
    
    
    # Pie Chart for Source of Tweets
    df_sot = count_values_in_column(ukm_df_processed,'Source_of_Tweet')
    fig_sot = px.pie(df_sot, values='Percentage', names= df_sot.index,
             title='Percentage of Source of Tweets for National University of Malaysia (UKM)',
             hover_data=['Count'], labels={'Count':'Count of Souce of Tweets'})
    fig_sot.update_traces(textposition='inside', textinfo='percent+label')
    
    
    
    # Bar Chart for Mean of Number of Likes of Tweets
    df_mnumlike = mean_number_of_likes_by_sentiment(ukm_df_processed)
    sentiment = df_mnumlike.index
    mean = df_mnumlike['Mean_of_Number_of_Likes']
    fig_mnumlike = px.bar(df_mnumlike, x=mean,y=sentiment,
                          title='Mean of Number of Likes of Tweets for National University of Malaysia (UKM)',
                          color='Mean_of_Number_of_Likes',labels={'Mean_of_Number_of_Likes':'Mean Number of Likes'}, 
                          height=500, width=900, orientation='h')


    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Percentages of Sentiments', 'Word Clouds', 'Source of Tweets', 'Mean of Number of Likes of Tweets'])
    with tab1:
        st.plotly_chart(fig_sen, theme="streamlit", use_conatiner_width=True)
    with tab2:
        st.image(image_pos, caption='Word cloud of Tweets with Positive Sentiment for National University of Malaysia (UKM)')
        st.image(image_neg, caption='Word cloud of Tweets with Negative Sentiment for National University of Malaysia (UKM)')
        st.image(image_neu, caption='Word cloud of Tweets with Neutral Sentiment for National University of Malaysia (UKM)')
    with tab3:
        st.plotly_chart(fig_sot, theme="streamlit", use_conatiner_width=True)
    with tab4:
        st.plotly_chart(fig_mnumlike, theme="streamlit", use_conatiner_width=True)
    
    
   
    