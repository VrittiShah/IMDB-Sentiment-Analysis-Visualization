import requests
from bs4 import BeautifulSoup
import csv
from transformers import pipeline, AutoTokenizer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import shutil
import os
import random
import base64

tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

pipe = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    tokenizer=tokenizer,
    return_all_scores=False 
)

# url = 'https://www.imdb.com/title/tt13016388/reviews?ref_=tt_urv'


stars_list = []
reviews_list = []
sentiments_list = []

def get_image(url):
    html_page = requests.get(url)
    soup = BeautifulSoup(html_page.content, 'html.parser')
    images = soup.findAll('img')
    for img_tag in images:
        src = img_tag.get('src')
        if src:
            break
    
    return  src


def convert_imdb_url(movie_url):
    url_without_query_params = movie_url.split('?')[0]

    converted_url = url_without_query_params + "reviews?ref_=tt_urv"
    return converted_url


def create_csv(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    count = 0

# Open CSV file in write mode
    with open('imdb_reviews.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Rating', 'Review',  'Sentiment'])  # Write header row

        reviews = soup.find_all('div', class_='lister-item-content')
        for review in reviews:
            review_text = review.find('div', class_='text show-more__control').text.strip()
            if review_text:
                inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, add_special_tokens = True)
                input_length = inputs['input_ids'].shape[1] 
                
                if input_length > 511:
                    continue
                
                output = pipe(review_text)
                result_label = output[0]['label']
                
                reviews_list.append(review_text)
                sentiments_list.append(result_label)
                count += 1
            
            rating_div = review.find('span', class_='rating-other-user-rating')
            if rating_div:
                rating = rating_div.find('span').text.strip()
                stars_list.append(rating)
                
            else:
                rating = 5
            csvwriter.writerow([rating, review_text, result_label])

    print("CSV file created successfully!")
    return stars_list, reviews_list, sentiments_list,



st.set_page_config( page_icon="🍿", page_title="Movie Analyzer📈",layout="wide")
st.title("🎬 IMDB Movie Reviews Analysis 🎞️")
st.write("Your perfect Review Analyst!")

    # User input for the URL
url_input = st.text_input("Enter the URL:")

def download_csv(data):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)


if st.button("Fetch!"):
        if url_input:
            converted_url = convert_imdb_url(url_input)
            st.success("Link Accessed Successfully!")
            create_csv(converted_url)

            image_url = get_image(converted_url)

            st.image(image_url, caption='Poster', width=200)

            reviews = []
            data = pd.read_csv('imdb_reviews.csv')
            
            st.write(data)

        else:
            st.warning("Please enter a valid URL.")

        # Create a download button for the DataFrame
        download_csv(data)

        positive = ""
        positive_count = 0
        negative = ""
        negative_count = 0
        neutral_count = 0
        adjective_list = []
        stars = 0
        total_rating_sum =0

        for index, row in data.iterrows():
            total_rating_sum += row['Rating']
            if str(row['Sentiment']) == 'positive':
                positive_count += 1
                if positive_count <= 20:
                    positive += str(row['Review'])
            elif str(row['Sentiment']) == 'negative':
                negative_count += 1
                if negative_count <= 20:
                    negative += str(row['Review'])
            else:
                neutral_count += 1
        
            doc = nlp(str(row['Review']))
            for token in doc:
                if token.pos_ == "ADJ" and [token.text.lower(), row['Sentiment']] not in adjective_list:
                    adjective_list.append([token.text.lower(), row['Sentiment']])

        display_adjectives = []
        for i in range(7):
            choice = random.choice(adjective_list)
            if choice not in display_adjectives:
                display_adjectives.append(choice)

        sentiment_score = (positive_count + neutral_count - negative_count) / (positive_count + negative_count + neutral_count)
        positivity_rate = (positive_count) / (positive_count + negative_count + neutral_count)
        number_of_reviews = data.shape[0]
        neutrality_rate = (neutral_count/(positive_count + negative_count + neutral_count))
        negativity_rate = (negative_count/(positive_count + negative_count + neutral_count))



        total1, total2, total4 = st.columns(3)
        with total1:
            st.info('Average Stars: ', icon="⭐")
            st.metric("Stars", round((total_rating_sum/number_of_reviews),1))
        with total2:
            st.info('Average Positivity Rate', icon="😄")
            st.metric("Positivity Rate", round(positivity_rate,2))
        # with total3:
        #     st.info('Average Neutrality Rate', icon="😐")
        #     st.metric("Neutrality Rate", round(neutrality_rate,2))
        with total4:
            st.info('Average Negativity Rate', icon="😔")
            st.metric("Negativity Rate",round(negativity_rate,2))
# Add a rectangle around each outer list element
        st.markdown('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)

        for aspect, sentiment in display_adjectives:
            if sentiment == "positive":
                bg_color = "#3d7eba"
            elif sentiment == "negative":
                bg_color = "#fa897f"
            else:
                bg_color = "#858282"
            
            st.markdown(f'<div style="border: 2px solid #e2cbcb; border-radius:1rem; padding: 10px; margin: 10px;">{aspect}<span style="background-color: {bg_color}; color:#FFFFFF;margin-left:10px; border-radius: 0.7rem; padding:5px; font-size:16px">{sentiment}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)





        col11, col12, col13 = st.columns(3)

        with col11:
            st.write("")

        with col13:
            st.write("")

        labels = ["Positive", "Negative"]
        colors = ['#3d7eba', '#fa897f'] 
        sizes = [positive_count, negative_count]
        with col12:
            # Create the pie chart with custom colors
            fig, ax = plt.subplots()
            patches, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.set_aspect('equal')
            fig.patch.set_facecolor('none')  # Equal aspect ratio ensures the pie chart is circular
            
            for text in texts:
                text.set_color('white')
            # Display the plot in Streamlit
            st.pyplot(fig)
            labels=["Postive","Negative"] 
            sizes=[positive_count, negative_count]

        