import requests
from bs4 import BeautifulSoup #For web scraping
import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from collections import Counter

################################################Web Scraping########################################################
keywords = ["stock", "tariff"]
query = "+".join(['"stock"', '"tariff"']) 
all_titles = []
page = 0


while len(all_titles) < 25 and page < 10:
    offset = page * 10 + 1 #1>11>21>31>41..
    search_url = f"https://www.bing.com/news/search?q={query}&first={offset}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headline_links = soup.find_all("a", class_="title")

    for t in headline_links:
        title_text = t.get_text(strip=True)
        title_lower = title_text.lower()

        
        if all(k in title_lower for k in keywords):
            if title_text.lower() not in [t.lower() for t in all_titles]:
                all_titles.append(title_text)

        if len(all_titles) >= 25:
            break  #Stop if hit 25

    page = page + 1

#Put titles into excel
df = pd.DataFrame(all_titles, columns=["Text Entry"])
df.to_excel("Assignment2_Data.xlsx", index=False)

################################################Calculate sentiment score#############################################
#Load and read NRC lexicon file
def load_nrc_lexicon(filepath):
    lexicon = {}  

    # Open the file and read each line
    with open(filepath, 'r', encoding='utf-8') as file: #r=reading mode
        for line in file:
            #Each line has: word, emotion, and score
            parts = line.strip().split('\t')    #threat\tnegative\t1
            word = parts[0]
            emotion = parts[1]
            score = int(parts[2])

            if score == 1:    # Keep score = 1
                if word not in lexicon:
                    lexicon[word] = {}  # Create a sub-dictionary for the word
                lexicon[word][emotion] = 1  # Store the emotion

    return lexicon

nrc_lexicon = load_nrc_lexicon("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

#Check the tokenized words with nrc lexicon
def analyze_title_nrc(words, lexicon):
    emotion_counts = Counter() #to count

    #Go through each word
    for word in words:
        if word in lexicon:  # Check if this word exists in the NRC lexicon
                for emotion in lexicon[word]:  # Go through each emotion linked to the word
                    emotion_counts[emotion] += 1  # Add 1 to the emotion count
    return emotion_counts

################################################Calculate sentiment score#############################################
def calculate_sentiment_score(emotion_counts):
    positive_emotions = {'positive', 'joy', 'trust'}

    negative_emotions = {'negative', 'sadness', 'anger', 'fear', 'disgust'}

    score = 0

    for emotion, count in emotion_counts.items():
        if emotion in positive_emotions:
            score += count
        elif emotion in negative_emotions:
            score -= count

    return score

#################################################DistilBERT#############################################
from sentence_transformers import SentenceTransformer

#Load DistilBERT model
model = SentenceTransformer("distilbert-base-nli-mean-tokens")

#################################################Extract as Excel#############################################
# Prepare data storage
titles = []
nrc_scores = []
bert_vectors = []


#Read Assignment2_Data
excel = pd.read_excel("Assignment2_Data.xlsx")
all_titles = excel["Text Entry"].tolist()


# Loop through each article title
for title in all_titles:
    titles.append(title)

    # --- Classic NRC Sentiment Score ---
    words = word_tokenize(title.lower())
    emotions = analyze_title_nrc(words, nrc_lexicon)
    score = calculate_sentiment_score(emotions)
    nrc_scores.append(score)

    # --- DistilBERT vector
    vector = model.encode(title)
    vector_str = ','.join([f"{v:.4f}" for v in vector[:5]]) + "..." 
    bert_vectors.append(vector_str)

# Create DataFrame
df = pd.DataFrame({
    "Text Entry": titles,
    "NRC Sentiment Scores": nrc_scores,    
    "DistilBERT": bert_vectors     
})

# Export to Excel
df.to_excel("Assignment2_Final.xlsx", index=False)
