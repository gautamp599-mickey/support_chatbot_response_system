import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import random

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

download_nltk_data()

lm=WordNetLemmatizer()

lr=joblib.load('chatbot_model.pkl')
golden_responses=joblib.load('golden_responses.pkl')
le=joblib.load('label_encoder.pkl')
tfidf=joblib.load('tfidf_vectorizer.pkl')

def cleaning(text):
    text=re.sub(r'\{\{.*?\}\}|[*?]', "", text)
    text=re.sub(r'[^\w\s]', "", text)
    return text.lower()

def tag_helper(txt):    
    if txt.startswith('J'): return wordnet.ADJ
    elif txt.startswith('V'): return wordnet.VERB
    elif txt.startswith('N'): return wordnet.NOUN
    elif txt.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def lemma_word(text):
    lm_setnz=[]
    tokens=nltk.word_tokenize(text)
    tagged=nltk.pos_tag(tokens)
    for word, tag in tagged:
        wn_tag=tag_helper(tag)
        cleaned_word=lm.lemmatize(word, pos=wn_tag)
        lm_setnz.append(cleaned_word)
    return ' '.join(lm_setnz)

def get_bot_response(user_input):
    cleaned_text = lemma_word(cleaning(user_input))
    transformed_input = tfidf.transform([cleaned_text])
    
    max_confidence = lr.predict_proba(transformed_input).max()
    
    if max_confidence < 0.25:
        return "I'm sorry, I don't quite understand. Could you rephrase your question?"
    
    vec_intent = lr.predict(transformed_input)
    intent_1 = le.inverse_transform(vec_intent)[0]
    
    if intent_1 in golden_responses:
        return random.choice(golden_responses[intent_1])
    return "System Error."

st.title("🤖 Enterprise IT Support Bot")
st.write("Ask me about orders, refunds, shipping, or account issues!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_bot_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})