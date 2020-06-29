import streamlit as st
import joblib,os
import spacy
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt

nlp = spacy.load('en')

def model_loader(filename):
    with open(f"resources/{filename}", 'rb') as file:
        model=joblib.load(file)

    return model

multinomial_logistic = model_loader('multinomial_logistic.plk')
LinearSVC = model_loader('LinearSVC.plk')
SVC = model_loader('SVC.pkl')

def get_keys(value,my_dic):
    for key,val in my_dic.items():
        if val==value:
            return key
def main():
    """tweet classifier"""
    
    
    activities =["Prediction","NLP","Data Exploritory Analysis","Model Comparison"]
    choice = st.sidebar.selectbox("Choose Activity",activities)

    if choice=='Model Comparison':
        st.title("Model Comparison")
        task = ['Weighted f1 score graph','Weighted f1 score table']
        choice_M = st.sidebar.selectbox("Choose Activity",task)
        if choice_M=='Weighted f1 score graph':
            image = Image.open('resources/imgs/weighted_score.PNG')
            st.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', format='PNG')
        if choice_M=='Weighted f1 score table':
            image = Image.open('resources/imgs/df_perfomance.PNG')
            st.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', format='PNG')

    elif choice=='Data Exploritory Analysis':
        st.title("Data Exploritory Analysis")
        task1 = ['Number of entries per class graph','Number of entries per class table']
        choice_D= st.sidebar.selectbox("Choose Activity",task1)
        if choice_D=='Number of entries per class graph':
            image1 = Image.open('resources/imgs/num_per_class.PNG')
            st.image(image1, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', format='PNG')
        if choice_D=='Number of entries per class table':
            image1 = Image.open('resources/imgs/df_num_perclass.PNG')
            st.image(image1, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', format='PNG')


    elif choice=='Prediction':
        st.title("Tweet Classifier")
        st.subheader("NLP and ML App with streamlit")
        st.info("Prediction with ML")
        tweet_text = st.text_area("Enter Text","Type Here")
        all_ml_modules = ["MLR","LSVC","SVC"]
        model_choice = st.selectbox("Choose the ML model",all_ml_modules)
        labels = {'News':2,'Pro':1,'Neutral':0,'Anti':-1}
        if st.button("Classify"):
            st.text("Original tweet ::\n{}".format(tweet_text))
            if model_choice=="MLR":
                # predictor = joblib.load(open(os.path.join("resources/multinomial_logistic.plk"),"rb"))
                prediction = multinomial_logistic.predict([tweet_text])
                ##st.write(prediction[0])
                st.success(f"The tweet is categorized as {get_keys(prediction[0],labels)}")
            if model_choice=="LSVC":
                # predictor = joblib.load(open(os.path.join("resources/LinearSVC.plk"),"rb"))
                prediction = LinearSVC.predict([tweet_text])
                ##st.write(prediction[0])
                st.success(f"The tweet is categorized as {get_keys(prediction[0],labels)}")
            if model_choice=="SVC":
                # predictor = joblib.load(open(os.path.join("resources/LinearSVC.plk"),"rb"))
                prediction = SVC.predict([tweet_text])
                ##st.write(prediction[0])
                st.success(f"The tweet is categorized as {get_keys(prediction[0],labels)}")


    elif choice=='NLP':
        st.title("Tweet Classifier")
        st.subheader("NLP and ML App with streamlit")
        st.info("Natural Language Processing")
        tweet_text = st.text_area("Enter Text","Type Here")
        nlp_task = ["Tokenization","NER","Lemmatization","POS"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button('Analyze'):
            st.info("Original tweet ::\n{}".format(tweet_text))

            doc = nlp(tweet_text)
            if task_choice =="Tokenization":
                tokens = [t.text for t in doc]

            elif task_choice =="NER":
                tokens = [(ent.text,ent.label_) for ent in doc.ents]

            elif task_choice =="Lemmatization":
                tokens = [f"'Token':{t.text},'Lemma':{t.lemma_}" for t in doc]
            elif task_choice =="POS":
                tokens = [f"'Token':{t.text},'POS':{t.pos_},'Dependency':{t.dep_}" for t in doc]
            st.json(tokens)
        if st.checkbox("Wordcloud"):
            wordcloud  = WordCloud().generate(tweet_text)
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot()


if __name__ == "__main__":
    main()