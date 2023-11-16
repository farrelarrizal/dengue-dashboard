import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Model import Model
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk import bigrams
from nltk import FreqDist
# model = Model()
# model.load_model('model/best_model_text-classification.h5')

st.title("Dengue Analysis Dashboard")
st.subheader("This is a dengue-dashboard to classify dengue text into 5 class with Indonesian language")
st.markdown("__5 class : Informative, Awareness, Infected, News, Others__")
st.divider()

random_number = np.random.randint(0, 3)

with st.form(key="single_tweet analysis"):
    st.write("This is a single tweet")
    sample_tweet = [
        'demam berdarah adalah wabah penyakit yang dibawa oleh nyamuk dengue. Infeksi yang diberikan selama 2-3 minggu.',
        'waspada demam berdarah di daerah anda. Jangan lupa untuk menjaga kebersihan lingkungan anda.',
        'demam berdarah, tapi waktu itu masih bisa masuk sekolah bawa motor lagi anjayy, padahal udah mimisan. pas ke dokter, blio kaget kok gua masih kuat, padahal trombosit udah kritis. wkwk.'
    ]
    txt = st.text_area("input your text here", sample_tweet[random_number])
    submit = st.form_submit_button("Predict Label 😎")

    if submit:
        # // spinner
        with st.spinner("Predicting..."):
            model = Model()
            model.load_model('model/best_model_text-classification.h5')
            prediction = model.predict_text(txt)
            st.write("Prediction: ", prediction)



# st.download_button("Download File Sample", pd.read_excel('data/labeled-test.xlsx').to_csv(), 'Click here to download')

with st.form(key="file_upload"):
    st.write("This is a file upload")
    file = st.file_uploader("Upload your file here", type=["csv", "txt", 'xlsx'])
    # st.link_button('Template File Upload', 'https://docs.google.com/spreadsheets/d/1-QVZyFsOqaHXPnmgbdGffH39aJ-4aZfV/edit?usp=share_link&ouid=102018509068416474328&rtpof=true&sd=true')

    st.divider()

    # split into 2 column
    col1, col2, col3 = st.columns(3)
    with col1:
        file_submit = st.form_submit_button("Get Insights 📊")
    with col3:
        st.link_button('Template File Upload',
                       'https://docs.google.com/spreadsheets/d/1-QVZyFsOqaHXPnmgbdGffH39aJ-4aZfV/edit?usp=share_link&ouid=102018509068416474328&rtpof=true&sd=true')
    if file_submit:
# // spinner
        with st.spinner("Predicting..."):
            # df = model.predict_file(file)
            #
            # st.write("Prediction: ", df)
            # st.download_button("Download File", df.to_csv(), 'Click here to download')

            # read file
            data = pd.read_excel(file)
            # data = pd.read_excel('data/labeled-test.xlsx')
            # read first column without name
            data = data.iloc[:, 0]
            text = ' '.join(data)

            # print len of text
            st.markdown('### Length of Text')
            total_tweet = len(data)
            st.write(f'Total Tweet: {total_tweet} tweet')

            st.markdown('### Wordcloud of Tweets')
            # Create some sample text
            # text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'
            text_stopword = StopWordRemoverFactory().get_stop_words()
            text_stopword.append('bgt')
            text_stopword.append('banget')
            text_stopword.append('aja')
            text_stopword.append('gw')
            text_stopword.append('pa')
            text_stopword.append('kalo')
            text_stopword.append('aku')
            text_stopword.append('gue')

            text = text.split(' ')
            text = [t.strip() for t in text]
            text = [t for t in text if t not in text_stopword]
            text = ' '.join(text)

            # Create and generate a word cloud image:
            wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
            wordcloud.generate(text)

            # Display the generated image with white background:
            plt.imshow(wordcloud, interpolation='bilinear' )
            plt.axis("off")
            plt.show()
            st.pyplot(plt)

            # top 10 bigram
            st.markdown('### Top 10 Bigram (Topic Frequency)')
            text = text.split(' ')
            text = [t.strip() for t in text]
            # remove '' from list
            text = [t for t in text if t != '']
            bigrams = list(bigrams(text))
            freq_dist = FreqDist(bigrams)

            print(freq_dist.most_common(10))
            df = pd.DataFrame(freq_dist.most_common(10), columns=['Bigram', 'Frequency'], index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
            st.dataframe(df)

            # predicting class
            model = Model()
            model.load_model('model/best_model_text-classification.h5')
            prediction = model.predict_texts(data.tolist())

            st.markdown('### Distribution of Clasisification')
            # df = pd.DataFrame([1, 2, 3, 4, 5], index=['Informative', 'Awareness','Infected', 'News', 'Others'], columns=['count'])
            data_class = pd.DataFrame(prediction, columns=['class'])
            df = data_class['class'].value_counts().rename_axis('class').reset_index(name='count')
            st.bar_chart(df, x='class', y='count')

            st.dataframe(df)

            # merge text and prediction
            st.markdown('### Classifications of Tweets')
            data = pd.DataFrame(data)
            data['class'] = prediction
            data.columns = ['text', 'class']
            st.dataframe(data)