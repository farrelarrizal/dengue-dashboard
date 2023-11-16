import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

class Model:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-large-p2')
        self.bert_model = TFBertModel.from_pretrained('indobenchmark/indobert-large-p2')
        print('Pretrained model loaded successfully...')

    def build_model(self, max_len):
        # Input layers
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        embedding = self.bert_model(input_word_ids, attention_mask=input_mask)[0]

        # CNN layer
        # conv + relu + maxpool strides = 1, kernel_size = 3, channel = 10
        X = tf.keras.layers.Conv1D(512, 3, activation='relu')(embedding)
        X = tf.keras.layers.MaxPooling1D(4)(X)
        X = tf.keras.layers.Conv1D(256, 5, activation='relu')(embedding)
        X = tf.keras.layers.MaxPooling1D(4)(X)
        X = tf.keras.layers.Conv1D(128, 7, activation='relu')(X)
        X = tf.keras.layers.MaxPooling1D(4)(X)

        # LSTM layer from CNN output
        X = tf.keras.layers.LSTM(128)(X)
        X = tf.keras.layers.Dense(64, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(32, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.2)(X)

        # Output layer
        output = tf.keras.layers.Dense(5, activation='softmax')(X)

        # Model definition
        model = tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=output)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = 'sparse_categorical_crossentropy'
        metrics = [
            tf.keras.metrics.SparseCategoricalCrossentropy(),
            tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),

        ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def load_model(self, model_path):
        self.model = self.build_model(280)
        self.model.load_weights(model_path)
        print('Model loaded successfully...')

    def predict_text(self, text):
        text = self.tokenizer.batch_encode_plus(
            [text],
            max_length=280,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )

        text_seq = tf.convert_to_tensor(text['input_ids'])
        text_mask = tf.convert_to_tensor(text['attention_mask'])

        # Predict
        predictions = self.model.predict([text_seq, text_mask])
        predictions = np.argmax(predictions, axis=1)

        # convert label to text
        predictions = self.label2text(predictions[0])
        print(predictions)
        return predictions

    def predict_texts(self, texts):
        texts = self.tokenizer.batch_encode_plus(
            texts,
            max_length=280,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )

        text_seq = tf.convert_to_tensor(texts['input_ids'])
        text_mask = tf.convert_to_tensor(texts['attention_mask'])

        # Predict
        predictions = self.model.predict([text_seq, text_mask])
        predictions = np.argmax(predictions, axis=1)

        # convert label to text
        predictions = [self.label2text(p) for p in predictions]
        return predictions

    def predict_file(self, file):
        # read the extension
        ext = file.name.split('.')[-1]
        if ext == 'csv':
            df = pd.read_csv(file)
        elif ext == 'txt':
            df = pd.read_csv(file, sep='\t')
        elif ext == 'xlsx':
            df = pd.read_excel(file)
        else:
            df = pd.DataFrame()
        print('start to predicting')

        return self.predict_texts(df['text'].tolist())


    def label2text(self, label):
        label2text = {
            0: "Informative",
            1: "Awareness",
            2: "Infected",
            3: "News",
            4: "Others"
        }
        return label2text[label]