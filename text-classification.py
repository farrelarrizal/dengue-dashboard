# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import transformers
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D
# from tensorflow.keras.models import Model
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, TFBertModel
#
#
# text2label = {
#     "Informative": 0,
#     "Awaress": 1,
#     "Awareness":1,
#     "Infected": 2,
#     "News": 3,
#     "Others": 4
# }
#
# label2text = {
#     0: "Informative",
#     1: "Awareness",
#     2: "Infected",
#     3: "News",
#     4: "Others"
# }
#
# # Define BERT model
# bert_model_name = "indobenchmark/indobert-large-p2"
# tokenizer = BertTokenizer.from_pretrained(bert_model_name)
# bert_model = TFBertModel.from_pretrained(bert_model_name)
#
# # LSTM - CNN
# def build_model(max_len):
#
#     # Input layers
#     input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
#     embedding = bert_model(input_word_ids, attention_mask=input_mask)[0]
#
#     # CNN layer
#     # conv + relu + maxpool strides = 1, kernel_size = 3, channel = 10
#     X = tf.keras.layers.Conv1D(512, 3, activation='relu')(embedding)
#     X = tf.keras.layers.MaxPooling1D(4)(X)
#     X = tf.keras.layers.Conv1D(256, 5, activation='relu')(embedding)
#     X = tf.keras.layers.MaxPooling1D(4)(X)
#     X = tf.keras.layers.Conv1D(128, 7, activation='relu')(X)
#     X = tf.keras.layers.MaxPooling1D(4)(X)
#
#     # LSTM layer from CNN output
#     X = tf.keras.layers.LSTM(128)(X)
#     X = tf.keras.layers.Dense(64, activation='relu')(X)
#     X = tf.keras.layers.Dropout(0.2)(X)
#     X = tf.keras.layers.Dense(32, activation='relu')(X)
#     X = tf.keras.layers.Dropout(0.2)(X)
#
#     # Output layer
#     output = tf.keras.layers.Dense(5, activation='softmax')(X)
#
#     # Model definition
#     model = tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=output)
#
#     # Compile model
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
#     loss = 'sparse_categorical_crossentropy'
#     metrics = [
#         tf.keras.metrics.SparseCategoricalCrossentropy(),
#         tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
#     ]
#
#     model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#
#     return model
#
# # Build model
# model = build_model(280)
#
# # text = 'Saya sakit demam berdarah sangat parah'
# # text = 'saya menghimbau warga untuk terus waspada dengan demam berdarah; Saya saat ini sakit demam berdarah sangat parah'
# text = 'saya menghimbau warga untuk terus waspada dengan demam berdarah'
#
#
# text = text.split(';')
# text = [t.strip() for t in text]
# print(text)
#
#
#
# def predict_text(text):
#     # Tokenize text
#     text = tokenizer.batch_encode_plus(
#         [text],
#         max_length=280,
#         truncation=True,
#         padding='max_length',
#         return_token_type_ids=False
#     )
#
#     # Convert train and test data to tensors
#     text_seq = tf.convert_to_tensor(text['input_ids'])
#     text_mask = tf.convert_to_tensor(text['attention_mask'])
#
#     # Predict
#     prediction = model.predict([text_seq, text_mask])
#     # Get label
#     label = np.argmax(prediction)
#     print(label2text[label])
#     return label2text[label]
#
# def predict_file(path):
#     # open file ['xlsx']
#     data = pd.read_excel('data/labeled-test.xlsx')
#
#     #read_column_with_name_text
#     data = data.text
#
#     # tokenize the text
#     texts = tokenizer.batch_encode_plus(
#         data,
#         max_length=280,
#         truncation=True,
#         padding='max_length',
#         return_token_type_ids=False
#     )
#
#     # Convert train and test data to tensors
#     text_seq = tf.convert_to_tensor(texts['input_ids'])
#     text_mask = tf.convert_to_tensor(texts['attention_mask'])
#
#     # Predict
#     prediction = model.predict([text_seq, text_mask])
#
#     # Get label
#     label = np.argmax(prediction)
#     print(label2text[label])
#
#     return label2text[label]
#
# predict_file('data/labeled-test.xlsx')
#
#
#
# predict_text('Saya sedang sakit demam berdarah sangat parah')
# predict_text('saya menghimbau warga untuk waspada dengan demam berdarah')
#
#
#
#
#

import pandas as pd
df = pd.DataFrame([1,2,3], index=['positive', 'neutral', 'negative'], columns=['count'])
print(df)