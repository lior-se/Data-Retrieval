import codecs
import re
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from tqdm import tqdm

# region CNN MODEL
def load_data(filename):
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    x, y = zip(*[d.strip().split('\t') for d in data])
    # Reducing any char-acter sequence of more than 3 consecutive repetitions to a respective 3-character sequence
    # (e.g. “!!!!!!!!”turns to “!!!”)
    # x = [re.sub(r'((.)\2{3,})', r'\2\2\2', i) for i in x]
    x = np.asarray(list(x))
    y = to_categorical(y, 3)

    return x, y


x_token_train, y_token_train = load_data('data/token_train.tsv')
x_token_test, y_token_test = load_data('data/token_test.tsv')
x_morph_train, y_morph_train = load_data('data/morph_train.tsv')
x_morph_test, y_morph_test = load_data('data/morph_test.tsv')

from keras.preprocessing import text, sequence


def tokenizer(x_train, x_test, vocabulary_size, char_level):
    tokenize = text.Tokenizer(num_words=vocabulary_size,
                              char_level=char_level,
                              filters='')
    tokenize.fit_on_texts(x_train)  # only fit on train
    # print('UNK index: {}'.format(tokenize.word_index['UNK']))

    x_train = tokenize.texts_to_sequences(x_train)
    x_test = tokenize.texts_to_sequences(x_test)

    return x_train, x_test


def pad(x_train, x_test, max_document_length):
    x_train = sequence.pad_sequences(x_train, maxlen=max_document_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_document_length, padding='post', truncating='post')

    return x_train, x_test


vocabulary_size = 5000

x_token_train, x_token_test = tokenizer(x_token_train, x_token_test, vocabulary_size, False)
x_morph_train, x_morph_test = tokenizer(x_morph_train, x_morph_test, vocabulary_size, False)

max_document_length = 512

x_token_train, x_token_test = pad(x_token_train, x_token_test, max_document_length)
x_morph_train, x_morph_test = pad(x_morph_train, x_morph_test, max_document_length)

import matplotlib.pyplot as plt


def plot_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model Accuracy')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    fig.tight_layout()
    plt.show()

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers import BatchNormalization
from keras import optimizers
from keras import metrics
from keras import backend as K

dropout_keep_prob = 0.5
embedding_size = 300
batch_size = 50
lr = 1e-4
dev_size = 0.2

num_epochs = 5

# Create new TF graph
K.clear_session()

# Construct model
convs = []
text_input = Input(shape=(max_document_length,))
x = Embedding(vocabulary_size, embedding_size)(text_input)
for fsz in [3, 8]:
    conv = Conv1D(128, fsz, padding='valid', activation='relu')(x)
    pool = MaxPool1D()(conv)
    convs.append(pool)
x = Concatenate(axis=1)(convs)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(dropout_keep_prob)(x)
preds = Dense(3, activation='softmax')(x)

model = Model(text_input, preds)

adam = optimizers.Adam(lr=lr)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# Train the model
history = model.fit(x_token_train, y_token_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=dev_size)

# Plot training accuracy and loss
#plot_loss_and_accuracy(history)

# Evaluate the model
scores = model.evaluate(x_token_test, y_token_test,
                        batch_size=batch_size, verbose=1)
print('\nAccurancy: {:.3f}'.format(scores[1]))

# Save the model
model.save('word_saved_models/CNN-Token-{:.3f}.h5'.format((scores[1] * 100)))

#endregion



#region Load data and load tokenizer
file_path = "15000.xlsx"  # Update the file path as necessary
df = pd.read_excel(file_path, engine='openpyxl')
df_A = df[df.iloc[:, 3] == 'A'].iloc[:, 2]
df_B = df[df.iloc[:, 3] == 'B'].iloc[:, 2]
df_C = df[df.iloc[:, 3] == 'C'].iloc[:, 2]


x_token_train, y_token_train = load_data('data/token_train.tsv')
tokenize = text.Tokenizer(num_words=vocabulary_size,
                              char_level=False,
                              filters='')
tokenize.fit_on_texts(x_token_train)
#endregion


# region Weighted chunks to fit maxlenght
def predict_chunks(chunks, model):
    """Predict sentiment for each chunk and return the weighted average prediction, excluding padding."""
    predictions = np.array(model.predict(chunks))

    # Initialize variables to hold the weighted sum of predictions and the total weight
    weighted_sum_predictions = np.zeros(predictions.shape[1])  # Assuming predictions are 2D: (num_chunks, num_classes)
    total_weight = 0  # This will store the sum of the weights for normalization

    # Iterate over each chunk
    for chunk, prediction in zip(chunks, predictions):
        # Count the non-zero elements in the chunk as its weight
        chunk_weight = np.count_nonzero(chunk)
        total_weight += chunk_weight

        # Add the weighted prediction of the current chunk to the weighted sum
        weighted_sum_predictions += chunk_weight * prediction

    # Normalize the weighted sum by the total weight to get the average prediction
    average_prediction = weighted_sum_predictions / total_weight if total_weight > 0 else np.zeros(predictions.shape[1])

    return average_prediction


def split_and_predict(txt, tokenizer, model, max_document_length):
    """Split a single text into chunks, process, and predict sentiment, then average."""
    # Tokenize the text first
    tokenized_text = tokenizer.texts_to_sequences([txt])

    # Flatten list since texts_to_sequences wraps each text in a list
    tokenized_text = [token for sublist in tokenized_text for token in sublist]

    if not tokenized_text:
        fallback_prediction = np.array([1 / 3, 1 / 3, 1 / 3])
        return fallback_prediction

    ''' 
    index_word = {i: word for word, i in tokenizer.word_index.items()}
    print("Original text length (characters):", len(txt))
    print(len(tokenized_text[0]))
    tokens_words = [index_word.get(token, '?') for token in tokenized_text]
    print("tokens as words:", ' '.join(tokens_words))

    '''

    # Split into chunks if necessary
    chunks = []
    for i in range(0, len(tokenized_text), max_document_length):
        chunks.append(tokenized_text[i:i + max_document_length])
    # Process each chunk
    processed_chunks = sequence.pad_sequences(chunks, maxlen=max_document_length, padding='post', truncating='post')

    # Predict and average
    return predict_chunks(processed_chunks, model)

#endregion


# region Normal names
predictions_A = []
for txt in tqdm(df_A):
    average_prediction = split_and_predict(txt, tokenize, model, max_document_length)
    predictions_A.append(average_prediction)

average_sentiment_A = np.mean(predictions_A, axis=0)
print("Average Sentiment A:", average_sentiment_A)

predictions_B = []
for txt in tqdm(df_B):
    average_prediction = split_and_predict(txt, tokenize, model, max_document_length)
    predictions_B.append(average_prediction)

average_sentiment_B = np.mean(predictions_B, axis=0)
print("Average Sentiment B:", average_sentiment_B)

predictions_C = []
for txt in tqdm(df_C):
    average_prediction = split_and_predict(txt, tokenize, model, max_document_length)
    predictions_C.append(average_prediction)

average_sentiment_C = np.mean(predictions_C, axis=0)
print("Average Sentiment C:", average_sentiment_C)

#endregion

# region Changed names
df_A_names = pd.read_excel("processed_df_A.xlsx", engine='openpyxl',usecols=[0],header=None)
df_B_names = pd.read_excel("processed_df_B.xlsx", engine='openpyxl',usecols=[0],header=None)
df_C_names = pd.read_excel("processed_df_C.xlsx", engine='openpyxl',usecols=[0],header=None)
df_A_names=df_A_names[0].tolist()
df_B_names=df_B_names[0].tolist()
df_C_names=df_C_names[0].tolist()

predictions_A_names = []
for txt in tqdm(df_A_names):
    average_prediction = split_and_predict(txt, tokenize, model, max_document_length)
    predictions_A_names.append(average_prediction)

average_sentiment_A = np.mean(predictions_A_names, axis=0)
print("Average Sentiment A, changed names:", average_sentiment_A)

predictions_B_names = []
for txt in tqdm(df_B_names):
    average_prediction = split_and_predict(txt, tokenize, model, max_document_length)
    predictions_B_names.append(average_prediction)

average_sentiment_B = np.mean(predictions_B_names, axis=0)
print("Average Sentiment B, changed names:", average_sentiment_B)

predictions_C_names = []
for txt in tqdm(df_C_names):
    average_prediction = split_and_predict(txt, tokenize, model, max_document_length)
    predictions_C_names.append(average_prediction)

average_sentiment_C = np.mean(predictions_C_names, axis=0)
print("Average Sentiment C, changed names:", average_sentiment_C)

#endregion