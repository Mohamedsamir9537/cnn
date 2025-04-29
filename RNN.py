#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Generate sequence
def generate_seq(model, tokenizer, enter_text, n_pred):
    in_text, result = enter_text, enter_text
    for _ in range(n_pred):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded[-1:]).reshape(1, 1)  # Fix input shape
        yhat = model.predict(encoded, verbose=0)
        yhat_index = yhat.argmax()
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat_index:
                out_word = word
                break
        in_text, result = out_word, result + ' ' + out_word
    return result

# Test
print(generate_seq(model, tokenizer, 'Piford', 6))
print(generate_seq(model, tokenizer, 'service', 3))


# In[62]:


from numpy import array
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# In[64]:


data = """Piford Technologies is a leading Software Development Company
Piford Technologies provide trainings to working professionals and students
We are product based and service based company
we have one of our office in IT Park, Mohali"""


# In[66]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded_data = tokenizer.texts_to_sequences([data])[0]


# In[68]:


vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size:', vocab_size)


# In[70]:


sequences = list()
for i in range(1, len(encoded_data)):
    sequence = encoded_data[i-1:i+1]
    sequences.append(sequence)
print('Total Sequences:', len(sequences))


# In[76]:


sequences = array(sequences)
X, y = sequences[:, 0], sequences[:, 1]
X = X.reshape((X.shape[0], 1))   
y = to_categorical(y, num_classes=vocab_size)


# In[78]:


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[80]:


model.fit(X, y, epochs=100, verbose=2)


# In[82]:


def generate_seq(model, tokenizer, enter_text, n_pred):
    in_text, result = enter_text, enter_text
    for _ in range(n_pred):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded[-1:]).reshape(1, 1)  # Fix input shape
        yhat = model.predict(encoded, verbose=0)
        yhat_index = yhat.argmax()
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat_index:
                out_word = word
                break
        in_text, result = out_word, result + ' ' + out_word
    return result


# In[84]:


print(generate_seq(model, tokenizer, 'Piford', 6))
print(generate_seq(model, tokenizer, 'service', 3))


# In[ ]:




