import pandas as pd
import numpy as np
import os
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from datetime import datetime
now = datetime. now()
import tensorflow as tf
from transformers import BertTokenizer
from html.parser import HTMLParser
import tensorflow_addons as tfa
from tqdm import tqdm
f1 = tfa.metrics.F1Score(num_classes=2)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['TFHUB_CACHE_DIR'] = '/home/ganeshsivam/workspace/tf_cache'
#bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
#                            trainable=True,name="bert")
bert_layer = hub.KerasLayer("/home/ganeshsivam/model1/1",trainable=True,name="bert")

print("Running adam model")
df = pd.read_csv('SA_150.csv',compression='zip')
df = df[["id",'question','candidate','long_answer','target']]
val = df.tail(20000)
df = df.sample(frac=1)
def add_rows(x,df):
    d = df.loc[df['target']==1]
    for _ in range(x):
        d = d.sample(frac=1)
        df = pd.concat([df,d])
    return df
val = add_rows(10,val)
upsample = input("Upsample? y/n").lower()
if upsample == "y":
    d1 = df.loc[df['target']==1]
    d0 = df.loc[df['target']==0]
    l = int(round(len(d0)/len(d1),0))
    l = 10
    df = add_rows(l,df)
    ds = int(input("Data size? 0 if all"))
    d1 = df.loc[df['target']==1].sample(n=int(ds/2))
    d0 = df.loc[df['target']==0].sample(n=int(ds*(2/4)))
    df = pd.concat([d1,d0])
    df = df.sample(frac=1)
    df = df.sample(frac=1)

    del d1
    del d0
elif upsample == 'n':
    df = df.sample(frac=1)
    d1 = df.loc[df['target']==1]
    for _ in range(10):
        df = pd.concat([df,d1])
    #df = add_rows(8,df)

    ds = int(input("Data size? 0 if all"))
    d1 = df.loc[df['target']==1].sample(n=int(ds/5))
    d0 = df.loc[df['target']==0].sample(n=int(ds*(4/5)))
    df = pd.concat([d1,d0])
    df = df.sample(frac=1)
    #df = df.head(ds)
    del d1
    del d0


print("target 1 : {}".format(len(df.loc[df['target']==1])))
print("target 0 : {}".format(len(df.loc[df['target']==0])))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def tester(x,df,y=256):
    sian = 0
    #del df
    df.reset_index(drop=True,inplace=True)
    input_ids = []
    input_masks = []
    input_segments = []
    p = list(df.head(x).index)
    ml = y
    for i in tqdm(p):
        ans = strip_tags(str(df.loc[i,'long_answer']))
        ans = ans.replace(".","[SEP]")
        ans = ans.replace('""', '')
        ans = ans.replace('``','')
        ans = ans.replace("''", "")
        #ans = "[CLS] " + ans
        ans = " ".join(ans.split())

        qn = strip_tags(str(df.loc[i,'question']))
        qn = qn.replace(".","[SEP]")
        qn = "[CLS] " + qn + "[CLS]"
        cn = str(df.loc[i,'candidate'])
        fn = qn + " " + cn +"[CLS]" + ans
        #cn = str(df.loc[i,'candidate'])
        #cn = str(temp['candidate'])
        #cn = str(df.loc[i,'candidate'])
        #fn = fn + "[CLS]" + cn + "[SEP]"
        #print(fn)
        toks = tokenizer.tokenize(fn)
        if len(toks) > 256:
            sian +=1
        tokens = toks[:ml]
        input_ids.append(get_ids(tokens,tokenizer,ml))
        input_masks.append(get_masks(tokens,ml))
        input_segments.append(get_segments(tokens,ml))


    inputs = [tf.convert_to_tensor(input_ids,dtype=float),tf.convert_to_tensor(input_masks,dtype=float),tf.convert_to_tensor(input_segments,dtype=float)
            ]

    output = tf.one_hot(tf.convert_to_tensor(df.head(x)['target'],dtype=tf.int32),depth=2)
    print("{} lines more than 256 tokens eh how sian".format(sian))
    return inputs,output

print(len(df))
df.reset_index(inplace=True)
val.reset_index(inplace=True)
inp,out = tester(len(df),df)
p = int(round(len(df)/100,0))
val_inp, val_out = tester(p,val)




#bs = int(input("Batch size?"))
#epochs = int(input("Epochs?"))

print("target 1 : {}".format(len(df.loc[df['target']==1])))
print("target 0 : {}".format(len(df.loc[df['target']==0])))
del df
del val

def load_weights(path):
    max_seq_length = 256  # Your choice here.

    #unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    bert_layer1 = bert_layer
    pooled_output, sequence_output = bert_layer1([input_ids, input_mask, segment_ids])


    x = tf.keras.layers.Dense(128,activation=tf.nn.relu,name='combined_layer_2',kernel_initializer=tf.random_normal_initializer)(pooled_output)
    #y = tf.keras.layers.Dense(64,activation=tf.nn.relu,name='combined_layer_3',kernel_initializer=tf.random_normal_initializer)(x)
    #z = tf.keras.layers.Dense(64,activation=tf.nn.relu,name='combined_layer_4',kernel_initializer=tf.random_normal_initializer)(y)
    logits = tf.keras.layers.Dense(2,activation='softmax')(x)
    #logits = tf.keras.layers.Dense(2,activation='softmax')(x)
    #output = tf.keras.layers.Dense(1, activation=tf.nn.sparse_softmax_cross_entropy_with_logits(),dtype=tf.int32)(x)
    #output = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(logits)
    model = Model(inputs=[input_ids,input_mask,segment_ids], outputs=logits)
    model.load_weights(path)
    adam = tf.keras.optimizers.Adam()
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.1)
    def what(y_true,y_pred):
        return tf.math.count_nonzero(y_pred,dtype=float)
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    hn = tf.keras.losses.Hinge()
    model.compile(optimizer=sgd,
                 #loss='sparse_categorical_crossentropy',
                 loss = loss,
                 metrics=['accuracy',f1])

    print("Fitting model")
    model.fit(x=[i for i in inp],y=out,epochs=1,verbose=1,batch_size=16,validation_data=([v for v in val_inp],val_out))
    try:
        print("sa_model_weights_latest_bergy.h5")
        model.save_weights("sa_model_weights_latest_bergy.h5")
        print("Model weights saved")

    except:
        print("Saving model weights as my_model_weights.h5")
        model.save_weights("my_model_weights.h5")
        print("Model weights saved")



path = "sa_model_weights_10_bergy.h5"
load_weights(path)
def build_model(batch_size=16,epochs=1):
    max_seq_length = 256  # Your choice here.

    #unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    bert_layer1 = bert_layer
    pooled_output, sequence_output = bert_layer1([input_ids, input_mask, segment_ids])


    x = tf.keras.layers.Dense(128,activation=tf.nn.relu,name='combined_layer_2',kernel_initializer=tf.random_normal_initializer)(pooled_output)
    #y = tf.keras.layers.Dense(64,activation=tf.nn.relu,name='combined_layer_3',kernel_initializer=tf.random_normal_initializer)(x)
    #z = tf.keras.layers.Dense(64,activation=tf.nn.relu,name='combined_layer_4',kernel_initializer=tf.random_normal_initializer)(y)
    logits = tf.keras.layers.Dense(2,activation='softmax')(x)
    #logits = tf.keras.layers.Dense(2,activation='softmax')(x)
    #output = tf.keras.layers.Dense(1, activation=tf.nn.sparse_softmax_cross_entropy_with_logits(),dtype=tf.int32)(x)
    #output = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(logits)
    model = Model(inputs=[input_ids,input_mask,segment_ids], outputs=logits)

    #from keras.optimizers import SGD
    #opt = SGD(lr=0.01, momentum=0.1)
    adam = tf.keras.optimizers.Adam()
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.1)
    m = tf.keras.metrics.Accuracy()
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=sgd,
                 #loss='sparse_categorical_crossentropy',
                 loss = loss,
                 metrics=['accuracy',f1])

    print("Fitting model")
    model.fit(x=[i for i in inp],y=out,epochs=epochs,verbose=1,batch_size=batch_size,validation_data=([v for v in val_inp],val_out))
    try:
        model.save_weights("sa_model_weights_10_bergy.h5")
        print("sa_model_weights_10_bergy.h5")

    except:
        model.save_weights("my_model_weights.h5")
#build_model()
