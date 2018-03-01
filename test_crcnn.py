'''
Created on 1 March 2018

@author: Bhanu

'''
import tensorflow as tf
import os
import numpy as np
import yaml 
from train_crcnn import load_sents_data_semeval2010, build_model,\
    build_data_streams, Vocab
import pickle
from sklearn.metrics.classification import f1_score, classification_report
import argparse
import sys

FLAGS = None               

def main(_):
    test_model = FLAGS.model_name
    config_file = FLAGS.config_file

    with open(config_file, 'r') as rf:
        params = yaml.load(rf)
    
    seed = params.get('seed')
    tf.set_random_seed(seed)
    
    test_data_filename = params.get('test_file')

    data_dir = params.get('data_dir')
    model_dir = params.get('model_dir')
    
    print("loading data...", flush=True)
    test_data_file = os.path.join(data_dir, test_data_filename)
    dftest = load_sents_data_semeval2010(test_data_file, testset=True)
    with open(os.path.join(model_dir, params.get('label_encoder_file')), 'rb') as rf:
        le = pickle.load(rf)
    print(le.classes_)
    
    #build pos vocab
    print("loading vocab...", flush=True)
    with open(os.path.join(model_dir, params.get('vocab_file')), 'rb') as rf:
        vocab = pickle.load(rf)        
    
    is_test_labels = dftest.class_.any()

    # build input data streams
    teststream = build_data_streams(dftest, vocab.dict, 
                    params.get('sent_length'), le)
    
    labels = teststream.label 
    if labels is None:
        labels = np.zeros(teststream.sent.shape[0])
    
    #build model
    mdl = build_model(params)
    test_feed_dict = {
                    mdl.sent: teststream.sent,
                    mdl.label: labels,
                    mdl.ent1_dist: teststream.ent1_dist,
                    mdl.ent2_dist: teststream.ent2_dist,
                    mdl.dropout_keep_proba: 1.0,
                    mdl.batch_size: teststream.sent.shape[0]
                }
    
    #run the graph
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    with tf.Session() as sess:    
        sess.run(init_op)
        saver.restore(sess, os.path.join(model_dir, test_model))
        print("Restored session from %s"%test_model)   
        pred_probas, preds = sess.run([mdl.pred_probas, mdl.preds], test_feed_dict)
    
    #print scores, if test_labels known
    if is_test_labels is not None:
        l = teststream.label
        p = preds 
        
        class_int_labels = list(range(len(le.classes_)))
        target_names=le.classes_
    
        eval_score = (f1_score(l, p, average='micro'),
                      f1_score(l, p, average='macro')
                    )
        print("EVAL f1_micro {:g} f1_macro {:g}"
              .format(eval_score[0], eval_score[1]), flush=True)
        
        print("Classification Report: \n%s"%
              classification_report(l, p, 
                        labels=class_int_labels, 
                        target_names=target_names,
                    ), flush=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, 
                        help='Checkpoint Prefix of the model to be tested')
    parser.add_argument('--config_file', type=str, default=None, 
                        help='Full path of the config file')
 
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
