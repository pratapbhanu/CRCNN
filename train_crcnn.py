'''
Created on 1 March 2018

@author: Bhanu

'''
import tensorflow as tf

from dataio import process_sequence, build_vocab, read_semeval2010_data,\
    read_embeddings
from model import CRCNN

import collections
import pandas as pd
import numpy as np
import pickle
import os
import yaml
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.metrics.classification import f1_score, classification_report
import argparse
import sys



FLAGS = None

DataStream = collections.namedtuple('DataStream', 
                field_names=['sent', 'label', 'ent1_dist', 'ent2_dist'])
Vocab = collections.namedtuple('Vocab', 
                field_names=['words', 'size', 'dict', 'inv_dict'])

def build_data_streams(df, vocab_dict, max_len, label_encoder):        
    sents, ent1_dist, ent2_dist = process_sequence(df, vocab_dict, max_len)
    if df.class_.any():
        labels = label_encoder.transform(df.class_.values)   
    else: #test dataframe
        labels = None        

    datastream = DataStream(sent=sents, label=labels,
                    ent1_dist=ent1_dist, ent2_dist=ent2_dist)
    
    return datastream

def build_model(params):
    mdl = CRCNN(params)
    return mdl

def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as rf:
        vocab_list = pickle.load(rf)
    vocab_size= len(vocab_list)
    vocab_dict = dict(zip(vocab_list, range(vocab_size)))
    vocab_inv_dict = dict(zip(range(vocab_size), vocab_list))
    vocab = Vocab(vocab_list, vocab_size, vocab_dict, vocab_inv_dict)
    return vocab

def load_sents_data_semeval2010(data_file, testset=False):
    
    df = read_semeval2010_data(data_file)
    
    non_other = ~(df.rel == 'OTHER')
    df['class_'] = 'OTHER'
    df.loc[non_other, 'class_'] = df.loc[non_other,:].rel
    
    return df


def main(_):
    if(FLAGS.config is None):
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'model_config.yml')
    else:
        config_file = FLAGS.config
    with open(config_file, 'r') as rf:
        params = yaml.load(rf)
    
    seed = params.get('seed')
    random_state = np.random.RandomState(seed)
    tf.set_random_seed(seed)
    
    data_dir = params.get('data_dir')
    model_dir = params.get('model_dir')
    experiment_name = params.get('experiment_name')
    train_data_filename = params.get('train_file')
    test_data_filename = params.get('test_file')
    
    #load sentences data
    print("loading data...", flush=True)
    train_data_file = os.path.join(data_dir, train_data_filename)
    test_data_file = os.path.join(data_dir, test_data_filename)
    dftrain = load_sents_data_semeval2010(train_data_file)
    dftest = load_sents_data_semeval2010(test_data_file, testset=True)
    dftraintest = pd.concat([dftrain, dftest], ignore_index=True).reset_index(drop=True)
    le = LabelEncoder().fit(dftrain.class_.values)
    params['nclass'] = len(le.classes_) 
    params['label_encoder_file'] = experiment_name+'_label_encoder.pkl'  
    #oversample class w/ only one example, hack for stratified cv
    dftrain = pd.concat([dftrain, dftrain[dftrain.rel=='ENTITY-DESTINATION(E2,E1)']],
                        ignore_index=True).reset_index(drop=True)
              
    #build vocab 
    print("building vocab...", flush=True)         
    vocab_list = build_vocab(dftraintest)
    vocab_size= len(vocab_list)
    vocab_dict = dict(zip(vocab_list, range(vocab_size)))
    vocab_inv_dict = dict(zip(range(vocab_size), vocab_list))
    vocab = Vocab(vocab_list, vocab_size, vocab_dict, vocab_inv_dict)
    params['vocab_file'] = experiment_name+'_vocab.pkl'

    #read embeddings
    print("reading embeddings...", flush=True)
    vocab_vec = read_embeddings(params['embeddings.file'], 
                                     vocab.words,
                                     params['embeddings.init_scale'],
                                     params['dtype'], random_state)
    embeddings_mat = np.asarray(vocab_vec.values, dtype=params['dtype'])
    embeddings_mat[0,:] = 0    #make embeddings of PADDING all zeros
    params['embeddings.mat.file'] = experiment_name+'_embeddings.pkl'
    
    
    #save params, vocab and embeddings in model directory for testing
    print("saving params, vocab, le and embeddings...", flush=True)
    with open(os.path.join(model_dir, experiment_name+'_params.yml'), 'w') as wf:
        yaml.dump(params, wf, default_flow_style=False)
    with open(os.path.join(model_dir, params.get('vocab_file')), 'wb') as wf:
        pickle.dump(vocab, wf)
    with open(os.path.join(model_dir, params.get('embeddings.mat.file')), 'wb') as wf:
        pickle.dump(embeddings_mat, wf) 
    with open(os.path.join(model_dir, params.get('label_encoder_file')), 'wb') as wf:
        pickle.dump(le, wf) 
    
        
    ##cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, random_state=random_state,
                                 test_size=params.get('devset_size'))
    for trainidx, devidx in sss.split(dftrain.values, dftrain.rel.values):
        cvtraindf = dftrain.iloc[trainidx,:]
        cvdevdf = dftrain.iloc[devidx,:]
        experiment_name = params.get('experiment_name') 

        tstream = build_data_streams(cvtraindf, vocab.dict, 
                    params.get('sent_length'), le

                )
        dstream = build_data_streams(cvdevdf, vocab.dict, 
                    params.get('sent_length'), le
                )        
            
        print("Training Data Shape: ", cvtraindf.shape)
        print("Dev Data Shape: ", cvdevdf.shape)
        print("Classes: ", le.classes_)        
        
        def graph_ops():
            #2. build model and define its loss minimization approach(training operation)
            mdl = build_model(params)
                
            ##defining an optimizer to minimize model's loss
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = params.get('learning_rate')
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                            momentum=0.8)
            train_op = optimizer.minimize(mdl.loss, global_step=global_step)
            
            # Summaries for loss & metrics
            loss_summary = tf.summary.scalar("loss", mdl.loss)  
            acc_summary = tf.summary.scalar("accuracy", mdl.accuracy)  
            
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            
            return mdl, global_step, train_op, loss_summary, acc_summary, init_op, \
                saver
    
        with tf.Session() as sess:   
            mdl, global_step, train_op, loss_summary, acc_summary, init_op, \
            saver = graph_ops()
            sess.run(init_op)
            
            #summaries
            ##train  summaries
            train_summary_dir = os.path.join(model_dir, "summaries", experiment_name, "train")
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph, flush_secs=3)
            ##dev summaries        
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(model_dir, "summaries", experiment_name, "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph, flush_secs=3)
            
            # train step
            def train_epoch():
                ntrain = tstream.sent.shape[0]
                bsize = params.get('batch_size')
                start = 0
                end = 0
                for start in range(0, ntrain, bsize):
                    end = start + bsize
                    if end > ntrain: 
                        end = ntrain
                    
                    train_feed_dict = {
                                       mdl.sent: tstream.sent[start:end,:],
                                       mdl.label: tstream.label[start:end],
                                       mdl.ent1_dist: tstream.ent1_dist[start:end,:],
                                       mdl.ent2_dist: tstream.ent2_dist[start:end,:],
                                       mdl.dropout_keep_proba: params.get('dropout'),
                                       mdl.batch_size: end-start
                                    }
                    sess.run([train_op, global_step, mdl.loss], train_feed_dict)

            def train_eval_step():
                sess.run(mdl.running_vars_initializer)
                train_feed_dict = {
                                   mdl.sent: tstream.sent,
                                   mdl.label: tstream.label,
                                   mdl.ent1_dist: tstream.ent1_dist,
                                   mdl.ent2_dist: tstream.ent2_dist,
                                   mdl.dropout_keep_proba: 1.0,
                                   mdl.batch_size: tstream.sent.shape[0]     
                                }
                tstep, tloss = sess.run([global_step, mdl.loss], train_feed_dict)
                sess.run(mdl.accuracy_op, train_feed_dict)
                tsummary = sess.run(train_summary_op, train_feed_dict)
                train_summary_writer.add_summary(tsummary, tstep)
                train_eval_score = sess.run(mdl.accuracy)
                return tstep, tloss, train_eval_score

            def eval_step():  
                sess.run(mdl.running_vars_initializer)
                dev_feed_dict = {
                                mdl.sent: dstream.sent,
                                mdl.label: dstream.label,
                                mdl.ent1_dist: dstream.ent1_dist,
                                mdl.ent2_dist: dstream.ent2_dist,
                                mdl.dropout_keep_proba: 1.0,
                                mdl.batch_size: dstream.label.shape[0]
                            } 
                
                dstep, dloss, preds = sess.run([global_step, mdl.loss, 
                        mdl.preds], dev_feed_dict)
                sess.run(mdl.accuracy_op, dev_feed_dict)
                dacc_ = sess.run(mdl.accuracy)
                l = dstream.label
                p = preds 
                
                class_int_labels = list(range(len(le.classes_)))
                target_names=le.classes_
                
                sess.run(mdl.accuracy_op, dev_feed_dict)
                dsummary = sess.run(dev_summary_op, dev_feed_dict)
                dev_summary_writer.add_summary(dsummary, dstep) 
                eval_score = (f1_score(l, p, average='micro'),
                              f1_score(l, p, average='macro'),
                              dacc_
                            )
                print("EVAL step {}, loss {:g}, f1_micro {:g} f1_macro {:g} accuracy {:g}"
                      .format(tstep, dloss, eval_score[0], eval_score[1], eval_score[2]), 
                      flush=True)
                official_score = eval_score[1]
                
                print("Classification Report: \n%s"%
                      classification_report(l, p, 
                                labels=class_int_labels, 
                                target_names=target_names,
                            ), flush=True)
                
                return official_score
    
            #training loop
            best_score = 0.0; best_step = 0; best_itr = 0;
            for ite in range(params.get('training_iters')):
                train_epoch()
                if ite%params.get('train_step_eval') == 0:
                    tstep, tloss, tacc_ = train_eval_step()
                
                if ite%params.get('train_step_eval') == 0:
                    print("TRAIN step {}, iteration {} loss {:g} accuracy {:g}"
                      .format(tstep, ite, tloss, tacc_), 
                      flush=True)
            
                current_step = tf.train.global_step(sess, global_step)
                if current_step % params.get('eval_interval') == 0:
                    official_score = eval_step()
                    if best_score < official_score:
                        checkpoint_prefix = os.path.join(params.get('model_dir'), 
                            "%s-score-%s"%(experiment_name, str(official_score)))
                        saver.save(sess, checkpoint_prefix, global_step=current_step)
                        
                        best_score = official_score
                        best_step = current_step
                        best_itr = ite
                    print("Best Score: %2.3f, Best Step: %d (iteration: %d)"
                          %(best_score, best_step, best_itr))  
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to the config file.')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)