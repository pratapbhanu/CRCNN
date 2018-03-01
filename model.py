'''
Created on 1 Mar 2018

@author: Bhanu

'''

import tensorflow as tf
import os
import pickle

def variable_on_device(name, shape, initializer, device):
    with tf.device(device):
        v = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return v

class CRCNN:
    '''
    Implementation of Classification by Ranking CNN. 
    Refer: Classifying Relations by Ranking with Convolutional Neural Networks.
    C´ıcero Nogueira dos Santos, Bing Xiang, Bowen Zhou
    '''
    
    def __init__(self, params):
        self.is_training = False if params.get('mode') == 'INFER' else True
    
        #graph inputs
        self.sent = tf.placeholder(dtype=tf.int32, 
                        shape=[None, params.get('sent_length')], name='sent')
        self.label = tf.placeholder(dtype=tf.int32, 
                        shape=[None], name='label')
        self.ent1_dist = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                        name='ent1_dist')
        self.ent2_dist = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                        name='ent2_dist')
        self.dropout_keep_proba = tf.placeholder(dtype=params.get('dtype'), 
                                        name='dropout')
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
    
        self.scope = tf.get_variable_scope()
        
        #graph variables for each of the layers in cnn architecture
        ## Embeddings layer
        with tf.device(params.get('device')):
            with open(os.path.join(params.get('model_dir'), 
                    params.get('embeddings.mat.file')), 'rb') as rf:
                embeddings_mat = pickle.load(rf)
            self.sent_embedding = tf.get_variable(name="W_s", 
                    trainable=params['embeddings.tune'],
                    initializer=tf.constant(embeddings_mat))
            self.dist_embedding = tf.get_variable(name='W_d', 
                    shape=[2*params.get('sent_length')-1, params['embeddings.dist.dim']],
                    initializer=tf.random_uniform_initializer(
                                    -params["embeddings.init_scale"],
                                    params["embeddings.init_scale"]))
        
        ##embeddings look-up operation
        sent_input = tf.nn.embedding_lookup(params=self.sent_embedding, ids=self.sent)
        ent1_dist_input = tf.nn.embedding_lookup(params=self.dist_embedding, 
                            ids=self.ent1_dist)
        ent2_dist_input = tf.nn.embedding_lookup(params=self.dist_embedding,
                            ids=self.ent2_dist)
        conv_input = tf.concat([sent_input, ent1_dist_input, ent2_dist_input], 
                                axis=-1)
        conv_input = tf.expand_dims(conv_input, -1, name='input')
        input_dim = params.get('embeddings.dim') + 2*params.get('embeddings.dist.dim') 
        
        ##Convolutional & pooling Layers
        with tf.variable_scope('conv') as scope:
            pool_tensors = []
            for w_size in params.get('window'):
                fw = variable_on_device(name='fw_'+str(w_size),
                    shape=[w_size, input_dim, 1, params.get('nfeature_map')], 
                    initializer=tf.random_uniform_initializer(
                                -params["embeddings.init_scale"],
                                params["embeddings.init_scale"]),
                    device=params.get('device'))
                conv = tf.nn.conv2d(input=conv_input, filter=fw, 
                            strides=[1,1,1,1], padding='VALID')
                biases = variable_on_device(name='biases_'+str(w_size), 
                            shape=[params.get('nfeature_map')], 
                            initializer=tf.constant_initializer(0.0),
                            device=params.get('device'))
                bias = tf.nn.bias_add(conv, biases)
                relu = tf.nn.relu(bias, name=scope.name)
                conv_len = relu.get_shape()[1]
                pool = tf.nn.max_pool(relu, ksize=[1,conv_len,1,1], 
                            strides=[1,1,1,1], padding='VALID')
                pool = tf.squeeze(pool,squeeze_dims=[1,2]) 
                pool_tensors.append(pool)
        
        ##pooling & concatenation operation
        num_filters = len(params.get('window'))
        pool_size = num_filters * params.get('nfeature_map')
        pool_layer = tf.concat(pool_tensors, -1, name='pool')
        pool_flat = tf.reshape(pool_layer, [-1, pool_size])
                
        ##Dropout Layer
        pool_dropout = tf.nn.dropout(pool_flat, keep_prob=self.dropout_keep_proba)
        
        ##Dense Projection Layer
        input_ = pool_dropout
        input_size = pool_size
        with tf.variable_scope('fc') as scope:
            W = variable_on_device(name='W', shape=[input_size, params.get('nclass')],
                                initializer=tf.random_uniform_initializer(
                                    -params["embeddings.init_scale"],
                                    params["embeddings.init_scale"]),
                                device=params.get('device'))

            biases = variable_on_device(name='biases', shape=[params.get('nclass')], 
                            initializer=tf.constant_initializer(0.01),
                            device=params.get('device'))
            ##dense layer operation
            self.logits = tf.nn.bias_add(tf.matmul(input_, W), biases)

        ##softmax 
        self.pred_probas = tf.nn.softmax(self.logits, name='class_proba')
        self.preds = tf.argmax(self.pred_probas, axis=-1, name='class_prediction')
        
        #loss using graph's output(s)
        self._loss = self._loss(params)
        self.l2loss = self._l2loss(params)
        self.loss = self._loss + self.l2loss
        
        #evaluation metric using graph's output(s)
        ##precision & recall evaluation metric
        with tf.variable_scope('eval_metric') as scope:
            self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.label, 
                                self.preds, name='accuracy')
            # Isolate the variables stored behind the scenes by the metric operation
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

        # Define initializer to initialize/reset running eval_metric variables
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    def _loss(self, params):
        return CRCNN.ranking_loss(params, self.label, self.logits, 
                                        self.batch_size)
    
    
    def _l2loss(self, params):
        vars_   = [v for v in tf.trainable_variables() if 'biases' not in v.name 
                   and 'W_d' not in v.name and 'W_s' not in v.name]
        l2loss = tf.multiply(tf.add_n([ tf.nn.l2_loss(v) for v in vars_ ]),
                        params.get('l2'), name='l2loss')
        return l2loss
    
    @staticmethod
    def ranking_loss(params, labels, logits, batch_size):        
        lm = tf.constant(params.get('lm')) #lambda
        m_plus = tf.constant(params.get('margin_plus'))
        m_minus = tf.constant(params.get('margin_minus'))

        L = tf.constant(0.0)
        i = tf.constant(0)
        cond = lambda i, L: tf.less(i, batch_size)

        def loop_body(i, L): 
            cplus = labels[i] #positive class label index
            #taking most informative negative class, use 2nd argmax
            _, cminus_indices = tf.nn.top_k(logits[i,:], k=2)
            cminus = tf.cond(tf.equal(cplus, cminus_indices[0]),
                             lambda: cminus_indices[1], lambda: cminus_indices[0])
            
            splus = logits[i,cplus] #score for gold class
            sminus = logits[i,cminus] #score for negative class
            
            l = tf.log((1.0+tf.exp((lm*(m_plus-splus))))) + \
                tf.log((1.0+tf.exp((lm*(m_minus+sminus)))))
            
            return [tf.add(i, 1), tf.add(L,l)]

        _, L = tf.while_loop(cond, loop_body, loop_vars=[i,L])
        nbatch = tf.to_float(batch_size)
        L = L/nbatch
        return L
    
    

        