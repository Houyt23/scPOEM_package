from scipy.io import mmwrite
from scipy import sparse
import numpy as np
import random
import os
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
#import argparse

class Dataset(object):
    def __init__(self, link_all, len_peak, len_gene, window_size):
        self.node_number = len_peak+len_gene
        self.peaks_number = len_peak
        self.genes_number = len_gene
        node_and_counts, node_neibor_pairs= self.parse_random_walk_pairs(link_all, window_size)
        self.window_size = window_size
        self.node_and_counts = node_and_counts
        self.type_node = {'peak':np.arange(len_peak), 'gene':np.arange(len_peak, len_peak+len_gene)}
        self.node_type = self.node_type_mapping()
        self.node_neibor_pairs= node_neibor_pairs
        self.prepare_sampling_dist(equal=False)
        self.shffule()
        self.count = 0
        self.epoch = 1

    def node_type_mapping(self):
        #this method does not modify any class variables
        node_type = {}
        for type in self.type_node:
            for node in self.type_node[type]:
                node_type[node] = type

        return node_type

    def parse_random_walk_pairs(self, link_all, window_size):
        #this method does not modify any class variables
        #this will NOT make any <UKN> so don't use for NLP.
        node_and_counts = {}
        node_neibor_pairs = []
        print("window size %d"%window_size)
        cc = 0
        for sent in link_all:
            #计算出现数量
            w = 1
            nodes = sent[0]
            p = sent[1]
            for i in range(len(sent[1])):       
                if nodes[i+1] in node_and_counts:
                    node_and_counts[nodes[i+1]] += 1
                else:
                    node_and_counts[nodes[i+1]] = 1
                node_neibor_pairs.append([nodes[0], nodes[i+1], w])#1000*p[i]
                w = w * np.exp(-0.1)
                
        print("The number of unique nodes:%d"%len(node_and_counts))
        print("The number of training pairs:%d"%len(node_neibor_pairs))
        #word_word=word_word.tocsr()
        return node_and_counts, node_neibor_pairs

    def get_batch(self, batch_size):
        if self.count == len(self.node_neibor_pairs):
            self.count=0
            self.epoch+=1
        if self.count+batch_size >= len(self.node_neibor_pairs):
            pairs = self.node_neibor_pairs[self.count:len(self.node_neibor_pairs)]
            self.count = len(self.node_neibor_pairs)
        else:
            pairs = self.node_neibor_pairs[self.count:self.count+batch_size]
            self.count+=batch_size
        pairs = np.array(pairs) 
        return pairs[:,0].reshape(-1, 1).astype(int),pairs[:,1].astype(int),pairs[:,2].reshape(-1, 1)

    def shffule(self):
        random.shuffle(self.node_neibor_pairs)

    def get_negative_samples(self,pos_index,num_negatives,care_type):
        # if care_type is True it's a heterogeneous negative sampling
        pos_prob = self.sampling_prob[pos_index] 
        if not care_type:
            negative_samples = np.random.choice(self.node_number, size=num_negatives*len(pos_index), replace=True, p=self.sampling_prob)
            negative_probs = self.sampling_prob[negative_samples]
        else:
            node_type = self.node_type[pos_index]
            sampling_probs = self.type_probs[node_type]
            sampling_candidates = self.type_node[node_type]
            negative_samples_indices = np.random.choice(len(sampling_candidates),size=num_negatives*len(pos_index),replace=True,p=sampling_probs)
            
            negative_samples = sampling_candidates[negative_samples_indices]
            negative_probs = sampling_probs[negative_samples_indices]

        # print(negative_samples,pos_prob,negative_probs) .reshape((-1,1))
        return negative_samples.reshape((-1,5)),pos_prob.reshape((-1,1,1)),negative_probs.reshape((-1,1,5))
    
    def get_negative_samples2(self,num_negatives):
        negative_samples1 = np.random.choice(self.node_number, size=num_negatives, replace=False, p=self.sampling_prob)
        negative_samples2 = np.random.choice(self.node_number, size=num_negatives, replace=False, p=self.sampling_prob)
        return negative_samples1, negative_samples2

    def prepare_sampling_dist(self, equal=False):
        if equal:
            sampling_prob = np.ones(self.node_number)/self.node_number
        else:
            sampling_prob = np.ones(self.node_number)
            for i in range(self.node_number):
                if i in self.node_and_counts:
                    sampling_prob[i]+=self.node_and_counts[i]
            sampling_prob = sampling_prob**(3.0/4.0) #from http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

        #normalize the distributions
        #for caring type
        all_types = {'peak', 'gene'}
        type_probs = {}
        for node_type in all_types:
            indicies_for_a_type = self.type_node[node_type]
            type_probs[node_type] = np.array(sampling_prob[indicies_for_a_type])
            type_probs[node_type] = type_probs[node_type]/np.sum(type_probs[node_type])


        #if not careing type
        sampling_prob = sampling_prob/np.sum(sampling_prob)

        self.sampling_prob = sampling_prob
        self.type_probs = type_probs

def nce_loss_my_batch(weights,
             labels,
             inputs,
             sampled_values,
             loss_weight,
             num_true=1,
             t_distribution=False):
    #weights = [weights]
    
    # labels batch_size*1
    # inputs batch_size*1*100
    # sampled batch_size*5
    
    sampled, true_expected_count, sampled_expected_count = (
        tf.stop_gradient(s) for s in sampled_values)#batchsize*5
    positive_samples = tf.gather(weights, labels, axis=0, batch_dims=1)#batch_size*1*100
    negative_samples = tf.gather(weights, sampled, axis=0, batch_dims=1)#batch_size*5*100
    if t_distribution:
        true_logits = tf.math.log(tf.math.reciprocal(1 + tf.reduce_sum(tf.square(inputs-positive_samples), axis=-1, keepdims=True))) #batch_size*1*1 
        sampled_logits = tf.math.log(tf.math.reciprocal(1 + tf.reduce_sum(tf.square(inputs-negative_samples), axis=-1, keepdims=True))) #batch_size*5*1
        sampled_losses = tf.concat([-true_logits, sampled_logits], axis=1)#batchsize*6*1
    else:
        true_logits = tf.matmul(inputs, positive_samples, transpose_b=True)#batch_size*1*1 
        sampled_logits = tf.matmul(inputs, negative_samples, transpose_b=True) #batch_size*1*5

        out_logits = tf.concat([true_logits, sampled_logits], axis=2)
        out_labels = tf.concat([
                tf.ones_like(true_logits) / num_true,
                tf.zeros_like(sampled_logits)
            ], axis=2)

        zeros = tf.zeros_like(out_logits, dtype=out_logits.dtype)
        cond = (out_logits >= zeros)
        relu_logits = tf.where(cond, out_logits, zeros)
        neg_abs_logits = tf.where(cond, -out_logits, out_logits)  # pylint: disable=invalid-unary-operand-type
        sampled_losses = relu_logits - out_logits * out_labels + tf.log1p(tf.exp(neg_abs_logits))

    return loss_weight * tf.reduce_sum(sampled_losses, axis=2)


def build_model(VOCAB_SIZE,EMBED_SIZE,NUM_SAMPLED):

    with tf.name_scope('data'):
        loss_weight = tf.stop_gradient(tf.placeholder(tf.float32, [None, 1]))
        center_node = tf.placeholder(tf.int32, [None,1])
        neibor_node = tf.placeholder(tf.int32, [None,1])
        negative_samples = (
            tf.placeholder(tf.int32, [None, NUM_SAMPLED]),
            tf.placeholder(tf.float32, [None, 1, 1]),
            tf.placeholder(tf.float32, [None, 1, NUM_SAMPLED])
        )

    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')
    
    with tf.name_scope('loss'):
        embed = tf.gather(embed_matrix, center_node, axis=0, batch_dims=1)#tf.nn.embedding_lookup(embed_matrix, center_node, name='embed')
        
        loss = tf.reduce_mean(nce_loss_my_batch(weights=embed_matrix, 
                                                labels=neibor_node, 
                                                inputs=embed,
                                                sampled_values = negative_samples,
                                                loss_weight = loss_weight), name='loss')#tf.nn.nce_loss

        loss_summary = tf.summary.scalar("loss_summary", loss)

    return loss_weight, center_node, neibor_node, negative_samples, loss


def traning_op(loss,LEARNING_RATE):
    '''
    Return optimizer
    define one step for SGD
    '''
    #define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)#tf.keras.optimizers.SGD(LEARNING_RATE)
    return optimizer

def train(w_placeholder,
          center_node_placeholder,
          context_node_placeholder,
          negative_samples_placeholder,
          loss,
          optimizer,
          mpg,
          numwalks,
          walklength_p,
          walklength_g,
          NUM_EPOCHS,
          BATCH_SIZE,
          NUM_SAMPLED,
          care_type,
          LOG_DIRECTORY,
          LOG_INTERVAL,
          MAX_KEEP_MODEL,
          len_peak,
          len_gene,
          window_size):
    '''
    tensorflow training loop
    define SGD trining
    *epoch index starts from 1! not 0.
    '''
    care_type = True if care_type==1 else False

    # For tensorboad  
    merged = tf.summary.merge_all()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=MAX_KEEP_MODEL)#tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0 # we use this to calculate late average loss in the last LOG_INTERVAL steps
        losses = []
        writer = tf.summary.FileWriter(LOG_DIRECTORY, sess.graph)
        global_iteration = 0
        iteration = 0
        PG_link = mpg.generate_random_PPGG(numwalks, walklength_p, walklength_g)
        #GG_link = mpg.generate_random_GG()
        link_all = PG_link#+GG_link
        dataset = Dataset(link_all, len_peak, len_gene, window_size)
        epoch = 0
        while (epoch <= NUM_EPOCHS):
            current_epoch=dataset.epoch
            center_node_batch, context_node_batch, w  = dataset.get_batch(BATCH_SIZE)
            negative_samples  = dataset.get_negative_samples(context_node_batch,NUM_SAMPLED,False)
            context_node_batch = context_node_batch.reshape(-1, 1)
            loss_batch, _ ,summary_str = sess.run([loss, optimizer, merged], 
                                    feed_dict={
                                    w_placeholder:w,
                                    center_node_placeholder:center_node_batch,
                                    context_node_placeholder:context_node_batch,
                                    negative_samples_placeholder: negative_samples
                                    })
            #writer.add_summary(summary_str,global_iteration)
            total_loss += loss_batch

            # print(loss_batch)

            iteration+=1
            global_iteration+=1

            if LOG_INTERVAL > 0:
                if global_iteration % LOG_INTERVAL == 0:
                    print('Average loss: {}'.format(total_loss / LOG_INTERVAL))
                    total_loss = 0.0
                    #save model
                    model_path=os.path.join(LOG_DIRECTORY,"/model_temp.ckpt")
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" % save_path)

            if dataset.epoch - current_epoch > 0:
                print("Epoch %d end"% current_epoch)
                #save model
                model_path=os.path.join(LOG_DIRECTORY,"model_epoch%d.ckpt"%epoch)
                save_path = saver.save(sess, model_path)
                print("Model saved in file: %s" % save_path)
                print('Average loss in this epoch: {}'.format(total_loss / iteration))
                losses.append(total_loss / iteration)
                total_loss = 0.0
                iteration=0
                PG_link = mpg.generate_random_PPGG(numwalks, walklength_p, walklength_g)
                #GG_link = mpg.generate_random_GG()
                link_all = PG_link#+GG_link
                dataset = Dataset(link_all, len_peak, len_gene, window_size)
                epoch+=1
                

        model_path=os.path.join(LOG_DIRECTORY,"model_final.ckpt")
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        writer.close()

        np_node_embeddings = tf.get_default_graph().get_tensor_by_name("embedding_matrix/embed_matrix:0")
        np_node_embeddings = sess.run(np_node_embeddings)
        sparse_matrix = sparse.csr_matrix(np_node_embeddings)
        mmwrite(os.path.join(LOG_DIRECTORY,"node_embeddings.mtx"), sparse_matrix)
        plt.plot(losses)
        plt.savefig(os.path.join(LOG_DIRECTORY,'loss.png'))
        print(f'Embedding results are saved in {LOG_DIRECTORY}')

        #with open(os.path.join(LOG_DIRECTORY,"index2nodeid.json"), 'w') as f:  
        #    json.dump(dataset.index2nodeid, f, sort_keys=True, indent=4) 
    return np_node_embeddings

class MetaPathGenerator:
    def __init__(self):
        self.node_neibor_peak = dict()
        self.node_neibor_gene = dict()
        self.node_neibor_peak_p = dict()
        self.node_neibor_gene_p = dict()
        
    def trans_relation(self, peak_net, pg_net, gp_net, gene_net):
        self.len_peak = pg_net.shape[0]
        self.len_gene = pg_net.shape[1]
        self.len_nodes = self.len_peak + self.len_gene
        for i in range(len(peak_net.data)):
            if peak_net.row[i] not in self.node_neibor_peak:
                self.node_neibor_peak[peak_net.row[i]] = []
                self.node_neibor_peak_p[peak_net.row[i]] = []
            self.node_neibor_peak[peak_net.row[i]].append(peak_net.col[i])
            self.node_neibor_peak_p[peak_net.row[i]].append(peak_net.data[i])
            
        for i in range(len(gene_net.data)):
            if gene_net.row[i]+self.len_peak not in self.node_neibor_gene:
                self.node_neibor_gene[gene_net.row[i]+self.len_peak] = []
                self.node_neibor_gene_p[gene_net.row[i]+self.len_peak] = []
            self.node_neibor_gene[gene_net.row[i]+self.len_peak].append(gene_net.col[i]+self.len_peak)
            self.node_neibor_gene_p[gene_net.row[i]+self.len_peak].append(gene_net.data[i])

        for i in range(len(pg_net.data)):
            if pg_net.row[i] not in self.node_neibor_gene:
                self.node_neibor_gene[pg_net.row[i]] = []
                self.node_neibor_gene_p[pg_net.row[i]] = []
            self.node_neibor_gene[pg_net.row[i]].append(pg_net.col[i]+self.len_peak)
            self.node_neibor_gene_p[pg_net.row[i]].append(pg_net.data[i])

        for i in range(len(gp_net.data)):
            if gp_net.col[i]+self.len_peak not in self.node_neibor_peak:
                self.node_neibor_peak[gp_net.col[i]+self.len_peak] = []
                self.node_neibor_peak_p[gp_net.col[i]+self.len_peak] = []
            self.node_neibor_peak[gp_net.col[i]+self.len_peak].append(gp_net.row[i])
            self.node_neibor_peak_p[gp_net.col[i]+self.len_peak].append(gp_net.data[i])

        for p in self.node_neibor_peak_p:
            self.node_neibor_peak_p[p] = np.array(self.node_neibor_peak_p[p])/sum(self.node_neibor_peak_p[p])

        for g in self.node_neibor_gene_p:
            self.node_neibor_gene_p[g] = np.array(self.node_neibor_gene_p[g])/sum(self.node_neibor_gene_p[g])
        

    def generate_random_PPGG(self, numwalks, walklength_p, walklength_g):
        PG_list = []
        for node in range(self.len_nodes):
            if node<self.len_peak:
                walklength_p_temp = walklength_p-1
                walklength_g_temp = walklength_g
            else:
                walklength_p_temp = walklength_p
                walklength_g_temp = walklength_g-1
            for n in range(numwalks):
                notes = [node]
                node_temp = node
                path_p = []
                p = 1
                for i in range(walklength_p_temp):
                    if node_temp in self.node_neibor_peak:#peak-peak net is assymmetric
                        peaks = self.node_neibor_peak[node_temp]
                        peaks_p = self.node_neibor_peak_p[node_temp]
                        if len(peaks):
                            nump = len(peaks)
                            peakid = np.random.choice(nump, size=1, p=peaks_p)
                            node_temp = peaks[peakid[0]]
                            #p *= peaks_p[peakid]
                            notes.append(node_temp)
                            path_p.append(p)
                if len(notes)!=1:
                    PG_list.append([notes, path_p])
                notes = [node]
                node_temp = node
                path_p = []
                p = 1
                for i in range(walklength_g_temp):
                    if node_temp in self.node_neibor_gene:#peak-peak net is assymmetric
                        genes = self.node_neibor_gene[node_temp]
                        genes_p = self.node_neibor_gene_p[node_temp]
                        if len(genes):
                            numg = len(genes)
                            geneid = np.random.choice(numg, size=1, p=genes_p)
                            node_temp = genes[geneid[0]]
                            #Sp *= genes_p[geneid]
                            notes.append(node_temp)
                            path_p.append(p)
                if len(notes)!=1:
                    PG_list.append([notes, path_p])
        return PG_list
                    
    def generate_random_GG(self):
        GG_list = []
        for G1 in self.gene_cogene:
            for G2 in self.gene_cogene[G1]:
                notes = (G1+self.len_peak, G2+self.len_peak)
                GG_list.append(notes)
        return GG_list



def get_co_relation(peak_net, pg_net, gp_net, gene_net):
    mpg = MetaPathGenerator()
    mpg.trans_relation(peak_net, pg_net, gp_net, gene_net)
    return mpg

def row_normalize_direct(sparse_matrix):
    row_sparse_matrix = sparse_matrix.copy()
    row_sums = np.array(row_sparse_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1 
    row_sparse_matrix.data /= row_sums[row_sparse_matrix.row]
    return row_sparse_matrix

def col_normalize_direct(sparse_matrix):
    col_sparse_matrix = sparse_matrix.copy()
    col_sums = np.array(col_sparse_matrix.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1
    col_sparse_matrix.data /= col_sums[col_sparse_matrix.col]
    return col_sparse_matrix
	

def metapath2vec_pg(gg_net, pp_net, net_lasso, net_RF, net_XGB, dirpath, d=100, seed=0):
    os.makedirs(os.path.join(dirpath, "embedding"), exist_ok = True)
    #gene_net = sparse.coo_matrix(mmread(os.path.join(dirpath, "test/GGN.mtx"))).toarray()
    gg_net = np.abs(gg_net)
    gg_net = (gg_net + gg_net.T)/2
    gg_net[gg_net  < np.percentile(gg_net, 95)] = 0
    gg_net = sparse.coo_matrix(gg_net)
    used_genes = list(gg_net.col)

    #pp_net = sparse.coo_matrix(mmread(os.path.join(dirpath, "test/PPN.mtx")))
    pp_net = np.abs(pp_net)
    pp_net = pp_net.tocoo()
    used_peaks = list(pp_net.row)

    #net_lasso = sparse.coo_matrix(mmread(os.path.join(dirpath, "test/PGN_Lasso.mtx")))
    net_lasso = np.abs(net_lasso).tocoo()
    #net_RF = sparse.coo_matrix(mmread(os.path.join(dirpath, "test/PGN_RF.mtx")))
    net_RF = np.abs(net_RF).tocoo()
    #net_XGB = sparse.coo_matrix(mmread(os.path.join(dirpath, "test/PGN_XGB.mtx")))
    net_XGB = np.abs(net_XGB).tocoo()
    pg_net = net_lasso + net_RF + net_XGB
    used_peaks += list((pg_net).tocoo().row)
    used_peaks = np.array(list(set(used_peaks)))
    used_genes += list((pg_net).tocoo().col)
    used_genes = np.array(list(set(used_genes)))

    clip_min = np.percentile(net_lasso.data, 10)
    clip_max = np.percentile(net_lasso.data, 90)
    net_lasso.data = np.clip(net_lasso.data, clip_min, clip_max)
    pg_net_lasso = row_normalize_direct(net_lasso)
    gp_net_lasso = col_normalize_direct(net_lasso)

    clip_min = np.percentile(net_RF.data, 10)
    clip_max = np.percentile(net_RF.data, 90)
    net_RF.data = np.clip(net_RF.data, clip_min, clip_max)
    pg_net_RF = row_normalize_direct(net_RF)
    gp_net_RF = col_normalize_direct(net_RF)

    clip_min = np.percentile(net_XGB.data, 10)
    clip_max = np.percentile(net_XGB.data, 90)
    net_XGB.data = np.clip(net_XGB.data, clip_min, clip_max)
    pg_net_XGB = row_normalize_direct(net_XGB)
    gp_net_XGB = col_normalize_direct(net_XGB)

    pg_net = pg_net_lasso + pg_net_RF + pg_net_XGB
    gp_net = gp_net_lasso + gp_net_RF + gp_net_XGB
    np.savetxt(os.path.join(dirpath, "embedding/node_used_peak.csv"), used_peaks, delimiter=",")
    np.savetxt(os.path.join(dirpath, "embedding/node_used_gene.csv"), used_genes, delimiter=",")
    len_peak = pg_net.shape[0]
    len_gene = pg_net.shape[1]
    
    pg_net = pg_net.tocsr()[used_peaks, :][:, used_genes].tocoo()
    gp_net = gp_net.tocsr()[used_peaks, :][:, used_genes].tocoo()
    len_peak = pg_net.shape[0]
    len_gene = pg_net.shape[1]
	
    pp_net = pp_net.tocsr()[used_peaks, :][:, used_peaks].tocoo()
    clip_min = np.percentile(pp_net.data, 10)
    clip_max = np.percentile(pp_net.data, 90)
    pp_net.data = np.clip(pp_net.data, clip_min, clip_max)


    gg_net = gg_net.tocsr()[used_genes, :][:, used_genes].tocoo()
    clip_min = np.percentile(gg_net.data, 10)
    clip_max = np.percentile(gg_net.data, 90)
    gg_net.data = np.clip(gg_net.data, clip_min, clip_max)
    
    node_numbers = len_peak+len_gene
    numwalks = 5
    walklength_p = 3
    walklength_g = 3
    window_size = 3
    BATCH_SIZE = 32
    EMBED_SIZE = d
    NUM_SAMPLED = 5
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 100
    care_type = 0
    LOG_DIRECTORY = os.path.join(dirpath, "embedding")
    LOG_INTERVAL = -1
    MAX_KEEP_MODEL = 1#int(NUM_EPOCHS/10) + 1

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.compat.v1.reset_default_graph()
    mpg = get_co_relation(pp_net, pg_net, gp_net, gg_net)
    w_placeholder,center_node_placeholder, context_node_placeholder, negative_samples_placeholder, loss = build_model(VOCAB_SIZE = node_numbers,
                                                                                                        EMBED_SIZE = EMBED_SIZE,
                                                                                                        NUM_SAMPLED = NUM_SAMPLED)
    optimizer = traning_op(loss,
                           LEARNING_RATE=LEARNING_RATE)
    np_node_embeddings = train(w_placeholder,
                               center_node_placeholder,
                               context_node_placeholder,
                               negative_samples_placeholder,
                               loss,
                               optimizer,
                               mpg,
                               numwalks,
                               walklength_p,
                               walklength_g,
                               NUM_EPOCHS = NUM_EPOCHS,
                               BATCH_SIZE = BATCH_SIZE,
                               NUM_SAMPLED = NUM_SAMPLED,
                               care_type = care_type,
                               LOG_DIRECTORY = LOG_DIRECTORY,
                               LOG_INTERVAL = LOG_INTERVAL,
                               MAX_KEEP_MODEL = MAX_KEEP_MODEL,
                               len_peak = len_peak,
                               len_gene = len_gene,
                               window_size = window_size)
    return np_node_embeddings, used_peaks, used_genes
	

#if __name__ == "__main__":
#    gg_net = sparse.coo_matrix(mmread("/Users/qingzhi/Documents/bio_information/scPOEM/data_example/compare/S2/test/GGN.mtx"))
#    pp_net = sparse.coo_matrix(mmread("/Users/qingzhi/Documents/bio_information/scPOEM/data_example/compare/S2/test/PPN.mtx"))
#    net_lasso = sparse.coo_matrix(mmread("/Users/qingzhi/Documents/bio_information/scPOEM/data_example/compare/S2/test/PGN_Lasso.mtx"))
#    net_RF = sparse.coo_matrix(mmread("/Users/qingzhi/Documents/bio_information/scPOEM/data_example/compare/S2/test/PGN_RF.mtx"))
#    net_XGB = sparse.coo_matrix(mmread("/Users/qingzhi/Documents/bio_information/scPOEM/data_example/compare/S2/test/PGN_XGB.mtx"))
#    metapath2vec_pg(gg_net, pp_net, net_lasso, net_RF, net_XGB, "/Users/qingzhi/Documents/bio_information/scPOEM/data_example/compare/S2", d=100, seed=0)
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--dirpath", type=str, default="data_example/single/")
#    args = parser.parse_args()
#    metapath2vec_pg(args.dirpath)
    
    
    
    
    
    