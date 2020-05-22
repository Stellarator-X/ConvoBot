#### TODO : VOCAB SIZE HANDLING


from scipy import sparse
import sys
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def f(x):
    xm = 100
    alpha = 0.75
    return (x/xm)**(alpha) if x<xm else 1 

class Glove():
    def __init__(self, vocab_size = 1e5):
        # Init vocab size, coocc
        self.vocab_size = vocab_size
        print("Model initialised v3.5")

    def build_co_occurence(self, word_index, corpus, window_size):
        vocab_size = len(word_index) + 1

        idx_to_word = {word_index[word]:word for word in word_index}

        # Collecting indices as a sparse matrix
        self.cooccurences = sparse.lil_matrix((vocab_size, vocab_size), dtype = np.float64)
        print(f"Shape of coocc = {self.cooccurences.shape}")
        # Get the tokenized sequence * TODO implement with tokenizer
        for i, line in enumerate(corpus):

            # TODO add progress bar
            print(f"\rForming the Co-Occurence Matrix : {(100*(i+1 )/len(corpus)):0.2f}%", end = "")
            sys.stdout.flush()

            tokens = line.strip().split()
            token_ids = [word_index[word.lower()] for word in tokens]

            # extracting context words to the left

            for center_i, center_id in enumerate(token_ids):
                context_ids = token_ids[max(0, center_i - window_size): center_i]
                contexts_len = len(context_ids)

                # Adding to the coocc matrix

                for left_i, left_id in enumerate(context_ids):
                    dist = contexts_len - left_i
                    inc = 1/float(dist)

                    self.cooccurences[center_id, left_id] += inc
                    self.cooccurences[left_id, center_id] += inc
        
        print()
        print(f"Generated co-occurence matrix of shape {self.cooccurences.shape}")
        return self.cooccurences

    def cost(self, Weights, b):
        X = self.cooccurences
        W = Weights
        cost = 0
        for i in range(self.vocab_size):
            Wi = W[i, :]
            bi = b[i]
            for j in range(self.vocab_size):
                if j==i:
                    continue
                Xij = X[i, j]
                Wj = W[j+self.vocab_size, :]
                bj = b[j+self.vocab_size]

                cost += f(Xij)*((Wi.dot(Wj) + bi + bj - np.log(1+Xij))**2)
                
        return cost

    def run_iter(self, cache, beta1, beta2, eps, eta):

        X = self.cooccurences

        # Unpacking the values from the cache
        mW_prev = cache['mW']   
        mb_prev = cache['mb'] 
        vW_prev = cache['vW']
        vb_prev = cache['vb']  
        t = cache['t'] +  1
        W = cache['W']   
        b = cache['b']   

        # Calculating Gradients
        
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)

        for i in range(self.vocab_size):
            dWi = np.zeros(dW[0].shape)
            dbi = np.zeros(db[0].shape)    
            Wi = W[i, :]
            bi = b[i]
            for j in range(self.vocab_size): # TODO figure out the -2
                Xij = X[i, j]
                Wj = W[self.vocab_size + j - 1, :]
                bj = b[self.vocab_size + j - 1]

                dWi += f(Xij)*(Wj.dot( Wi.T @ Wj + bi + bj - np.log(1+Xij)))
                dbi += f(Xij)*( Wi.T @ Wj + bi + bj - np.log(1+Xij))
        dW[i] = dWi

        # Updating paramters

        mW = beta1*mW_prev + (1-beta1)*dW
        vW = beta2*vW_prev + (1-beta2)*(np.square(dW))

        m_W = mW/(1+beta1**t)
        v_W = vW/(1+beta2**t)

        mb = beta1*mb_prev + (1-beta1)*db
        vb = beta2*vb_prev + (1-beta2)*(np.square(db))

        m_b = mb/(1+beta1**t)
        v_b = vb/(1+beta2**t)

        W = W - m_W*eta/np.sqrt(v_W + eps)
        b = b - m_b*eta/np.sqrt(v_b + eps)


        cost = self.cost(W, b)

        cache = {
            'mW' : mW, 
            'mb' : mb,
            'vW' : vW,
            'vb' : vb,
            't'  : t,
            'W'  : W,
            'b'  : b,
            'cost' : cost
        }

        return cache
    
    def train(self, word_index, cooccurences, embedding_dim = 100, iterations = 20, 
                beta1 = 0.9,beta2=0.99, eta =0.01,eps = 1e-8):
        
        vocab_size = len(word_index) + 1

        history = {'cost' : [], 'W' : None}

        W = (np.random.rand(vocab_size*2, embedding_dim)-0.5)/(float(embedding_dim)+1)
        b = (np.random.rand(vocab_size*2)-0.5)/(float(embedding_dim)+1)

        cache = {
            'mW' : np.zeros(W.shape), 
            'mb' : np.zeros(b.shape),
            'vW' : np.zeros(W.shape),
            'vb' : np.zeros(b.shape),
            't'  : 0,
            'W'  : W,
            'b'  : b,
            'cost' : 0
        }


        for i in range(iterations):

            cache = self.run_iter(cache, beta1, beta2, eps, eta)
            cost = cache['cost']
            history['cost'].append(cost)
            if i >  0:
                assert cache['t'] > 0
            # TODO : Add Status Bar
            print(f"\rIteration {i+1} : Cost = {cost}", end = "")
            sys.stdout.flush()

        print()
        history['W'] = cache['W']
        return history
        
    def fit_on_corpus(self, corpus, embedding_dim = 100, iterations = 20):
        
        # Corpus needs to be a set of sentences
        
        #     build cooccurence
        #     train on cooccurence
        #     return embeddings matrix
        
        # builing the cooc
        tokenizer = Tokenizer(num_words = self.vocab_size, lower = True, oov_token = '<OOV>')
        tokenizer.fit_on_texts(corpus)
        word_index = {e:i for e,i in tokenizer.word_index.items() if i <= self.vocab_size}

        # print(word_index)
        
        self.cooccurences = self.build_co_occurence(word_index = word_index, corpus = corpus, window_size = 10)
        history = self.train(word_index, self.cooccurences, embedding_dim=embedding_dim, iterations = iterations)

        return history