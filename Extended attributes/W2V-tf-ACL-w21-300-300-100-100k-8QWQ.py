from __future__ import absolute_import
from __future__ import division
from __future__ import division
from __future__ import print_function
import collections
import math
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import statistics
import time
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize

print(time.strftime("%Y%m%d-%H%M%S"))
batch_size = 300
embedding_size = 300  # Dimension of the embedding vector.
num_sampled = 100    # Number of negative examples to sample.
num_skips = 2
data_index = 0 
outfile = ("sk-ACL-w21-300-300-100-100k"+str(batch_size)+"-"+str(embedding_size)+"-"+str(num_sampled)+"-"+str(num_skips)+time.strftime("%Y%m%d-%H%M%S")+".txt")

#************************** FUNCTIONS  *************************
def maybe_download(filename):
  if not os.path.exists(filename):
    print('no path for',filename)
  statinfo = os.stat(filename)
  print(filename)
  return filename

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary_size = 999999999
def build_dataset(words, n_words):
  count_ = [['UNK', -1]]
  count_.extend(collections.Counter(words).most_common())
  count = [['UNK', -1]]
  dictionary = dict()
  for word, c in count_:
    word_tuple = [word, c]
    if word == 'UNK': 
        count[0][1] = c
        continue
    if c > 0:
        count.append(word_tuple)
  for word, _ in count:
    dictionary[word] = len(dictionary) 
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count

  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)

  for _ in range(span):                           
      buffer.append(data[data_index])             
      data_index = (data_index + 1) % len(data)   

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
		  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

#************************** END FUNCTIONS  *************************


#**************** READ VOCABULARY DATA FROM DATA FILE

filename = maybe_download('/mnt/volume1/data.zip')  #data.zip is ACL data
vocabulary = read_data(filename)

for item in vocabulary:               #====preprocessing starts here ==convert words to lower cases
    item.lower()
vocabulary2 = vocabulary
vocabulary = [x.replace('.','') for x in vocabulary]
vocabulary = [x.replace(',','') for x in vocabulary]
stops = set(stopwords.words("english"))
vocabulary = [word for word in vocabulary if word not in stops]
del vocabulary[0]
vocabulary_size = len(vocabulary)
data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary 
vocabulary_size = len(reversed_dictionary)

#*********************** END READING VOCABULARY FROM DATA FILE





#*********************START MAIN EXECUTION TO TRAIN EMBEDDINGS FROM DATA
f = open(outfile,'w')
#print(' reversed_dictionary  ',reversed_dictionary)
print(' vocabulary_size  ',vocabulary_size)

print('win\tbatch\tembed\tsampled\tsteps\tinitial\tlowest\tstep\tlast\t% lowest\t% last',file=f)
#window={1:1}
window={1:21}





for win in window:
  skip_window = window[win]

  graph = tf.Graph()                  # ==============  Step 4: Build the graph  ===
  with graph.as_default():
                 # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

                 # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
                 # Look up embeddings for inputs.
      embeddings = tf.Variable(
                 tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)
      reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window*2)
                 # Construct the variables for the NCE loss
      nce_weights = tf.Variable(
                  tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
                 # Compute the average NCE loss for the batch.
                 # tf.nce_loss automatically draws a new sample of the negative labels each
                 # time we evaluate the loss.
    loss = tf.reduce_mean(
           tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

                # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    init = tf.global_variables_initializer()     # Add variable initializer.
  with tf.Session(graph=graph) as session:  #=========   Step 5: =========== Begin training  (running the TF graph) =====
#    steps_li=[4] 
    steps_li=[100001]
    for run in range(len(steps_li)):
      if run < len(steps_li):
          sum_ave_loss = []
          num_steps = steps_li[run]
          print(skip_window,'\t',batch_size,
              '\t',embedding_size,'\t',num_sampled,
              '\t',end='', flush=True,file=f)
          print(num_steps,'\t',end='', flush=True,file=f)         
          init.run()    # We must initialize all variables before we use them.
                                 #print("Initialized")
          average_loss = 0
          for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step == 1:
              initial = average_loss
              print(str.format("{0:.2f}", initial),'\t',end='', flush=True,file=f) #initial
            space = 1000 #change space when decreasing steps
            if step % space == 0:
              if step > 0:
                average_loss /= space        # The average loss is an estimate of the loss over the last value of space batches.
                #print("Average loss at step ", step, ": ", average_loss)
                sum_ave_loss.append(average_loss)
                last = average_loss
                average_loss = 0
          lowest = min(sum_ave_loss)
          print('%.2f'% min(sum_ave_loss),'\t',
                 (((sum_ave_loss.index(min(sum_ave_loss))))+1)*space,'\t',
                '%.2f'% last,'\t',
                '%.2f'% (((initial-lowest)/initial)*100),'\t',
                '%.2f'% (((initial-last)/initial)*100),'\t',end='', flush=True,file=f)                

          final_embeddings = normalized_embeddings.eval()     # above is last from tensorflow code except for visualization




            #************************* 8Q starts here 
    file_pairs = {1:['/mnt/volume1/cited/C0101.zip','/mnt/volume1/query/Q01-4000.zip'],
                  2:['/mnt/volume1/cited/C0102.zip','/mnt/volume1/query/Q01-5000.zip'],
                  3:['/mnt/volume1/cited/C0103.zip','/mnt/volume1/query/Q01-1000.zip'],
                  4:['/mnt/volume1/cited/C0104.zip','/mnt/volume1/query/Q01-7000.zip'],
                  5:['/mnt/volume1/cited/C0105.zip','/mnt/volume1/query/Q01-2000.zip'],
                  6:['/mnt/volume1/cited/C0106.zip','/mnt/volume1/query/Q01-1000.zip'],
                  7:['/mnt/volume1/cited/C0107.zip','/mnt/volume1/query/Q01-3000.zip'],
                  8:['/mnt/volume1/cited/C0108.zip','/mnt/volume1/query/Q01-1000.zip'],
                  9:['/mnt/volume1/cited/C0109.zip','/mnt/volume1/query/Q01-1000.zip'],
                  10:['/mnt/volume1/cited/C0110.zip','/mnt/volume1/query/Q01-4000.zip'],
                  11:['/mnt/volume1/cited/C0201.zip','/mnt/volume1/query/Q02-3000.zip'],
                  12:['/mnt/volume1/cited/C0202.zip','/mnt/volume1/query/Q02-2000.zip'],
                  13:['/mnt/volume1/cited/C0203.zip','/mnt/volume1/query/Q02-1000.zip'],
                  14:['/mnt/volume1/cited/C0204.zip','/mnt/volume1/query/Q02-2000.zip'],
                  15:['/mnt/volume1/cited/C0205.zip','/mnt/volume1/query/Q02-8000.zip'],
                  16:['/mnt/volume1/cited/C0206.zip','/mnt/volume1/query/Q02-1000.zip'],
                  17:['/mnt/volume1/cited/C0207.zip','/mnt/volume1/query/Q02-1000.zip'],
                  18:['/mnt/volume1/cited/C0208.zip','/mnt/volume1/query/Q02-1000.zip'],
                  19:['/mnt/volume1/cited/C0209.zip','/mnt/volume1/query/Q02-1000.zip'],
                  20:['/mnt/volume1/cited/C0210.zip','/mnt/volume1/query/Q02-1000.zip'],
                  21:['/mnt/volume1/cited/C0301.zip','/mnt/volume1/query/Q03-1000.zip'],
                  22:['/mnt/volume1/cited/C0302.zip','/mnt/volume1/query/Q03-2000.zip'],
                  23:['/mnt/volume1/cited/C0303.zip','/mnt/volume1/query/Q03-5000.zip'],
                  24:['/mnt/volume1/cited/C0304.zip','/mnt/volume1/query/Q03-2000.zip'],
                  25:['/mnt/volume1/cited/C0305.zip','/mnt/volume1/query/Q03-1000.zip'],
                  26:['/mnt/volume1/cited/C0306.zip','/mnt/volume1/query/Q03-8000.zip'],
                  27:['/mnt/volume1/cited/C0307.zip','/mnt/volume1/query/Q03-1000.zip'],
                  28:['/mnt/volume1/cited/C0308.zip','/mnt/volume1/query/Q03-2000.zip'],
                  29:['/mnt/volume1/cited/C0309.zip','/mnt/volume1/query/Q03-2000.zip'],
                  30:['/mnt/volume1/cited/C0310.zip','/mnt/volume1/query/Q03-5000.zip'],
                  31:['/mnt/volume1/cited/C0401.zip','/mnt/volume1/query/Q04-7000.zip'],
                  32:['/mnt/volume1/cited/C0402.zip','/mnt/volume1/query/Q04-2000.zip'],
                  33:['/mnt/volume1/cited/C0403.zip','/mnt/volume1/query/Q04-2000.zip'],
                  34:['/mnt/volume1/cited/C0404.zip','/mnt/volume1/query/Q04-2000.zip'],
                  35:['/mnt/volume1/cited/C0405.zip','/mnt/volume1/query/Q04-2000.zip'],
                  36:['/mnt/volume1/cited/C0406.zip','/mnt/volume1/query/Q04-1000.zip'],
                  37:['/mnt/volume1/cited/C0407.zip','/mnt/volume1/query/Q04-7000.zip'],
                  38:['/mnt/volume1/cited/C0408.zip','/mnt/volume1/query/Q04-7000.zip'],
                  39:['/mnt/volume1/cited/C0409.zip','/mnt/volume1/query/Q04-2000.zip'],
                  40:['/mnt/volume1/cited/C0410.zip','/mnt/volume1/query/Q04-2000.zip'],
                  41:['/mnt/volume1/cited/C0501.zip','/mnt/volume1/query/Q05-2000.zip'],
                  42:['/mnt/volume1/cited/C0502.zip','/mnt/volume1/query/Q05-2000.zip'],
                  43:['/mnt/volume1/cited/C0503.zip','/mnt/volume1/query/Q05-2000.zip'],
                  44:['/mnt/volume1/cited/C0504.zip','/mnt/volume1/query/Q05-2000.zip'],
                  45:['/mnt/volume1/cited/C0505.zip','/mnt/volume1/query/Q05-4000.zip'],
                  46:['/mnt/volume1/cited/C0506.zip','/mnt/volume1/query/Q05-2000.zip'],
                  47:['/mnt/volume1/cited/C0507.zip','/mnt/volume1/query/Q05-4000.zip'],
                  48:['/mnt/volume1/cited/C0508.zip','/mnt/volume1/query/Q05-2000.zip'],
                  49:['/mnt/volume1/cited/C0509.zip','/mnt/volume1/query/Q05-2000.zip'],
                  50:['/mnt/volume1/cited/C0510.zip','/mnt/volume1/query/Q05-2000.zip'],
                  51:['/mnt/volume1/cited/C0601.zip','/mnt/volume1/query/Q06-6000.zip'],
                  52:['/mnt/volume1/cited/C0602.zip','/mnt/volume1/query/Q06-1000.zip'],
                  53:['/mnt/volume1/cited/C0603.zip','/mnt/volume1/query/Q06-1000.zip'],
                  54:['/mnt/volume1/cited/C0604.zip','/mnt/volume1/query/Q06-2000.zip'],
                  55:['/mnt/volume1/cited/C0605.zip','/mnt/volume1/query/Q06-4000.zip'],
                  56:['/mnt/volume1/cited/C0606.zip','/mnt/volume1/query/Q06-1000.zip'],
                  57:['/mnt/volume1/cited/C0607.zip','/mnt/volume1/query/Q06-2000.zip'],
                  58:['/mnt/volume1/cited/C0608.zip','/mnt/volume1/query/Q06-1000.zip'],
                  59:['/mnt/volume1/cited/C0609.zip','/mnt/volume1/query/Q06-2000.zip'],
                  60:['/mnt/volume1/cited/C0610.zip','/mnt/volume1/query/Q06-1000.zip'],
                  61:['/mnt/volume1/cited/C0701.zip','/mnt/volume1/query/Q07-2000.zip'],
                  62:['/mnt/volume1/cited/C0702.zip','/mnt/volume1/query/Q07-2000.zip'],
                  63:['/mnt/volume1/cited/C0703.zip','/mnt/volume1/query/Q07-1000.zip'],
                  64:['/mnt/volume1/cited/C0704.zip','/mnt/volume1/query/Q07-2000.zip'],
                  65:['/mnt/volume1/cited/C0705.zip','/mnt/volume1/query/Q07-2000.zip'],
                  66:['/mnt/volume1/cited/C0706.zip','/mnt/volume1/query/Q07-2000.zip'],
                  67:['/mnt/volume1/cited/C0707.zip','/mnt/volume1/query/Q07-2000.zip'],
                  68:['/mnt/volume1/cited/C0708.zip','/mnt/volume1/query/Q07-1000.zip'],
                  69:['/mnt/volume1/cited/C0709.zip','/mnt/volume1/query/Q07-2000.zip'],
                  70:['/mnt/volume1/cited/C0710.zip','/mnt/volume1/query/Q07-1000.zip'],
                  71:['/mnt/volume1/cited/C0801.zip','/mnt/volume1/query/Q08-1000.zip'],
                  72:['/mnt/volume1/cited/C0802.zip','/mnt/volume1/query/Q08-8000.zip'],
                  73:['/mnt/volume1/cited/C0803.zip','/mnt/volume1/query/Q08-7000.zip'],
                  74:['/mnt/volume1/cited/C0804.zip','/mnt/volume1/query/Q08-8000.zip'],
                  75:['/mnt/volume1/cited/C0805.zip','/mnt/volume1/query/Q08-8000.zip'],
                  76:['/mnt/volume1/cited/C0806.zip','/mnt/volume1/query/Q08-7000.zip'],
                  77:['/mnt/volume1/cited/C0807.zip','/mnt/volume1/query/Q08-1000.zip'],
                  78:['/mnt/volume1/cited/C0808.zip','/mnt/volume1/query/Q08-7000.zip'],
                  79:['/mnt/volume1/cited/C0809.zip','/mnt/volume1/query/Q08-8000.zip'],
                  80:['/mnt/volume1/cited/C0810.zip','/mnt/volume1/query/Q08-4000.zip'],
                  81:['/mnt/volume1/cited/C0901.zip','/mnt/volume1/query/Q09-2000.zip'],
                  82:['/mnt/volume1/cited/C0902.zip','/mnt/volume1/query/Q09-2000.zip'],
                  83:['/mnt/volume1/cited/C0903.zip','/mnt/volume1/query/Q09-2000.zip'],
                  84:['/mnt/volume1/cited/C0904.zip','/mnt/volume1/query/Q09-1000.zip'],
                  85:['/mnt/volume1/cited/C0905.zip','/mnt/volume1/query/Q09-2000.zip'],
                  86:['/mnt/volume1/cited/C0906.zip','/mnt/volume1/query/Q09-2000.zip'],
                  87:['/mnt/volume1/cited/C0907.zip','/mnt/volume1/query/Q09-2000.zip'],
                  88:['/mnt/volume1/cited/C0908.zip','/mnt/volume1/query/Q09-2000.zip'],
                  89:['/mnt/volume1/cited/C0909.zip','/mnt/volume1/query/Q09-2000.zip'],
                  90:['/mnt/volume1/cited/C0910.zip','/mnt/volume1/query/Q09-2000.zip'],
                  91:['/mnt/volume1/cited/C1001.zip','/mnt/volume1/query/Q10-1000.zip'],
                  92:['/mnt/volume1/cited/C1002.zip','/mnt/volume1/query/Q10-7000.zip'],
                  93:['/mnt/volume1/cited/C1003.zip','/mnt/volume1/query/Q10-3000.zip'],
                  94:['/mnt/volume1/cited/C1004.zip','/mnt/volume1/query/Q10-8000.zip'],
                  95:['/mnt/volume1/cited/C1005.zip','/mnt/volume1/query/Q10-7000.zip'],
                  96:['/mnt/volume1/cited/C1006.zip','/mnt/volume1/query/Q10-7000.zip'],
                  97:['/mnt/volume1/cited/C1007.zip','/mnt/volume1/query/Q10-7000.zip'],
                  98:['/mnt/volume1/cited/C1008.zip','/mnt/volume1/query/Q10-1000.zip'],
                  99:['/mnt/volume1/cited/C1009.zip','/mnt/volume1/query/Q10-2000.zip'],
                  100:['/mnt/volume1/cited/C1010.zip','/mnt/volume1/query/Q10-1000.zip']}



    #**************** READ DATA FROM QUERY DOC
    a = open('W2V-tf-ACL-w21-300-300-100-100k-8Q'+time.strftime("%Y%m%d-%H%M%S")+'.txt','w')
    print('Query','\t','Cited','\t','ave word emb sim\n',file=a)
    for qc in file_pairs:
      filenamerw = maybe_download(file_pairs[qc][1])
#      print('checking q file name',file_pairs[qc][1])
      q_vocabulary = read_data(filenamerw)
      q_indices = []  
      for item in q_vocabulary:               
        item.lower()
      q_vocabulary2 = q_vocabulary
      q_vocabulary = [x.replace('.','') for x in q_vocabulary]
      q_vocabulary = [x.replace(',','') for x in q_vocabulary]
      stops = set(stopwords.words("english"))
      q_vocabulary = [word for word in q_vocabulary if word not in stops]
      del q_vocabulary[0]
      vocabulary_size_q = len(q_vocabulary) 
                                              #using word from file that were copied to q_vocabulary to get indices of those words
      for var in q_vocabulary:
          if var in dictionary:               #looking in the keys that are words
            q_indices.append(dictionary[var])      
      q_size_in_dictionary = len(q_indices)
      q_terms = []
      for terms in q_indices:
          q_terms.append(reversed_dictionary[terms])
      q_size = len(q_vocabulary)         # Set of q_icon words 

	  
      #**************** READ DATA FROM CITED DOCS 	
      filename_ced = maybe_download(file_pairs[qc][0])
      ced_vocabulary = read_data(filename_ced)
      ced_indices = []
      for item in ced_vocabulary:               
        item.lower()
      ced_vocabulary2 = ced_vocabulary
      ced_vocabulary = [x.replace('.','') for x in ced_vocabulary]
      ced_vocabulary = [x.replace(',','') for x in ced_vocabulary]
      stops = set(stopwords.words("english"))
      ced_vocabulary = [word for word in ced_vocabulary if word not in stops]      
      del ced_vocabulary[0]
      vocabulary_size_ced = len(ced_vocabulary)
                                  
                                 #using word from file that were copied to ced_vocabulary to get indices of those words
      for j in ced_vocabulary:
          if j in dictionary:               #looking in the keys that are words
            ced_indices.append(dictionary[j])      
      ced_size_in_dictionary = len(ced_indices)
      print('   ced_size_in_dictionary   ',ced_size_in_dictionary)
      ced_terms = []
      for term in ced_indices:
          ced_terms.append(reversed_dictionary[term])
      ced_size = len(ced_vocabulary)         # Set of cited words 

      # *************** from here we will compute similarity between query and each document

      q_indicesA = np.asarray(q_indices)
      q_indicesA.astype(np.int32)
      q_embeddings = tf.nn.embedding_lookup(normalized_embeddings, q_indicesA) 
                     #print('printing q_embeddings',q_embeddings)
      final_q_emb = q_embeddings.eval()
                     #print('printing final_q_emb',final_q_emb)
      ced_indicesA = np.asarray(ced_indices)
      ced_indicesA.astype(np.int32)
      ced_embeddings = tf.nn.embedding_lookup(normalized_embeddings, ced_indicesA) 
                     #print('printing ced_embeddings',ced_embeddings)			
      final_ced_emb = ced_embeddings.eval()
                    # create ced_embeddings and a similarity_ced_q to compare ced against q_
      similarity_q_ced = tf.matmul(q_embeddings, ced_embeddings, transpose_b=True)
      sim_q_ced = similarity_q_ced.eval()
                    # assess sim of every word in each and then average them or firat concatenate words and then assess similarity of resulting vectors

      p_range = ced_size_in_dictionary
      t_range = q_size_in_dictionary
      #p_range = 5
      #t_range = 3
      word_sim_sum = 0
      counter = 0  
      for p in xrange(1, p_range):
        for t in xrange(t_range):
            word_sim_sum += (float(sim_q_ced[t, p]))
            counter +=1
            #print(str(reversed_dictionary[q_indicesA[t]]),str(reversed_dictionary[ced_indicesA[p]]),float(sim_q_ced[t, p]),file=a)

      ave = word_sim_sum/counter
      print(file_pairs[qc][1],'\t',file_pairs[qc][0],'\t',ave,file=a)

    a.close()
#end of 8Q






    b = open('W2V-tf-ACL-w21-300-300-100-100k-WQ'+time.strftime("%Y%m%d-%H%M%S")+'.txt','w')
    # WHOLE Q HERE  

                #************************* CREATE LOOP

    queries  =  {'1':['/mnt/volume1/query/Q1.zip'],  
                    '2':['/mnt/volume1/query/Q2.zip'],
                    '3':['/mnt/volume1/query/Q3.zip'],
                    '4':['/mnt/volume1/query/Q4.zip'],
                    '5':['/mnt/volume1/query/Q5.zip'],
                    '6':['/mnt/volume1/query/Q6.zip'],
                    '7':['/mnt/volume1/query/Q7.zip'],			 
                    '8':['/mnt/volume1/query/Q8.zip'],
                    '9':['/mnt/volume1/query/Q9.zip'],
                    '10':['/mnt/volume1/query/Q10.zip']			 
                    }


        #**************** READ DATA FROM QUERY DOC

    print('\nQuery','\t','Cited','\t','ave word emb sim',file=b)
    for q in queries:
      filenamerw = maybe_download(queries[q][0])
      q_vocabulary = read_data(filenamerw)
      q_indices = []  
      for item in q_vocabulary:               
        item.lower()
      q_vocabulary2 = q_vocabulary
      q_vocabulary = [x.replace('.','') for x in q_vocabulary]
      q_vocabulary = [x.replace(',','') for x in q_vocabulary]
      del q_vocabulary[0]
      vocabulary_size_q = len(q_vocabulary) 
      #print(' q_vocabulary  ',q_vocabulary) #using word from file that were copied to q_vocabulary to get indices of those words
      for var in q_vocabulary:
          if var in dictionary:               #looking in the keys that are words
            q_indices.append(dictionary[var])      
      q_size_in_dictionary = len(q_indices)
      #print('   q_size_in_dictionary   ',q_size_in_dictionary)
      q_terms = []
      #print('here printing q_size_in_dictionary   ', q_size_in_dictionary)   
      for terms in q_indices:
          q_terms.append(reversed_dictionary[terms])
      q_size = len(q_vocabulary)         # Set of q_icon words 
      #print('q terms ',q_terms)
      #print('q vocabulary    ', q_vocabulary)
      #**************** READ DATA FROM CITED DOCS 

      arguments = {'1':['/mnt/volume1/cited/C'+q+' 1.zip'],  
                    '2':['/mnt/volume1/cited/C'+q+' 2.zip'],
                    '3':['/mnt/volume1/cited/C'+q+' 3.zip'],
                    '4':['/mnt/volume1/cited/C'+q+' 4.zip'],
                    '5':['/mnt/volume1/cited/C'+q+' 5.zip'],
                    '6':['/mnt/volume1/cited/C'+q+' 6.zip'],
                    '7':['/mnt/volume1/cited/C'+q+' 7.zip'],			 
                    '8':['/mnt/volume1/cited/C'+q+' 8.zip'],
                    '9':['/mnt/volume1/cited/C'+q+' 9.zip'],
                    '10':['/mnt/volume1/cited/C'+q+' 10.zip']			 
                                }
      for key in arguments:
        filesource = arguments[key][0]		
        filename_ced = maybe_download(filesource)
        ced_vocabulary = read_data(filesource)
        ced_indices = []
        for item in ced_vocabulary:               
          item.lower()
        ced_vocabulary2 = ced_vocabulary
        ced_vocabulary = [x.replace('.','') for x in ced_vocabulary]
        ced_vocabulary = [x.replace(',','') for x in ced_vocabulary]
        del ced_vocabulary[0]
        vocabulary_size_ced = len(ced_vocabulary)
        #print(' ced_vocabulary  ',ced_vocabulary)
                                    #using word from file that were copied to ced_vocabulary to get indices of those words
        for j in ced_vocabulary:
            if j in dictionary:               #looking in the keys that are words
              ced_indices.append(dictionary[j])      
        ced_size_in_dictionary = len(ced_indices)
        print('   ced_size_in_dictionary   ',ced_size_in_dictionary)
        ced_terms = []
        #print('here printing ced_size_in_dictionary   ', ced_size_in_dictionary)   
        for term in ced_indices:
            ced_terms.append(reversed_dictionary[term])
        ced_size = len(ced_vocabulary)         # Set of cited words 
        #print('ced ',ced_terms)
        #print('ced_vocabulary ',ced_vocabulary)


              # from here we will compute similarity between query and each document
        q_indicesA = np.asarray(q_indices)
        q_indicesA.astype(np.int32)
        q_embeddings = tf.nn.embedding_lookup(normalized_embeddings, q_indicesA) 
        #print('printing q_embeddings',q_embeddings)
        final_q_emb = q_embeddings.eval()
        #print('printing final_q_emb',final_q_emb)
        ced_indicesA = np.asarray(ced_indices)
        ced_indicesA.astype(np.int32)
        ced_embeddings = tf.nn.embedding_lookup(normalized_embeddings, ced_indicesA) 
        #print('printing ced_embeddings',ced_embeddings)			
        final_ced_emb = ced_embeddings.eval()
                                            # create ced_embeddings and a similarity_ced_q to compare ced against q_
        similarity_q_ced = tf.matmul(q_embeddings, ced_embeddings, transpose_b=True)
        sim_q_ced = similarity_q_ced.eval()
        # assess sim of every word in each and then average them or firat concatenate words and then assess similarity of resulting vectors
        #p_range = ced_size_in_dictionary
        #t_range = q_size_in_dictionary

        p_range = 5
        t_range = 3

        lines = p_range * t_range
        word_sim_sum = 0
        counter = 0  
        for p in xrange(1, p_range):
          for t in xrange(t_range):
              word_sim_sum += (float(sim_q_ced[t, p]))
              counter +=1
              #print(str(reversed_dictionary[q_indicesA[t]]),str(reversed_dictionary[ced_indicesA[p]]),float(sim_q_ced[t, p]),file=a)

        ave = word_sim_sum/counter
        print('Q',q,'\t','C',q,' ',key,'\t',ave,file=b)

    b.close()	  



#******************************** Step 6: Visualize the embeddings.

# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib
  import matplotlib.pyplot as plt
  matplotlib.pyplot.switch_backend('agg')
  
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reversed_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'camneg_sk.png')

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)

print(time.strftime("%Y%m%d-%H%M%S"))
