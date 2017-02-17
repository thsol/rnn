# Split words in sentence & tokenise
def tokeniser(x):
    arr = []
    regex = re.compile("[\w]+(?=n't)|n't|\'m|\'ll|[\w]+|[.?!;,\-\(\)—\:']")
    for i in x:
        j = re.findall(regex, i)
        arr.append(j)
    return(arr)

def outDict(dict, filename, sep):
    with open(filename, "w") as f:
        for i in dict.keys():            
            f.write(i + " " + sep.join([str(x) for x in dict[i]]) + "\n")

            
def unique_words(lines):
    return set(chain(*(line.split() for line in lines if line)))

relw2v = {}
w2v = {}
_arr = {}
isvocab = {}

def word2vec(f):
    for line in open(f):
        l = {line.split()[0]: np.array((line.split()[1:]),dtype=float)}
        # w2v = glove words
        w2v.update(l)
     
    for w,v in w2v.items():
        # if word in glove matches with unique tokenised words (exported .txt file) 
        # - then update as a relevant word to dictionary relw2v
        if w in uniq_train_w:
            relw2v.update({w:v})
    return relw2v

thefile = open('tokenised_arr.txt', 'w')

for i in range(len(data_train)):
    corpus = data_train[i]['story']
    _array = tokeniser(corpus)
    w = set()
    result = ''
    for sentence in _array:
        for word in sentence:
            word = word.lower()
            if word not in w:
                w.add(word)
                result = result + word + ' '
    thefile.write("%s\n" % result)
    
with open('tokenised_arr.txt', 'r') as f:
    uniq_train_w = (unique_words(f))
    
_relw2v = word2vec('glove.6B.50d.txt')

outDict(_relw2v, 'output.txt', ' ')

# about 25889
    # add <OOV> to vocab, add <OOV> as random vector
isvocab = {'<OOV>': 1}
num = len(isvocab.items())+1
np.set_printoptions(suppress=True)
_v = np.array([-0.00001] * 50)
_relw2v.update({"<OOV>":_v})
new_data = {}
    
for w,v in _relw2v.items():
    if (w not in isvocab.keys()):
        num+=1
        isvocab.update({w:num})
        new_data.update({w:v})    
    else:
        new_data.update({w:v})

__relw2v = sorted(new_data.items(), key=lambda x: isvocab.get(x[0]))


# create matrix
   
vocab_dim = 50 # dimensionality of your word vectors
n_symbols = len(isvocab) + 1 # adding 1 to account for 0th index (for masking)
embedding_weights = np.zeros((n_symbols,vocab_dim))

inc = 0
for i in __relw2v:
    for _set in i:
        if (type(_set) == np.ndarray):
            vec_arr = _set
            embedding_weights[inc,:] = vec_arr
            inc+=1

# actually, padding just causes error in model : ValueError: Dimensions 25882 and 30000 are not compatible

# add the padding
#pad = np.array([0.0] * 50)
#_range = 40000 - len(embedding_weights)
#for i in range(_range):
#    embedding_weights = np.vstack((embedding_weights,pad))
    
embedding_weights = np.float32(embedding_weights)
print(len(embedding_weights))