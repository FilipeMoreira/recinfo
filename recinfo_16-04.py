import nltk
from math import log, sqrt
from nltk.tokenize import RegexpTokenizer

def lower_text(text):
    return text.lower()

def tokenize_text(text, tokenizer):
    return tokenizer.tokenize(text)

def remove_stopwords(text_token, stopwords):
    return [word for word in text_token if word not in stopwords]

def preprocess_text(text, tokenizer, stopwords):
    new_text = lower_text(text)
    new_text = tokenize_text(new_text, tokenizer)
    new_text = remove_stopwords(new_text, stopwords)
    return new_text

def preprocess_texts(docs, tokenizer, stopwords):
    new_texts = []
    for doc in docs:
        new_texts.append(preprocess_text(doc, tokenizer, stopwords))
    return new_texts

def get_terms(M): # Separa todos os termos existentes da base de forma unica em um dicionario
    termos = {}
    for doc in M:
        for w in doc:
            if w not in termos:
                termos[w] = []
    return termos

def get_terms_index(terms): # indexa os termos, fazendo associacao com numeros do tipo: numero -> termo
    terms_index = ['']*len(terms)
    i = 0
    for term in terms:
        terms_index[i] = term
        i += 1
    return terms_index

def tf(term, doc):
    count = 0
    f = doc.count(term)
    return 1 + log(f, 2) if f > 0 else 0

def f_colecao(term, docs):
    return sum(1 for doc in docs if term in doc)

def idf(term, docs):
    return log(len(docs)/f_colecao(term, docs), 2)

def get_tfidf(term, doc, docs):
    return tf(term, doc) * idf(term, docs)

def build_tfidf(docs):
    termos = get_terms(docs)
    termos_index = get_terms_index(termos)
    num_termos = len(termos)
    num_docs = len(docs)
    matriz = [[0]*num_docs for i in range(num_termos)]

    for i in range(num_docs):
        for j in range(num_termos):
            termo = termos_index[j]
            matriz[j][i] = get_tfidf(termo, docs[i], docs)

    return matriz, termos_index

def print_matrix(matrix, terms_index):
    num_termos = len(terms_index)
    num_docs = len(matrix[0])
    for i in range(num_termos):
        print(terms_index[i] + ":\t", end='')
        for j in range(num_docs):
            print(str(matrix[i][j]) + "\t", end='')
        print("")

def vector_norms(matrix, terms_index):
    num_termos = len(terms_index)
    num_docs = len(matrix[0])
    norms = [1]*len(matrix[0])
    for i in range(num_termos):
        for j in range(num_docs):
            norms[j] += matrix[i][j] * matrix[i][j]
    for i in range(num_docs):
        norms[i] = sqrt(norms[i])
    return norms

#--------------------------------------------------------------------------

def generate_ranks(matrix, terms_index, docs, query_idfs, vector_norms):
    ranks = [0]*len(docs)
    for i in range(len(docs)):
        for j in range(len(query_idfs)):
            if (index(terms_index, query_idfs[j])):
                ranks[i] = query_idfs[j] * matrix[index(terms_index, query_idfs[j])][i]
    for i in range(len(vector_norms)):
        ranks[i] /= vector_norms[i]
    return ranks

def index(list, value):
    try:
        return list.index(value)
    except ValueError:
        return -1

        

docs= ['O peã e o caval são pec de xadrez. O caval é o melhor do jog.',
        'A jog envolv a torr, o peã e o rei.',
        'Caval de rodei!',
        'Polic o jog no xadrez.'
      ]


stopwords = ['a', 'o', 'e', 'é', 'de', 'do', 'no', 'são']
regex_tokenizer = RegexpTokenizer(r'\w+')

docs = preprocess_texts(docs, regex_tokenizer, stopwords)

matriz_tfidf, termos_index = build_tfidf(docs)

vector_norms = vector_norms(matriz_tfidf, termos_index)

print_matrix(matriz_tfidf, termos_index)
print(vector_norms)

q_terms = ['xadrez', 'peã', 'caval', 'torr']
query_idfs = [0]*len(q_terms)
for i in range(len(q_terms)):
    query_idfs[i] = idf(q_terms[i],docs)

print(query_idfs)

# rank computation
ranks = generate_ranks(matriz_tfidf, termos_index, docs, query_idfs, vector_norms)
print(ranks)

