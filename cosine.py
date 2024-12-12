import math
from collections import defaultdict

# Sample documents
documents = [
    "Earth is round.",
    "Moon is round.",
    "Day is nice."
]

# Remove stop words and lowercase conversion
stop_words = {"is"}
processed_docs = []
for doc in documents:
    words = doc.lower().split()
    processed_words = [word.strip('.,') for word in words if word not in stop_words]
    processed_docs.append(processed_words)

print("Processed Documents:", processed_docs)

# Function to calculate term frequency (TF)
def term_frequency(term, document):
    return document.count(term) / len(document)

# Function to calculate inverse document frequency (IDF)
def inverse_document_frequency(term, all_documents):
    containing_docs = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + containing_docs))

# Calculate TF-IDF vectors
tf_idf_vectors = []
all_terms = set(term for doc in processed_docs for term in doc)

for doc in processed_docs:
    tf_idf_vector = {}
    for term in all_terms:
        tf = term_frequency(term, doc)
        idf = inverse_document_frequency(term, processed_docs)
        tf_idf_vector[term] = tf * idf
    tf_idf_vectors.append(tf_idf_vector)

print("TF-IDF Vectors:", tf_idf_vectors)

# Function to calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[term] * vector2.get(term, 0) for term in vector1)
    magnitude1 = math.sqrt(sum(val**2 for val in vector1.values()))
    magnitude2 = math.sqrt(sum(val**2 for val in vector2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

# Calculate cosine similarity with Document 1
similarity_scores = []
doc1_vector = tf_idf_vectors[0]
for i in range(1, len(tf_idf_vectors)):
    similarity = cosine_similarity(doc1_vector, tf_idf_vectors[i])
    similarity_scores.append((i, similarity))

# Find the most similar document
similarity_scores.sort(key=lambda x: x[1], reverse=True)
most_similar_doc_index = similarity_scores[0][0] if similarity_scores else None

print("Similarity Scores:", similarity_scores)
print("Most Similar Document Index:", most_similar_doc_index)