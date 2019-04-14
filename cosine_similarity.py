text = ["London Paris London","Paris Paris London"]

from sklearn.feature_extraction.text import CountVectorizer #count vectorizer class
cv = CountVectorizer()                                      #initializing an object
count_matrix = cv.fit_transform(text)                       #gives a sparse matrix

print(cv.get_feature_names())                               #gives us features that have been fed to the fit transform method
print(count_matrix.toarray())                               #makes it more readable

from sklearn.metrics.pairwise import cosine_similarity      #cosine similarity is a method
similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)