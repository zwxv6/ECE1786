def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

print_closest_words(glove["cat"], n=10)

def print_closest_cosine_words(vec, n=5):
    dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0))# compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[:-n-1:-1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

print_closest_cosine_words(glove["cat"], n=10)

print_closest_words(glove['dog'], n=10)
print_closest_cosine_words(glove['dog'], n=10)

print_closest_words(glove['computer'], n=10)
print_closest_cosine_words(glove['computer'], n=10)