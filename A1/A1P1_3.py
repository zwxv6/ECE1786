def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)


print_closest_words(glove['easiest'] - glove['easy'] + glove['player'])
print_closest_words(glove['luckiest'] - glove['lucky'] + glove['hunter'])
print_closest_words(glove['biggest'] - glove['big'] + glove['mountain'])
print_closest_words(glove['widest'] - glove['wide'] + glove['river'])
print_closest_words(glove['hardest'] - glove['hard'] + glove['exam'])
print_closest_words(glove['fastest'] - glove['fast'] + glove['airplane'])
print_closest_words(glove['solidest'] - glove['solid'] + glove['castle'])
print_closest_words(glove['largest'] - glove['large'] + glove['planet'])
print_closest_words(glove['happiest'] - glove['happy'] + glove['holiday'])
print_closest_words(glove['best'] - glove['good'] + glove['gift'])