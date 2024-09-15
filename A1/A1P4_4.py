def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []
    neighbor = (window - 1) / 2

    # Tokenize the input

    # TO DO

    n = 0
    for word in textlist:
        for i in range(1, int(neighbor) + 1):
            if n - i >= 0:
                X.append(w2i[word])
                T.append(w2i[textlist[n - i]])
                Y.append(1)
                X.append(w2i[word])
                T.append(np.random.randint(0, len(w2i)))
                Y.append(0)
            if n + i < len(textlist):
                X.append(w2i[word])
                T.append(w2i[textlist[n + i]])
                Y.append(1)
                X.append(w2i[word])
                T.append(np.random.randint(0, len(w2i)))
                Y.append(0)
        n = n + 1
    # Loop through each token

    # TO DO

    return X, T, Y

x,t,y=tokenize_and_preprocess_text(fl, wi, 3)
print(len(x))
print(len(t))
print(len(y))