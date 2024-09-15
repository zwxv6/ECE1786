def prepare_textsPlus(text, min_frequency=3):

    # Get a callable object from spacy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")

    # Some text cleaning. Do it by sentence, and eliminate punctuation.
    lemmas = []
    for sent in sent_tokenize(text):  # sent_tokenize separates the sentences
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    # Count the frequency of each lemmatized word
    freqs = Counter()  # word -> occurrence
    for w in lemmas:
        freqs[w] += 1

    vocab = list(freqs.items())  # List of (word, occurrence)
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    print(vocab)
    print(len(vocab))
    # per Mikolov, don't use the infrequent words, as there isn't much to learn in that case


    # Filter words that appear too frequently and too frequently
    frequent_vocab = list(filter(lambda item: item[1]>=min_frequency and item[1]<=2200, vocab))
    print(frequent_vocab)
    print(len(frequent_vocab))


    # Create the dictionaries to go from word to index or vice-verse

    w2i = {w[0]:i for i,w in enumerate(frequent_vocab)}
    i2w = {i:w[0] for i,w in enumerate(frequent_vocab)}

    # Create an Out Of Vocabulary (oov) token as well
    w2i["<oov>"] = len(frequent_vocab)
    i2w[len(frequent_vocab)] = "<oov>"

    # Set all of the words not included in vocabulary nuas oov
    filtered_lemmas = []
    for lem in lemmas:
        if lem not in w2i:
            filtered_lemmas.append("<oov>")
        else:
            filtered_lemmas.append(lem)

    return filtered_lemmas, w2i, i2w

flp, wip, iwp = prepare_textsPlus(lines)