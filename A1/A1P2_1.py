def compare_words_to_category(category, word):
  cosineSimilarityList = []
  for each in category:
    cosineSimilarity = torch.cosine_similarity(each, word.unsqueeze(0))
    cosineSimilarityList.append(cosineSimilarity)
  average = sum(cosineSimilarityList) / len(cosineSimilarityList)
  averageCategory =  torch.mean(torch.cat(category,0),0)
  average1 = torch.cosine_similarity(averageCategory.unsqueeze(0), word.unsqueeze(0))
  return average,average1

colour = glove['colour'].unsqueeze(0)
red = glove['red'].unsqueeze(0)
green = glove['green'].unsqueeze(0)
blue = glove['blue'].unsqueeze(0)
yellow = glove['yellow'].unsqueeze(0)
colorCategory = []
colorCategory.append(colour)
colorCategory.append(red)
colorCategory.append(green)
colorCategory.append(blue)
colorCategory.append(yellow)

givenWord = glove['greenhouse']
compare_words_to_category(colorCategory, givenWord)