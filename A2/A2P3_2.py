def splitData():
  df = pd.read_table('/content/data.tsv')

  train, test = train_test_split(df, test_size=0.2, random_state=40, stratify=df.label)
  training, validation = train_test_split(train, test_size=0.2, random_state=40, stratify=train.label)

  print('length of text of train set:',len(training.text))
  print('length of label of train set:',len(training.label))
  print('length of text of test set:',len(test.text))
  print('length of label of test set:',len(test.label))
  print('length of text of validation set:',len(validation.text))
  print('length of label of validation set:',len(validation.label))
  print('===========')
  print('length of label 0 of train set:',len(training[training['label']==0]))
  print('length of label 1 of train set:',len(training[training['label']==1]))
  print('length of label 0 of validation set:',len(validation[validation['label']==0]))
  print('length of label 1 of validation set:',len(validation[validation['label']==1]))
  print('length of label 0 of test set:',len(test[test['label']==0]))
  print('length of label 1 of test set:',len(test[test['label']==1]))
  print('===========')
  df3=pd.merge(training,test, how='inner')
  print('Do train set and test set have the same sample? ', not df3.empty)
  df4=pd.merge(training,validation, how='inner')
  print('Do train set and validation set have the same sample? ', not df4.empty)
  df5=pd.merge(test,validation, how='inner')
  print('Do test set and validation set have the same sample? ', not df5.empty)

  overFit = df.sample(n=50)
  overFit.to_csv('overfit.tsv', sep="\t")
  training.to_csv('train.tsv', sep="\t")
  test.to_csv('test.tsv', sep="\t")
  validation.to_csv('validation.tsv', sep="\t")

splitData()
