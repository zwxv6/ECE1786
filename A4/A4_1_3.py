from treelib import Node, Tree
tree = Tree()
tree.create_node('because', 'because')
count = 0
pcount = 0
for i in range(len(generated_tokens[0])):
  tree.create_node(tokenizer.decode(generated_tokens[0][i]), count, parent='because')
  pcount = count
  count = count + 1
  tree.create_node(f"{p1[i]:.3%}", count, parent=pcount)
  count = count + 1
  tree.create_node(f"{p2[i]:.3%}", count, parent=pcount)
  count = count + 1
  tree.create_node(f"{p3[i]:.3%}", count, parent=pcount)
  count = count + 1
print(tree.show(stdout=False))