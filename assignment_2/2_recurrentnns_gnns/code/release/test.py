def get_subtrees(line):
""build subtrees""
    stack = []
    subtrees = []
    for c in line:
        if c != ')':
            stack.append(c)
        else:
            s0 = stack.pop()
            subtree = s0 + c
            while subtree.count('(') != subtree.count(')'):  
                s0 = stack.pop()
                subtree = s0 + subtree
            for c0 in subtree:
                stack.append(c0)
            subtrees.append(subtree)
    return subtrees


line = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) \
    (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .))) (2 It)"
print(get_subtrees(line))