from treelib import Node, Tree
import pandas as pd

def getTree(parse):
    #create a tree data structure and return
    letterstack = []
    parentstack = []
    leaves = []
    tree = Tree()
    idcounter=1
    firstime=1
    j=0
    for letter in parse:
        #print(j)
        j+=1
        if letter=='(':
            #print("open")
            word = "".join(letterstack)
            word = word.strip()
            firstime=1
            if len(word)!=0:

                data = [False, False, False]
                if word[0]=='/':
                    data[0] = True
                    word = word.strip('/')
                elif word[0]=='^':
                    data[1] = True
                    word = word.strip('^')
                elif word[0]=='|':
                    data[2] = True
                    word = word.strip('|')

                #node with id and word
                element = dict({'id':idcounter,'word':word})
                idcounter += 1

                if(len(parentstack)==0):
                    #create root node
                    tree.create_node(element['word'], element['id'], data=data)
                    #print(element['word'], element['id'])
                else:
                    #create other nodes
                    tree.create_node(element['word'], element['id'], parent=parentstack[-1]['id'], data=data)
                    #print(element['word'], element['id'], parentstack[-1]['id'])
                #push word to parent stack
                parentstack.append(element)

            #empty letter stack
            letterstack = []
        elif letter==')':
            #print("close")
            if(firstime):
                #add node and leaf
                word = "".join(letterstack)
                word = word.strip()
                [node, leaf] = word.split(' ')
                node = node.strip()
                leaf = leaf.strip()
                node_data = [False, False, False]
                leaf_data = [False, False, False]
                if node[0]=='/':
                    node_data[0] = True
                    node = node.strip('/')
                elif node[0]=='^':
                    node_data[1] = True
                    node = node.strip('^')
                elif node[0]=='|':
                    node_data[2] = True
                    node = node.strip('|')


                if leaf[0]=='/':
                    leaf_data[0] = True
                    leaf = leaf.strip('/')
                elif(leaf[0]=='^'):
                    leaf_data[1] = True
                    leaf = leaf.strip('^')
                elif(leaf[0]=='|'):
                    leaf_data[2] = True
                    leaf = leaf.strip('|')

                node = dict({'id':idcounter,'word': node})
                idcounter += 1
                leaf = dict({'id':idcounter,'word': leaf})
                idcounter += 1
                if(len(parentstack)==0):
                    tree.create_node(node['word'], node['id'], data = node_data)
                else:
                    tree.create_node(node['word'], node['id'], parent=parentstack[-1]['id'], data = node_data)
                tree.create_node(leaf['word'], leaf['id'], parent=node['id'], data = leaf_data)
                leaves.append(leaf['id'])
                #print(node['word'],node['id'],parentstack[-1]['id'])
                #print(leaf['word'],leaf['id'])
            else:
                #pop parent
                #print("pop")
                parentstack.pop()
            #empty letter stack
            letterstack = []
            firstime=0
        elif letter=='\n' or letter=='\t':
            letterstack=letterstack
        else:
            letterstack.append(letter)

    return tree, leaves

def getDependencyTree(parse, tokens, treeid):
        #initialize tree
        tree = Tree()
        #create root node
        tree.create_node('ROOT', '0', data='root')
        data = pd.DataFrame(parse)
        tokens = pd.DataFrame(tokens)
        #print(tokens)
        stack = [0]
        rootvisited = 0
        #loop through each dependency link
        while(len(stack)>0):
            current = stack.pop(0)
            subdata = data[data['governor']==current]
            data = data.drop(data[data['governor']==current].index)
            new = subdata['dependent'].unique()
            stack.extend(new)
            #print(new)
            if(len(subdata)>0):
                subdata = subdata.to_dict(orient='records')
                for dep in subdata:
                    parent = dep['governorGloss']
                    parentnum = dep['governor']
                    if(rootvisited):
                        parentid = str(parentnum) + '_' + str(treeid)
                    else:
                        parentid = str(parentnum)
                        rootvisited = 1
                        
                    child = dep['dependentGloss']
                    childnum = dep['dependent']
                    childid = str(childnum) + '_' + str(treeid)
                    dependency = dep['dep']
                    dependencyid = str(parentnum) + '_' + str(childid)
                    POS = tokens[tokens['index']==childnum]['pos'].as_matrix()[0]

                    tree.create_node(dependency, dependencyid, parent=parentid, data=['dep', dependencyid])
                    tree.create_node(POS, 'pos_'+str(childid), parent=dependencyid, data=['pos', 'pos_'+str(childid)])
                    tree.create_node(child, str(childid), parent='pos_'+str(childid), data=['word', str(childid)])


        return tree
