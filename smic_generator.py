from pycorenlp import StanfordCoreNLP
from treelib import Node, Tree
import treeparser
import itertools
from rouge import Rouge
import json
import numpy as np
import pandas as pd
import pickle
import scipy.stats
from simplenlg import changePOS, changePlurality
#import lm

nlp = StanfordCoreNLP('http://localhost:9000')
rules = 'rules.json'

def getRougeScore(gold, sys):
    rouge = Rouge()
    scores = rouge.get_scores(gold, sys)
    return scores



class smic_generator(object):
    """docstring for smic_generator."""
    def __init__(self, acceptability_type=None, confidence = 0.95):
        super(smic_generator, self).__init__()
        if(acceptability_type is not None):
            self.avgchange = pickle.load( open( "rougefilter.p", "rb" ) )
            [self.data_mean, self.data_std, self.data_totalmean] = pickle.load( open( "rouge_stats.p", "rb" ) )
        #type of acceptability measure valid options are:
        # 1. absolute
        # 2. difference
        # 3. fitting_absolute
        # 4. absolute_global
        # 5. fitting_absolute_global
        self.acceptability_type = acceptability_type
        #set confidence
        self.confidence = confidence

    def generate(self,source, gold, smc, smcscore, sourceid, judgeid, debugrule=0):
        #parse smc using corenlp
        sys_parse = nlp.annotate(smc, properties={
          'annotators': 'tokenize,ssplit,pos,depparse,parse',
          'outputFormat': 'json'
        })
        if(gold is not None):
            #parse gold standard using corenlp
            gold_parse = nlp.annotate(gold, properties={
              'annotators': 'tokenize,ssplit,pos,depparse,parse',
              'outputFormat': 'json'
            })
            sents_gold = gold_parse['sentences']

        #parse source using corenlp
        source_parse = nlp.annotate(source, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })

        #print(sys_parse.keys())
        sents = sys_parse['sentences']

        sents_source = source_parse['sentences']
        #print(dparse)
        #print(smc)
        #print(sys_parse['sentences'][0])


        total = []
        with open(rules) as data_file:
            data = json.load(data_file)
        #smcLMScore = lm.getLMScore(smc)
        smcLMScore = 1
        #get dependency parse convert it to tree and store in list for smc
        dtrees = []
        treeid = 1
        for sent in sents:
            cparse = sent['parse']
            dparse = sent['basicDependencies']
            tokens = sent['tokens']
            #if multiple sentences create multiple dependency trees
            dtrees.append(treeparser.getDependencyTree(dparse, tokens, treeid))
            treeid += 1


        #get dependency parse convert it to tree and store in list for gold standard
        dtrees_gold = []

        if(gold is not None):
            for sent_gold in sents_gold:
                cparse = sent_gold['parse']
                dparse = sent_gold['basicDependencies']
                tokens = sent_gold['tokens']
                #if multiple sentences create multiple dependency trees
                dtrees_gold.append(treeparser.getDependencyTree(dparse, tokens, treeid))
                treeid += 1

        dtrees_source = []

        for sent_source in sents_source:
            cparse = sent_source['parse']
            dparse = sent_source['basicDependencies']
            tokens = sent_source['tokens']
            #if multiple sentences create multiple dependency trees
            dtrees_source.append(treeparser.getDependencyTree(dparse, tokens, treeid))
            treeid += 1

        total_coverage = []
        #for each rule in rule.json
        for rule in data:
            #for debugging rules, debugrule is set to the id of the rule which needs to be deubged
            if(rule["id"]==debugrule or debugrule==0 ):

                matched=[]
                matched_gold=[]
                if rule['from']=='SMC':
                    #each rule may contain subset of rules whose matches can be substituted
                    subsets = rule['match'].split(",")

                    for subset in subsets:
                        #each subset may contain different matching criteria but the matched rules may intermix
                        matches = subset.split(" or ")
                        submatched=[]
                        for match in matches:
                            match = match.strip()
                            #perform mathing of rules with the dependency parse of given sentences
                            submatch = self.getMatches(match, dtrees)
                            #we can intermix submatches
                            submatched.extend(submatch)
                        #matched will contain list of matches for each subset
                        matched.append(submatched)
                        #print("matched",matched)
                else:
                    #match either reference/gold summary or source to smc
                    dtrees_secondary = ""
                    if(rule['from']=='GOLD'):
                        if(gold is None):
                            continue
                        dtrees_secondary = dtrees_gold
                    else:
                        dtrees_secondary = dtrees_source
                    subsets = rule['match'].split(",")
                    for subset in subsets:
                        #each subset may contain different matching criteria but the matched rules may intermix
                        matches = subset.split(" or ")
                        submatched=[]
                        submatched_gold=[]
                        for match in matches:
                            match = match.strip()
                            #perform mathing of rules with the dependency parse of given sentences
                            submatch = self.getMatches(match, dtrees)
                            #repeat for matches in gold standard
                            submatch_gold = self.getMatches(match, dtrees_secondary)
                            #we can intermix submatches
                            submatched.extend(submatch)
                            #repeat for matches in gold standard
                            submatched_gold.extend(submatch_gold)
                        #matched will contain list of matches for each subset
                        matched.append(submatched)
                        #repeat for matches in gold standard
                        matched_gold.append(submatched_gold)


                #get sentences after matching
                sentences = []
                count = 0
                if rule['from']=='SMC':
                    #for each set of matches corresponding to each submatch generate all possible sentences
                    for match in matched:
                        if(len(match)>0):
                            #function which generates the sentences
                            sentence, coverage = self.generateSentence(dtrees, [], match, [], gold, int(rule['hardmatch']), smcscore, sourceid, smcLMScore, rule["id"], smc, judgeid, rule['from'], total_coverage)

                            sentences.extend(sentence)
                            count += 1
                else:
                    dtrees_secondary = ""
                    if(rule['from']=='GOLD'):
                        dtrees_secondary = dtrees_gold
                    else:
                        dtrees_secondary = dtrees_source
                    for match, match_gold in zip(matched, matched_gold):
                        if(len(match)>0 and len(match_gold)>0):
                            #function which generates the sentences
                            sentence, coverage = self.generateSentence(dtrees, dtrees_secondary, match, match_gold, gold, int(rule['hardmatch']), smcscore, sourceid, smcLMScore, rule["id"], smc, judgeid, rule['from'], total_coverage)
                            if(len(coverage)!=0):
                                total_coverage.extend(coverage)
                            sentences.extend(sentence)
                            count += 1

                total.extend(sentences)
        return total

    def getMatches(self, match, dtrees):
        tree, leaves = treeparser.getTree(match)
        #tree.show(line_type="ascii-em")
        matched=[]
        #get root node of searcher
        node = tree.get_node(1)
        #maintain a dtree id
        dtree_id=0
        #for each dependency tree of each sentence
        for dtree in dtrees:

            for target_node in dtree.expand_tree():
                target_node = dtree.get_node(target_node)
                #if(target_node.data[0]!='p'):
                #    continue
                #print(node.tag, target_node.tag)
                if (node.tag == "." or node.tag == target_node.tag):
                    subtree = Tree()
                    subtree.create_node('ROOT', 0, data=True)

                    subtree.create_node(target_node.tag, target_node.identifier, parent=0, data=[list(target_node.data),list(node.data)])
                    #dtree.show(line_type="ascii-em")
                    #tree.show(line_type="ascii-em")
                    matchsuccess, subtree, word_target = self.checkmatch(node, target_node, subtree, dtree, tree)
                    #print("did i catch the right word ", word_target)
                    if matchsuccess:
                        matched.append([subtree, dtree_id, word_target])
                        #subtree.show(line_type="ascii-em")
            dtree_id+=1

        return matched

    def checkmatch(self, node, target_node, subtree, dtree, tree, word_target=""):

        #if(node.data[1]==True):
            #get the word assuming that the target is always POS tag node
            #print(target_node.fpointer[0])
            #print(target_node.identifier)
            #word_target = dtree.get_node(target_node.fpointer[0]).tag




        #node matches check for children
        #assume the sub tree has maximum 1 child and take the first one
        children = tree.children(node.identifier)


        if(children is None or len(children)==0):
            #leaf has been reached
            return 1, subtree, word_target
        else:
            #take the first child

            child = children[0]


            #if(child.data[0]==True):
                #print(child.tag)
            target_children = dtree.children(target_node.identifier)
            if(target_children is None or len(target_children)==0):
                #this means that the dependency tree reached an end but the rule didn't
                #so check if the next node was optional
                if(child.data[0]==True):
                    return 1, subtree, word_target

            found_match = [0]
            found=0
            for target_child in target_children:
                #print(target_child.tag)
                #target_child = tree.get_node(target_child)
                #print(target_child)

                if(child.tag=='.' or target_child.tag == child.tag):
                    found=1
                    subtreecopy = Tree(subtree, deep=True)
                    #check if it is the target


                    #if optional node then check whether the node is within 1 distance away w.r.t sentence
                    if(child.data[0]==True):
                        relid = target_child.identifier.split('_')
                        #print(relid, target_child.tag)

                        if(len(relid)!=3 or abs(int(relid[0])-int(relid[1]))!=1):
                            return 1, subtree, word_target

                    subtreecopy.create_node(target_child.tag, target_child.identifier, parent=target_node.identifier, data=[list(target_child.data),list(child.data)])
                    matchsuccess, subtree, word_target = self.checkmatch(child, target_child, subtreecopy, dtree, tree, word_target)
                    if(matchsuccess==0):
                        #print('yes something got removed')
                        subtree.remove_node(target_child.identifier)
                    found_match.append(matchsuccess)


            if(np.sum(found_match)!=0):
                return 1, subtree, word_target
            elif(child.data[0]==True and found!=1):
                #print(child.tag)

                return 1, subtree, word_target
            else:
                return 0, subtree, word_target


    def verifymatch(self, tree1, tree2):
        #fetch the root node
        tree1_node = tree1.get_node(0)
        tree2_node = tree2.get_node(0)

        #iterate through both trees, assume same structure
        while(tree1_node.is_leaf() is False):
            #move pointer to next child, assume one child only
            tree1_node = tree1.get_node(tree1_node.fpointer[0])
            tree2_node = tree2.get_node(tree2_node.fpointer[0])

            #check if node is set for hard match if yes then compare the node tags
            if(tree1_node.data[1][2] and tree1_node.tag!=tree2_node.tag):
                return False
        return True

    def extractSentence(self, dtree1, dtree2, tree1_node, tree2_node, bool_goldsent, treecopies, treecopies_gold, gold, sourceid, judgeid, ruleid, tree1, tree2):
        #switch words keeping the identifier the same
        dtree1.get_node(tree1_node.identifier).tag = tree2_node.tag
        dtree2.get_node(tree2_node.identifier).tag = tree1_node.tag

        target_ident1 = tree1_node.identifier
        target_ident2 = tree2_node.identifier

        #print(target_ident1)
        #print(target_ident2)

        #
        dwshow=0
        if(tree1_node.is_leaf() is False):
            dwshow=1
            #print("yesss")
            parentwordid = tree1_node.identifier
            new_parentwordid = tree2_node.identifier


            tree1 = tree1.subtree(tree1_node.fpointer[0])
            node = tree1.get_node(tree1_node.fpointer[0])


            #change subtree
            fcounter  = 1
            bcounter = 1
            #create a partition of 1000 words, which means maximum 1000 words can be added before or after the parent
            partition = 1.0/1000.0
            while(node is not None):
                if(node.data[0][0]=='word'):
                    wordid = node.identifier

                    if(parentwordid<wordid):

                        node.data[0][1] =  str(float(new_parentwordid) + float(fcounter)*partition)
                        fcounter +=1
                    else:

                        node.data[0][1] =  str(float(new_parentwordid) - float(bcounter)*partition)
                        bcounter += 1


                if(node.is_leaf() is True):
                    break
                node = tree1.get_node(node.fpointer[0])
            #remove subtree
            #dtree2.show(line_type="ascii-em")
            dtree1.remove_node(tree1_node.fpointer[0])
            #tree1.show(line_type="ascii-em")
            #print(node.tag)
            #print("is the data okay ", node.data)
            #print("are ids same ", match[0][1], match[1][1])
            #add subtree
            dtree2.paste(tree2_node.identifier, tree1)
            #dtree1.show(line_type="ascii-em")

        if(tree2_node.is_leaf() is False):
            #for comparison
            parentwordid = tree2_node.identifier
            #for replacement
            new_parentwordid = tree1_node.identifier
            tree2 = tree2.subtree(tree2_node.fpointer[0])
            node = tree2.get_node(tree2_node.fpointer[0])

            #change subtree
            fcounter  = 1
            bcounter = 1
            #create a partition of 1000 words, which means maximum 1000 words can be added before or after the parent
            partition = 1.0/1000.0
            while(node is not None):
                if(node.data[0][0]=='word'):
                    wordid = node.identifier

                    if(parentwordid<wordid):
                        node.data[0][1] =  str(float(new_parentwordid) + float(fcounter)*partition)
                        fcounter +=1
                    else:
                        node.data[0][1] =  str(float(new_parentwordid) - float(bcounter)*partition)
                        bcounter += 1

                if(node.is_leaf() is True):
                    break
                node = tree2.get_node(node.fpointer[0])

            #remove subtree
            dtree2.remove_node(tree2_node.fpointer[0])
            #add subtree
            dtree1.paste(tree1_node.identifier, tree2)

        window_size = 2
        sents = []
        #this is to handle context and conjugation exceptions
        furthercheck = True
        target_window_sent = ''
        for treecopy in treecopies:
            sent = []
            target_posid1 = -1
            target_posid2 = -1
            for nodeid in treecopy.expand_tree():
                node = treecopy.get_node(nodeid)

                #print(node.tag)
                if(node.data[0] == 'word'):
                    if(nodeid==target_ident1):
                        target_posid1 = float(node.data[1])
                        sent.append([float(node.data[1]),node.tag, 1])
                        #print(node.tag)
                    elif(nodeid==target_ident2):
                        target_posid2 = float(node.data[1])
                        sent.append([float(node.data[1]),node.tag, 2])
                    else:
                        sent.append([float(node.data[1]),node.tag, 0])
                        #print(node.tag)

                elif(node.data[0][0]=='word'):

                    if(nodeid==target_ident1):
                        target_posid1 = float(node.data[0][1])
                        sent.append([float(node.data[0][1]),node.tag, 1])
                        #print(node.tag)
                    elif(nodeid==target_ident2):
                        target_posid2 = float(node.data[0][1])
                        sent.append([float(node.data[0][1]),node.tag, 2])
                    else:
                        sent.append([float(node.data[0][1]),node.tag, 0])


            sent = pd.DataFrame(sent)
            sent.columns = ['id', 'word', 'target']
            sent = sent.sort_values('id',axis=0)
            sent = sent.reset_index(drop=True)
            #print(sent)

            #if both the targets are present in the same sentence
            if(len(sent[sent['target']!=0])==2):
                #then check if the words are linked only through conjugations such as ,, and, or,
                target_posid1 = list(sent[sent['target']==1].index)[0]
                target_posid2 = list(sent[sent['target']==2].index)[0]
                word_list = list(sent['word'])
                sub_sent = ''
                sub_sent_len = 0

                if(target_posid1>target_posid2):
                    sub_sent_len = target_posid1 - target_posid2 + 1
                    sub_sent = " ".join(word_list[target_posid2 : target_posid1+1])
                else:
                    sub_sent_len = target_posid2 - target_posid1 + 1
                    sub_sent = " ".join(word_list[target_posid1 : target_posid2+1])

                if((len(sub_sent.split(' , '))*2-1 == sub_sent_len) or (len(sub_sent.split(' and '))*2-1 == sub_sent_len) or (len(sub_sent.split(' or '))*2-1 == sub_sent_len)):
                    #print(':O')
                    furthercheck = False

            if(bool_goldsent and len(sent[sent['target']!=0])==1):
                word_list = list(sent['word'])
                target_posid1 = list(sent[sent['target']==1].index)[0]
                lower = target_posid1 - window_size
                upper = target_posid1 + window_size + 1
                #make lower zero if it becomes less than zero
                lower = 0 if lower<0 else lower
                upper = len(word_list) if upper>len(word_list) else upper

                #target_window_sent = word_list[lower:upper]
                target_window_sent = word_list[lower:target_posid1]
                target_window_sent.extend(word_list[target_posid1+1:upper])


            #print(sent)
            #if(dwshow):
                #print(sent)
            sent = " ".join(list(sent['word']))
            sents.append(sent)

        gold_window_sent = ''
        #get context of word from gold sent, only when rules use gold sents
        if(bool_goldsent):
            for treecopy in treecopies_gold:
                sent = []
                target_posid1 = -1
                target_posid2 = -1
                for nodeid in treecopy.expand_tree():
                    node = treecopy.get_node(nodeid)

                    #print(node.tag)
                    if(node.data[0] == 'word'):
                        if(nodeid==target_ident1):
                            target_posid1 = float(node.data[1])
                            sent.append([float(node.data[1]),node.tag, 1])
                            #print(node.tag)
                        elif(nodeid==target_ident2):
                            target_posid2 = float(node.data[1])
                            sent.append([float(node.data[1]),node.tag, 2])
                        else:
                            sent.append([float(node.data[1]),node.tag, 0])
                            #print(node.tag)

                    elif(node.data[0][0]=='word'):

                        if(nodeid==target_ident1):
                            target_posid1 = float(node.data[0][1])
                            sent.append([float(node.data[0][1]),node.tag, 1])
                            #print(node.tag)
                        elif(nodeid==target_ident2):
                            target_posid2 = float(node.data[0][1])
                            sent.append([float(node.data[0][1]),node.tag, 2])
                        else:
                            sent.append([float(node.data[0][1]),node.tag, 0])


                sent = pd.DataFrame(sent)
                sent.columns = ['id', 'word', 'target']
                sent = sent.sort_values('id',axis=0)
                sent = sent.reset_index(drop=True)
                #print(sent)



                if(len(sent[sent['target']!=0])==1):
                    word_list = list(sent['word'])
                    target_posid1 = list(sent[sent['target']==2].index)[0]
                    lower = target_posid1 - window_size
                    upper = target_posid1 + window_size + 1
                    #make lower zero if it becomes less than zero
                    lower = 0 if lower<0 else lower
                    upper = len(word_list) if upper>len(word_list) else upper

                    gold_window_sent = word_list[lower:target_posid1]
                    gold_window_sent.extend(word_list[target_posid1+1:upper])

                    common_words = list(set(target_window_sent).intersection(gold_window_sent))
                    #print(gold_window_sent)
                    #print(target_window_sent)
                    #print(common_words)

                    gold_window_len = len(gold_window_sent)
                    target_window_len = len(target_window_sent)
                    common_words_len = len(common_words)
                    min_len = min(gold_window_len, target_window_len)
                    if(min_len!=0):
                        overlap_ratio = float(common_words_len)/float(min_len)
                        if(overlap_ratio>=0.65):
                            furthercheck = False
                    #print(overlap_ratio)

                sent = " ".join(list(sent['word']))


        if(furthercheck is False):
            return 0

        #final sentence
        sent = " ".join(sents)
        #print(sent)

        if(gold is not None):
            #print(sent)
            score = getRougeScore(gold, sent)

            scores = score[0]
            #get whether each type of rouge metric was acceptable or not
            #acceptability, smc_score_list, smic_score_list = self.rougeAcceptability(score, smcscore, sourceid)

            #smicLMScore = lm.getLMScore(sent)
            #smicLMScore = 1
            #relscore = (smicLMScore - smcLMScore) / smcLMScore
            datarow = {'sourceid':sourceid, 'judgeid':judgeid, 'smic': sent, 'ruleid': ruleid, 'rouge1_f':scores['rouge-1']['f'], 'rouge1_p':scores['rouge-1']['p'], 'rouge1_r':scores['rouge-1']['r'], 'rouge2_f':scores['rouge-2']['f'], 'rouge2_p':scores['rouge-2']['p'], 'rouge2_r':scores['rouge-2']['r'], 'rougel_f':scores['rouge-l']['f'], 'rougel_p':scores['rouge-l']['p'], 'rougel_r':scores['rouge-l']['r'] }
        else:
            datarow = {'sourceid':sourceid, 'judgeid':judgeid, 'smic': sent, 'ruleid': ruleid}

        return datarow

    def generateSentence(self, dtrees, dtrees_gold, matches, matches_gold, gold, hardmatch, smcscore, sourceid, smcLMScore, ruleid, smc, judgeid, rulefrom, cov_check):

        bool_goldsent = False
        #check whether substitution needs to be done within smc standard or not
        if(len(dtrees_gold)==0):
            #create combinations of matches to switch subtrees within smc
            matched = itertools.combinations(matches, 2)
        else:
            #create combinations of matches to switch subtrees between gold and smc
            matched = itertools.product(matches, matches_gold)
            bool_goldsent = True

        sentences = []

        coverage = []

        match_coverage = []
        #print('start####################')
        #loop through combinations of all the matches
        for match in (list(matched)):

            #the first index is for accessing the combination returned by combinations and the second is for accessing the tree or dtree id
            tree1 = match[0][0]
            tree2 = match[1][0]

            #if(bool_goldsent):
                #verify if the same target word was chosen

                #if(match[1][2]==""):
                #    raise ValueError('Could not find target, is this possible?')
                #elif(match[1][2].strip().lower()==match[0][2].strip().lower()):
                #    continue

            if(hardmatch):
                if(self.verifymatch(tree1, tree2) is False):
                    continue
                #print("hard match accepted")
                #tree1.show(line_type="ascii-em")
                #tree2.show(line_type="ascii-em")



            #maintain copy of each dependency tree
            treecopies = []
            for dtree in dtrees:
                treecopies.append(Tree(dtree, deep=True))
            treecopies_gold = []
            if(bool_goldsent):
                for dtree in dtrees_gold:
                    treecopies_gold.append(Tree(dtree, deep=True))

            #get corresponding dtree in which the matched rules are present
            dtree1 = treecopies[match[0][1]]
            dtree2 = ""
            if(bool_goldsent):
                dtree2 = treecopies_gold[match[1][1]]
            else:
                dtree2 = treecopies[match[1][1]]


            match1_id = tree1.get_node(0).fpointer[0]
            match2_id = tree2.get_node(0).fpointer[0]
            parent1_id = dtree1.get_node(match1_id).bpointer
            parent2_id = dtree2.get_node(match2_id).bpointer

            #get tree1 word node
            tree1_node = tree1.get_node(match1_id)
            while(tree1_node.data[1][1] is False):
                tree1_node = tree1.get_node(tree1_node.fpointer[0])

            #get tree2 word node
            tree2_node = tree2.get_node(match2_id)
            while(tree2_node.data[1][1] is False):
                tree2_node = tree2.get_node(tree2_node.fpointer[0])



            tree1_node_pos = tree1_node.tag
            tree2_node_pos = tree2_node.tag

            #assuming the target node is always the pos node
            #get the child which will be the word node
            tree1_node = tree1.get_node(tree1_node.fpointer[0])
            tree2_node = tree2.get_node(tree2_node.fpointer[0])

            #do not switch same words
            if(tree2_node.tag.strip().lower() == tree1_node.tag.strip().lower()):
                continue

            #print(tree1_node.tag, tree1_node.data)
            #print(tree2_node.tag, tree2_node.data)


            datarow = 0

            #if the rule is for verbs
            if(ruleid==3 or ruleid==7 or ruleid==11):
                #if pos of the two targets differ then we need to convert it using simplenlg
                if(tree1_node_pos!=tree2_node_pos):
                    #print("POS dont match")
                    #print(tree1_node.tag, tree2_node_pos)
                    tree1_word_list = changePOS(tree1_node.tag, tree1_node_pos)
                    tree2_word_list = changePOS(tree2_node.tag, tree2_node_pos)
                    #print(tree1_node.tag, tree1_word_list, tree2_node_pos, len(tree1_word_list))
                    #print(tree2_node.tag, tree2_word_list, tree1_node_pos, len(tree2_word_list))
                    #print('###############')


                    if(len(tree1_word_list)==0 or len(tree2_word_list)==0):
                        if(len(tree1_word_list)!=0):
                            for tree1_word in tree1_word_list:
                                tree1_node.tag = tree1_word
                                if(rulefrom == 'SMC'):
                                    #check of the combination of word was already covered
                                    tuple_tosearch = (tree1_node.tag.lower(), tree2_node.tag.lower())

                                    if(tuple_tosearch in match_coverage or tuple(reversed(tuple_tosearch)) in match_coverage):
                                        #will reach only if rulefrom SMC
                                        continue
                                datarow = self.extractSentence(dtree1, dtree2, tree1_node, tree2_node, bool_goldsent, treecopies, treecopies_gold, gold, sourceid, judgeid, ruleid, tree1, tree2)
                                if(datarow==0):
                                    continue
                                match_coverage.append((tree1_node.tag.lower(), tree2_node.tag.lower()))

                                sentences.append(datarow)
                        else:

                            for tree2_word in tree2_word_list:
                                tree2_node.tag = tree2_word
                                tuple_tosearch = (tree1_node.tag.lower(), tree2_node.tag.lower())
                                #to track target nodes in GOLD rules so that they dont repeat in SOURCE rules
                                if(rulefrom == 'GOLD'):
                                    coverage.append(tree2_node.tag.lower())
                                elif(rulefrom == 'SOURCE'):
                                    if(tree2_node.tag.lower() in cov_check):
                                        continue
                                #check of the combination of word was already covered
                                elif(tuple_tosearch in match_coverage or tuple(reversed(tuple_tosearch)) in match_coverage):
                                    #will reach only if rulefrom SMC
                                    continue

                                datarow = self.extractSentence(dtree1, dtree2, tree1_node, tree2_node, bool_goldsent, treecopies, treecopies_gold, gold, sourceid, judgeid, ruleid, tree1, tree2)
                                if(datarow==0):
                                    continue
                                match_coverage.append((tree1_node.tag.lower(), tree2_node.tag.lower()))
                                sentences.append(datarow)

                    else:
                        list_combinations = itertools.product(tree1_word_list, tree2_word_list)
                        for list_combination in list(list_combinations):
                            tree1_node.tag = list_combination[0]
                            tree2_node.tag = list_combination[1]
                            tuple_tosearch = (tree1_node.tag.lower(), tree2_node.tag.lower())
                            #check of the combination of word was already covered

                            #to track target nodes in GOLD rules so that they dont repeat in SOURCE rules
                            if(rulefrom == 'GOLD'):
                                coverage.append(tree2_node.tag.lower())
                            elif(rulefrom == 'SOURCE'):
                                if(tree2_node.tag.lower() in cov_check):
                                    continue
                            #check of the combination of word was already covered
                            elif(tuple_tosearch in match_coverage or tuple(reversed(tuple_tosearch)) in match_coverage):
                                #will reach only if rulefrom SMC
                                continue
                            datarow = self.extractSentence(dtree1, dtree2, tree1_node, tree2_node, bool_goldsent, treecopies, treecopies_gold, gold, sourceid, judgeid, ruleid, tree1, tree2)
                            if(datarow==0):
                                continue
                            match_coverage.append((tree1_node.tag.lower(), tree2_node.tag.lower()))
                            sentences.append(datarow)
            elif((ruleid==1 or ruleid==5 or ruleid==9)):
                if(((tree1_node_pos=='NNS' and tree2_node_pos=='NN') or (tree1_node_pos=='NNS' and tree2_node_pos=='NN'))):
                    #print('Noun with different number')
                    #print(tree2_node_pos, tree2_node.tag)
                    #print(tree1_node_pos, tree1_node.tag)
                    #print(tree1_node.tag, tree2_node_pos)
                    tree1_node_new = changePlurality(tree1_node.tag, tree1_node_pos)
                    tree2_node_new = changePlurality(tree2_node.tag, tree2_node_pos)
                    if(tree1_node_new!=''):
                        tree1_node.tag = tree1_node_new
                    if(tree2_node_new!=''):
                        tree2_node.tag = tree2_node_new
                    #print(tree2_node_pos, tree2_node.tag)
                    #print(tree1_node_pos, tree1_node.tag)
                    #print('############################')

                tuple_tosearch = (tree1_node.tag.lower(), tree2_node.tag.lower())
                #to track target nodes in GOLD rules so that they dont repeat in SOURCE rules
                if(rulefrom == 'GOLD'):
                    coverage.append(tree2_node.tag.lower())
                elif(rulefrom == 'SOURCE'):
                    if(tree2_node.tag.lower() in cov_check):
                        continue
                #check of the combination of word was already covered
                elif(tuple_tosearch in match_coverage or tuple(reversed(tuple_tosearch)) in match_coverage):
                    #will reach only if rulefrom SMC
                    continue
                datarow = self.extractSentence(dtree1, dtree2, tree1_node, tree2_node, bool_goldsent, treecopies, treecopies_gold, gold, sourceid, judgeid, ruleid, tree1, tree2)
                if(datarow==0):
                    continue

                match_coverage.append((tree1_node.tag.lower(), tree2_node.tag.lower()))
                sentences.append(datarow)

            else:
                #to track target nodes in GOLD rules so that they dont repeat in SOURCE rules
                if(rulefrom == 'GOLD'):
                    coverage.append(tree2_node.tag.lower())
                elif(rulefrom == 'SOURCE'):
                    if(tree2_node.tag.lower() in cov_check):
                        continue

                datarow = self.extractSentence(dtree1, dtree2, tree1_node, tree2_node, bool_goldsent, treecopies, treecopies_gold, gold, sourceid, judgeid, ruleid, tree1, tree2)

                if(datarow==0):
                    continue

                sentences.append(datarow)

            #print(sent)
            #print(smc)
            #print(gold)
            #print('\n')

            #print(" ".join(sent))
            #dtree.paste(parent1_id, tree2.subtree(match2_id))
            #dtree.paste(parent2_id, tree1.subtree(match1_id))
            #tree1.show(line_type="ascii-em")
            #tree2.show(line_type="ascii-em")
            #treecopy.show(line_type="ascii-em")
        #print('end####################')
        return sentences, coverage

    def rougeAcceptability(self, smicscores, smcscores, sourceid):
        reduction = smcscores['rouge-1']['f'] - smicscores['rouge-1']['f']
        result = []
        smc_score_list = []
        smic_score_list = []
        for score_cat in smicscores:
            score_cat_ident = score_cat.split('-')[1]
            for score_type in smicscores[score_cat]:

                smc_score_list.append(smcscores[score_cat][score_type])
                smic_score_list.append(smicscores[score_cat][score_type])

                if(self.acceptability_type == 'difference'):
                    reduction = smcscores[score_cat][score_type] - smicscores[score_cat][score_type]
                    if(reduction <= float(self.avgchange['rouge'+ score_cat_ident +'_'+ score_type])):
                        result.append(1)
                    else:
                        result.append(0)
                elif(self.acceptability_type == 'absolute' or self.acceptability_type == 'absolute_global'):
                    outij = smicscores[score_cat][score_type]

                    if(self.acceptability_type == 'absolute_global'):
                        #global mean
                        ui = self.data_totalmean['rouge'+ score_cat_ident +'_'+ score_type]
                    else:
                        #individual mean
                        ui = self.data_mean.loc[str(sourceid)]['rouge'+ score_cat_ident +'_'+ score_type]
                    sigma = self.data_std['rouge'+ score_cat_ident +'_'+ score_type]
                    zvalue = abs((outij - ui) / sigma)
                    pvalue = scipy.stats.norm.cdf(zvalue)
                    #print(pvalue)
                    if(pvalue>= self.confidence):
                        #print(pvalue)
                        result.append(1)
                    else:
                        result.append(0)
                else:
                    #for fitting we will have to select sentences at the extend
                    result.append(1)

        return result, smc_score_list, smic_score_list

    def getSentence(self, tree, matched, leaves, gold):
        bestscore=0
        bestsent=""
        matched = itertools.combinations(matched, 2)
        for match in (list(matched)):
            #deep copy the original tree
            treecopy = Tree(tree, deep=True)

            #print(treecopy.get_node(treecopy.get_node(match[0]).fpointer[0]).tag)
            #print(treecopy.get_node(treecopy.get_node(match[1]).fpointer[0]).tag)
            #print(treecopy.parent(match[0]).identifier)
            #replace nodes
            parent1 = treecopy.parent(match[0]).identifier
            parent2 = treecopy.parent(match[1]).identifier
            child1 = treecopy.children(match[0])[0].identifier
            child2 = treecopy.children(match[1])[0].identifier

            treecopy.move_node(match[0],parent2)
            treecopy.move_node(match[1],parent1)
            sent = self.linearize(leaves, child1, child2, treecopy)
            score = getRougeScore(gold, sent)[0]['rouge-l']['f']

            if(score>bestscore):
                bestscore = score
                bestsent = sent


        return bestsent, bestscore

    def linearize(self, leaves, child1, child2, tree):
        sent=[]

        for leafid in leaves:
            if(leafid==child1 or leafid==child2):
                if(leafid==child1):
                    #append child2
                    sent.append(tree.get_node(child2).tag)
                else:
                    #append child1
                    sent.append(tree.get_node(child1).tag)
            else:
                #append other words
                sent.append(tree.get_node(leafid).tag)
        sent = " ".join(sent)
        return sent
