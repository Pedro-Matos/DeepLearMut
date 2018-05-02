import glob
from collections import defaultdict


class Eval:
    def __init__(self):
        self.keys = []
        self.gold = defaultdict(list)
        self.silver = defaultdict(list)

    def read(self):
        path_gold = '../corpus_char/tmVarCorpus/treated/gold_results/'
        docs = glob.glob(path_gold + "*.a1")

        path_silver = '../model/silver_minibatch_20epoch/'
        docs_silver = glob.glob(path_silver + "*.a1")

        if len(docs) != len(docs_silver):
            print("The folders do not contain the same size.")
            exit()

        for i in range(len(docs)):
            file = docs[i].split("/")[-1]
            self.keys.append(file)

        for key in self.keys:
            path = path_gold + key
            with open(path) as reading:
                sentences = reading.readlines()
                if sentences != []:
                    id = key.split(".")[0]
                    for sent in sentences:
                        sent = sent.rstrip()
                        sent = sent.split("\t")
                        t = (sent[0], sent[1], sent[2])
                        self.gold[id].append(t)

            path = path_silver + key
            with open(path) as reading:
                sentences = reading.readlines()
                if sentences != []:
                    id = key.split(".")[0]
                    for sent in sentences:
                        sent = sent.rstrip()
                        sent = sent.split("\t")
                        t = (sent[0], sent[1], sent[2])
                        self.silver[id].append(t)


        for i in range(len(self.keys)):
            self.keys[i] = self.keys[i].split(".")[0]

    def evaluate(self):
        tp_all = 0
        fp_all = 0
        fn_all = 0

        for i in range(len(self.keys)):
            id = self.keys[i]

            tp = 0
            fp = 0
            fn = 0

            correct_ents = self.gold[id]
            pred_ents = self.silver[id]
            for ent in pred_ents:
                if ent in correct_ents:
                    tp += 1
                else:
                    fp += 1
            fn = len(correct_ents) - tp

            tp_all += tp
            fp_all += fp
            fn_all += fn

        f = (2 * tp_all / (tp_all + tp_all + fp_all + fn_all))
        if tp_all == 0:
            precision = 0
        else:
            precision = tp_all / (tp_all + fp_all)

        print("TP: " + str(tp_all))
        print("FP: " + str(fp_all))
        print("FN: " + str(fn_all))
        print("F: " + str(f))
        print("Precision: " + str(precision))
        print("Recal: " + str(tp_all / (tp_all + fn_all)))



if __name__ == "__main__":
    eval = Eval()
    eval.read()
    eval.evaluate()