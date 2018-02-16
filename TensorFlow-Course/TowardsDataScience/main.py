import collections
import numpy as np

'''
Suppose we want to train a LSTM to predict the next word using a sample short story
'''

class Example:
    def __init__(self):
        self.type = "LSTM"
        self.training_file = 'belling_the_cat.txt'


    def main(self):
        training_data = self.read_data(self.training_file)
        print("Loaded training data...")

        dictionary, reverse_dictionary = self.build_dataset(training_data)
        vocab_size = len(dictionary)




    ''' 
    Technically, LSTM inputs can only understand real numbers. A way to convert symbol to number
    is to assign a unique integer to each symbol based on frequency of occurrence. For example,
    there are 112 unique symbols in the text above. The function in Listing 2 builds a dictionary with the
    following entries [ “,” : 0 ] [ “the” : 1 ], …, [ “council” : 37 ],…,[ “spoke” : 111 ].
    The reverse dictionary is also generated since it will be used in decoding the output of LSTM.
    '''
    def build_dataset(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    def read_data(self,fname):
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [content[i].split() for i in range(len(content))]
        content = np.array(content)
        content = np.reshape(content, [-1, ])
        return content





if __name__ == "__main__":
    example = Example()
    example.main()

