import numpy as np
from random import shuffle
from model import Model


class Monik:
    def __init__(self):
        self.NUM_EXAMPLES = 10000

    def main(self):
        # Input created
        train_input = ['{0:020b}'.format(i) for i in range(2**20)]
        shuffle(train_input)
        train_input = [map(int,i) for i in train_input]
        ti = []
        for i in train_input:
            temp_list = []
            for j in i:
                temp_list.append([j])
            ti.append(np.array(temp_list))
        train_input = ti

        #One hot vector input representation
        train_output = []
        for i in train_input:
            count = 0
            for j in i:
                if j[0] == 1:
                    count += 1
            temp_list = ([0] * 21)
            temp_list[count] = 1
            train_output.append(temp_list)


        # Splitting data to train and test
        test_input = train_input[self.NUM_EXAMPLES:]
        test_output = train_output[self.NUM_EXAMPLES:]

        train_input = train_input[:self.NUM_EXAMPLES]
        train_output = train_output[:self.NUM_EXAMPLES]

        print("aa")
        model = Model()
        model.create_model(train_input, train_output, test_input, test_output)


if __name__ == "__main__":
    monik = Monik()
    monik.main()
