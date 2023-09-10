import numpy
import scipy.special
import random

MUTATION_WEIGHT_MODIFY_CHANCE = 0.2
MUTATION_ARRAY_MIX_PERC = 0.5

class Nnet:
    def __init__(self, input, hidden, output):
        self.num_input = input
        self.num_hidden = hidden
        self.num_output = output
        self.weight_input_hidden = numpy.random.uniform(-0.5, 0.5, size=(self.num_hidden, self.num_input))
        self.weight_hidden_output = numpy.random.uniform(-0.5, 0.5, size=(self.num_output, self.num_hidden))
        self.activation_function = lambda x: scipy.special.expit(x)
    def get_outputs(self, inputs_list):
        input = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.weight_input_hidden,input)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_input = numpy.dot(self.weight_hidden_output,hidden_outputs)
        final_output = self.activation_function(final_input)
        return final_output
    def get_max(self, inputs_list):
        outputs = self.get_outputs(inputs_list)
        max = numpy.max(outputs)
        return max

    def modify_weights(self):
        Nnet.modify_array(self.weight_input_hidden)
        Nnet.modify_array(self.weight_hidden_output) #independent of instance

    def create_mixed_weights(self, net1, net2):
        self.weight_input_hidden = Nnet.get_mix_from_arrays(net1.weight_input_hidden, net2.weight_input_hidden)
        self.weight_hidden_output = Nnet.get_mix_from_arrays(net1.weight_hidden_output, net2.weight_hidden_output)

    def save_weights(self):
        with open("saved_weights.npy", "wb") as f:
            numpy.save(f, self.weight_input_hidden)
            numpy.save(f, self.weight_hidden_output)

    def load_weights(self):
        with open("saved_weights.npy", "rb") as f:
            self.weight_input_hidden = numpy.load(f)
            self.weight_hidden_output = numpy.load(f)

    @staticmethod
    def modify_array(a):
        for x in numpy.nditer(a,op_flags=["readwrite"]):
            if random.random() < MUTATION_WEIGHT_MODIFY_CHANCE:
                x[...] = numpy.random.random_sample() - 0.5

    @staticmethod
    def get_mix_from_arrays(ar1, ar2):
        total_entries = ar1.size
        num_rows = ar1.shape[0]
        num_cols = ar1.shape[1]

        num_to_take = total_entries - (int(total_entries * MUTATION_ARRAY_MIX_PERC))
        idx = numpy.random.choice(numpy.arange(total_entries), num_to_take, replace=False)

        res = numpy.random.rand(num_rows, num_cols)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row*num_cols+col #conversion to 1d-list-like index
                if index in idx:
                    res[row][col] = ar1[row][col]
                else:
                    res[row][col] = ar2[row][col]

        return res

def tests():
    ar1 = numpy.random.uniform(-0.5, 0.5, size=(3,4))
    ar2 = numpy.random.uniform(-0.5, 0.5, size=(3,4))
    print("ar1.size", ar1.size, sep="\n")
    print("ar1", ar1, sep="\n")

    Nnet.modify_array(ar1)
    print("ar1", ar1, sep="\n")

    print("")

    print("ar1", ar1, sep="\n")
    print("ar2", ar2, sep="\n")

    mixed = Nnet.get_mix_from_arrays(ar1, ar2)
    print("mixed", mixed, sep="\n")

if __name__ == "__main__":
    tests()