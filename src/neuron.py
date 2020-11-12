from random import random, randint
from synapse import Synapse


class Neuron:

    def ReLU(self):
        def compute(neuron):
            i = 0
            neuron.output = 0
            for syn in neuron.synapses:
                if syn.input == 0 or len(syn.next_neurons) != 0:
                    continue
                i += 1
                if syn.input < 0:
                    syn.input = 0
                neuron.output += syn.input * syn.weight
            neuron.output /= i
            for next_neuron in syn.next_neurons:
                for syn in next_neuron.synapses:
                    if len(syn.next_neuron) != 0:
                        syn.input = neuron.output
                compute(next_neuron)
        compute(self)

    def __init__(self, id, synapses):
        self.id = id
        self.synapses = synapses
        self.ReLU()

    def iterate_children_recursive(self, callback):
        def iterate(neuron):
            for syn in neuron.synapses:
                for next_neuron in syn.next_neurons:
                    callback(neuron, next_neuron, syn)
                    iterate(next_neuron)
        iterate(self)

    def find_child(self, callback):
        def iterate(neuron):
            for syn in neuron.synapses:
                for next_neuron in syn.next_neurons:
                    if callback(neuron, next_neuron, syn):
                        return neuron
                    iterate(next_neuron)
        iterate(self)

    def mutate(self):
        def propagate(neuron, next, syn):
            if random() > 0.5:
                synapse_index = randint(0, len(next.synapses) - 1)
                next.synapses[synapse_index].weight = random()
        self.iterate_children_recursive(propagate)
        self.ReLU()

    def mutate_struct(self, genetic):
        neurons = [self]

        def pick_neuron(neuron, next_neuron, syn):
            if not any(neuron.id == next_neuron.id for neuron in neurons):
                neurons.append(next_neuron)

        self.iterate_children_recursive(pick_neuron)

        sorted(neurons, key=lambda n: n.id)
        last_neuron_id = neurons[len(neurons) - 1].id

        gen_index = randint(1, len(neurons) - 2) if len(neurons) > 1 else 0

        if gen_index == 1 and len(neurons) <= 1:
            gen_index = 0

        def gen_neuron():
            neuron = Neuron(gen_index, genetic.map_data(genetic.gen_data))
            synapse = Synapse(neurons[gen_index].output, random(), [])
            if len(neurons) > gen_index + 1:
                synapse.next_neurons.append(neurons[gen_index + 1])
            neuron.synapses.append(synapse)
            neurons.insert(gen_index, neuron)
            neurons[gen_index - 1].synapses.append(
                Synapse(random(), random(), [neuron]))
            for i in range(gen_index, len(neurons)):
                neurons[i].id += 1

        def gen_synapse():
            ascending_exploration = random() > 0.5

            def explore(ascending):
                for i in range(gen_index, len(neurons) if ascending else 0):
                    for syn in neurons[i].synapses:
                        for next_neuron in syn.next_neurons:
                            new_neuron_bond = [not any(o_neuron.id == next_neuron.id or o_neuron.id <= i
                                                       for o_neuron in neurons) for neuron in neurons]
                            if new_neuron_bond != False and len(new_neuron_bond) > 0:
                                next_neuron.synapses.append(
                                    Synapse(next_neuron.output, random(), new_neuron_bond))
                                return True
                return False
            success = explore(ascending_exploration)
            if not success:
                explore(not ascending_exploration)
            return success

        should_gen_node = random() > 0.5
        if should_gen_node:
            gen_neuron()
        elif not gen_synapse():
            gen_neuron()
        self.ReLU()
