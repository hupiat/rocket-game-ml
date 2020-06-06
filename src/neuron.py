from random import random, randint
from synapse import Synapse


class Neuron:

    def ReLU(self):
        self.output = 0
        for syn in self.synapses:
            if len(syn.next_neurons) != 0:
                continue
            if syn.input < 0:
                syn.input = 0
            self.output += syn.input * syn.weight
        self.output /= len(self.synapses)

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
        synapse_index = randint(0, len(self.synapses) - 1)
        self.synapses[synapse_index].weight = random()
        self.iterate_children_recursive(lambda n, next, syn: n.ReLU())

    def mutate_struct(self):
        neurons = [self]

        self.iterate_children_recursive(lambda neuron, next_neuron, syn:
                                        neurons.append(next_neuron) if not any(neuron.id == next_neuron.id for neuron in neurons) else neuron)

        sorted(neurons, key=lambda n: n.id)
        last_neuron_id = neurons[len(neurons) - 1].id

        gen_index = randint(1, len(neurons) - 2) if len(neurons) > 1 else 0

        if gen_index == 1 and len(neurons) <= 1:
            gen_index = 0

        def gen_neuron():
            neuron = Neuron(gen_index, [Synapse(random(), random(), [])])
            synapse = Synapse(neurons[gen_index].output, random(), [])
            if len(neurons) > gen_index + 1:
                synapse.next_neurons.append(neurons[gen_index + 1])
            neuron.synapses.append(synapse)
            neurons.insert(gen_index, neuron)
            neurons[gen_index - 1].synapses.append(
                Synapse(neurons[gen_index - 1].output, random(), [neuron]))
            for i in range(gen_index, len(neurons)):
                neurons[i].id += 1

        def gen_synapse():
            ascending_exploration = random() > 0.5

            def explore(ascending):
                for i in range(gen_index if ascending else gen_index, len(neurons) if ascending else 0):
                    for syn in neurons[i].synapses:
                        for next_neuron in syn.next_neurons:
                            new_neuron_bond = [not any(o_neuron.id == next_neuron.id or o_neuron.id == neurons[i].id
                                                       for o_neuron in neurons) for neuron in neurons]
                            if new_neuron_bond != False and len(new_neuron_bond) > 0:
                                next_neuron.synapses.append(
                                    Synapse(next_neuron.output, random(), new_neuron_bond))
                                next_neuron.ReLU()
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
        self.iterate_children_recursive(lambda n, next, syn: n.ReLU())
