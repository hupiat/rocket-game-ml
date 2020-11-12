from data import Data
from random import random
from synapse import Synapse
from neuron import Neuron


class Genetic:

    count_individuals = 5

    mutation_prob = 0.2
    mutation_struct_prob = 0.3
    crossover_unique_gene_transfer_prob = 0.2

    weights = [1, 0.5, 1, 1]

    last_input_datas = None
    last_best = None

    def adjust_weights(self, final_datas):
        best = (0, 0, 0)
        for i in range(len(final_datas)):
            if final_datas[i].score > best[0].score:
                best[0] = final_datas[i]
                best[1] = self.last_input_datas[i]
                best[2] = i

        if self.last_best is None or self.last_best.score < best[0].score:
            self.last_best = best[0]
        else:
            return

        total = 0
        for attr in best[0].get_attributes():
            diff = best[0][attr] - best[1][attr]
            total += diff * diff
        total /= len(final_datas)

        for i in range(len(final_datas)):
            wall_to_bottom = final_datas[i].wall_direction == 0
            def update_weight(weight):
                if wall_to_bottom and i != 1:
                    weight -= total
                else:
                    weight += total
            for j in range(len(self.weights)):
              update_weight(self.weights[j])

    def gen_data(self):
        return Data(0, random(), random(), random(),
                    1 if random() > 0.5 else 0, random())

    def map_data(self, data):
        return [
            Synapse('rocket_top', data.rocket_top, self.weights[0], []),
            Synapse('wall_direction', data.wall_direction,
                    self.weights[1], []),
            Synapse('wall_left', data.wall_left, self.weights[2], []),
            Synapse('wall_length', data.wall_length,
                    self.weights[3], []),
        ]

    def generation_gen(self):
        self.root_neurons = []
        for _ in range(0, self.count_individuals):
            self.root_neurons.append(Neuron(
                0, self.map_data(self.gen_data())))

    def generation_train(self, datas):
        self.last_input_datas = datas
        for i in range(0, len(datas)):
            data = self.map_data(datas[i])
            self.root_neurons[i].synapses = list(data)
            self.root_neurons[i].ReLU()
            output = self.root_neurons[i].output
            yield output

    def generation_crossover(self, datas):
        new_gen = []

        for i in range(0, self.count_individuals):
            should_mutate = random() < self.mutation_prob
            if should_mutate:
                self.root_neurons[i].mutate()
            should_struct_mutate = random() < self.mutation_struct_prob
            if should_struct_mutate:
                self.root_neurons[i].mutate_struct()

        sorted(datas, key=lambda d: d.score, reverse=True)
        for i in range(0, len(datas)):
            for j in range(0, len(datas)):
                if i == j:
                    continue
                root_synapses = self.root_neurons[i].synapses
                for syn in root_synapses:
                    syn.weight = (
                        syn.weight + self.root_neurons[j].synapses[root_synapses.index(syn)].weight) / 2
                root_neuron = Neuron(0, root_synapses)
                f_genes, s_genes = []
                self.root_neurons[i].iterate_children_recursive(
                    lambda neuron, next_neuron, syn: f_genes.append((neuron, next_neuron, syn)))
                self.root_neurons[j].iterate_children_recursive(
                    lambda neuron, next_neuron, syn: s_genes.append((neuron, next_neuron, syn)))

                for gene in f_genes:
                    for o_gene in s_genes:
                        if gene[0].id == o_gene[0].id and gene[1].id == o_gene[1].id or random() < self.crossover_unique_gene_transfer_prob:
                            match_neuron = root_neuron.find_child(
                                lambda n: n.id == o_gene[0].id) if o_gene[0].id != 0 else False
                            ancestor_genes = root_neuron
                            new_syn = Synapse(o_gene[2].label
                                              (gene[2].input + o_gene[2].input) / 2, (gene[2].weight + o_gene[2].weight) / 2, [])
                            if match_neuron is None:
                                match_neuron = Neuron(o_gene[0].id, [new_syn])
                                match_ancestor_genes = filter(
                                    lambda g: g[1].id == match_neuron.id, s_genes)
                                if len(match_ancestor_genes) > 0:
                                    ancestor_genes = match_ancestor_genes[0]

                            new_neuron = Neuron(o_gene[1].id + len(map(lambda n:
                                                                       map(lambda s: s.next_neurons, n.synapses), match_neuron)) + 1, [new_syn])

                            if ancestor_genes is root_neuron:
                                root_neuron.synapses.append(
                                    Synapse(new_syn.label, new_syn.input, new_syn.weight, [new_neuron]))
                            else:
                                ancestor_genes[2].next_neurons.append(
                                    new_neuron)
                    if len(new_gen) == self.count_individuals:
                        break

        self.root_neurons = new_gen
        self.adjust_weights(datas)
