from data import Data
from random import random
from synapse import Synapse
from neuron import Neuron


class Genetic:

    count_individuals = 10
    mutation_prob = 0.8
    mutation_struct_prob = 0.5
    crossover_unique_gene_transfer_prob = 0.2

    def __init__(self):
        self.rocket_top_weight = 2
        self.wall_direction_weight = 1
        self.wall_left_weight = 1
        self.wall_length_weight = 1

    def adjust_weight(self, final_datas):
        def mse(index, label):
            total = 0
            for i in range(len(final_datas)):
                diff = final_datas[i][label] - self.last_input_datas[i][label]
                total += diff * diff
            total /= len(final_datas)
            success = self.root_neurons[index].output > 0.5
            if final_datas[i].wall_direction == 0:
                success = self.root_neurons[index].output <= 0.5
            if success:
                self[label + '_weight'] -= total
            else:
                self[label + '_weight'] += total

        max_score = (0, 0)
        for i in range(len(final_datas)):
            if final_datas[i].score > max_score[0]:
                max_score[0] = final_datas[i].score
                max_score[1] = i

        mse(max_score[1], 'rocket_top')
        mse(max_score[1], 'wall_direction')
        mse(max_score[1], 'wall_left')
        mse(max_score[1], 'wall_length')

    def map_data(self, data):
        return [
            Synapse(data.rocket_top, self.rocket_top_weight, []),
            Synapse(data.wall_direction, self.wall_direction_weight, []),
            Synapse(data.wall_left, self.wall_left_weight, []),
            Synapse(data.wall_length, self.wall_length_weight, []),
        ]

    def generation_gen(self):
        self.root_neurons = []
        for _ in range(0, self.count_individuals):
            self.root_neurons.append(Neuron(
                0, self.map_data(Data(0, random(), random(), random(), 1 if random() > 0.5 else 0, random()))))

    def generation_train(self, datas):
        self.last_input_datas = datas
        for i in range(0, len(datas)):
            data = self.map_data(datas[i])
            self.root_neurons[i].synapses = list(data)
            self.root_neurons[i].ReLU()
            yield self.root_neurons[i].output

    def generation_crossover(self, datas):
        new_gen = []

        for i in range(0, self.count_individuals):
            should_mutate = random() <= self.mutation_prob
            if should_mutate:
                self.root_neurons[i].mutate()
            should_struct_mutate = random() <= self.mutation_struct_prob
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
                        if gene[0].id == o_gene[0].id and gene[1].id == o_gene[1].id or random() <= self.crossover_unique_gene_transfer_prob:
                            match_neuron = root_neuron.find_child(
                                lambda n: n.id == o_gene[0].id) if o_gene[0].id != 0 else root_neuron
                            should_insert_at_root = False
                            if match_neuron is None:
                                match_neuron = Neuron(o_gene[0].id, [Synapse(
                                    (gene[2].input + o_gene[2].input) / 2, (gene[2].weight + o_gene[2].weight) / 2, [])])
                                match_ancestor_genes = filter(
                                    lambda g: g[1].id == match_neuron.id, s_genes)
                                if len(match_ancestor_genes) > 0:
                                    should_insert_at_root = match_ancestor_genes[0]

                            new_syn = Synapse((gene[3].input + o_gene[3].input) / 2,
                                              (gene[3].weight + o_gene[3].weight) / 2, [])

                            new_neuron = Neuron(o_gene[1].id + len(map(lambda n:
                                                                       map(lambda s: s.next_neurons, n.synapses), match_neuron)) + 1, [new_syn])
                            if should_insert_at_root != False:
                                should_insert_at_root[2].next_neurons = [
                                    match_neuron]
                                root_neuron.synapses.append(
                                    should_insert_at_root[2])

                    if len(new_gen) == self.count_individuals:
                        break

        self.adjust_weight(datas)
        self.root_neurons = new_gen
