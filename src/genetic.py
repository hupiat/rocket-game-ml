from data import Data
from random import random
from synapse import Synapse
from neuron import Neuron

class Genetic:

  generation_count = 10
  mutation_prob = 0.2
  mutation_struct_prob = 0.3
  crossover_unique_gene_transfer_prob = 0.2

  def map_data(self, data):
    return [
      Synapse(data.rocket_top, 1, []),
      Synapse(data.wall_direction, 1, []),
      Synapse(data.wall_left, 1, []),
      Synapse(data.wall_top, 1, []),
    ]

  def generation_gen(self):
    self.root_neurons = []
    for _ in range(0, self.generation_count):
      self.root_neurons.append(Neuron(
        0, self.map_data(Data(0, random(), random(), 1 if random() > 0.5 else 0, random(), random()))))

  def generation_train(self, datas):
    for i in range(0, len(datas)):
      data = self.map_data(datas[i])
      for neuron in self.root_neurons:
        neuron.synapses = list(data)
        neuron.ReLU()
        yield neuron.output
      
  def generation_crossover(self, datas):
    new_gen = []

    for i in range(0, self.generation_count):
      should_mutate = random() <= self.mutation_prob
      if should_mutate: self.root_neurons[i].mutate()
      should_struct_mutate = random() <= self.mutation_struct_prob
      if should_struct_mutate: self.root_neurons[i].mutate_struct()

    datas.sort(lambda d: d.score, True)
    for i in range (0, len(datas)):
      for j in range (0, len(datas)):
        if i == j: continue
        root_neuron = Neuron(0, [])
        f_genes, s_genes = []
        self.root_neurons[i].iterate_children_recursive(
          lambda neuron, next_neuron, syn: f_genes.append((neuron, next_neuron, syn)))
        self.root_neurons[j].iterate_children_recursive(
          lambda neuron, next_neuron, syn: s_genes.append((neuron, next_neuron, syn)))
        
        for gene in f_genes:
          for o_gene in s_genes:
            if gene[0].id == o_gene[0].id and gene[1].id == o_gene[1].id or random() <= self.crossover_unique_gene_transfer_prob:
              match_neuron = root_neuron.find_child(lambda n: n.id == o_gene[0].id) if o_gene[0].id != 0 else root_neuron 
              should_insert_at_root = False 
              if match_neuron is None:
                match_neuron = Neuron(o_gene[0].id, [])
                match_ancestor_genes = filter(lambda g: g[1].id == match_neuron.id, s_genes)
                if len(match_ancestor_genes) > 0: should_insert_at_root = match_ancestor_genes[0]

              new_neuron = Neuron(o_gene[1].id + len(map(lambda n: 
                map(lambda s: s.next_neurons, n.synapses), match_neuron)) + 1, [])
              new_syn = Synapse(gene[3].input + o_gene[3].input / 2, 
                gene[3].weight + o_gene[3].weight / 2, [new_neuron])

              match_neuron.synapses.append(new_syn)
              if should_insert_at_root is not False: 
                should_insert_at_root[2].next_neurons = [match_neuron]
                root_neuron.synapses.append(should_insert_at_root[2])

          if len(new_gen) == self.generation_count: break

    self.root_neurons = new_gen

