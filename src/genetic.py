from data import Data
from random import random
from synapse import Synapse
from neuron import Neuron

class Genetic:

  generation_size = 20
  mutation_prob = 0.2
  mutation_struct_prob = 0.3

  def generation_gen(self):
    self.neurons = []
    for _ in range(0, self.generation_size):
      synapses = []
      for label in Data: synapses.append(Synapse(random(), random(), []))
      self.neurons.append(Neuron(0, synapses))
      
  def generation_crossover(self):
    for i in range(0, self.generation_size):
      should_mutate = random() <= self.mutation_prob
      if should_mutate: self.neurons[i].mutate()
      should_struct_mutate = random() <= self.mutation_struct_prob
      if should_struct_mutate: self.neurons[i].mutate_struct()
      
