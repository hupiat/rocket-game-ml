from random import random, randrange
from synapse import Synapse

class Neuron:

  def ReLU(self):
    self.output = 0
    for syn in self.synapses:
      if syn.input < 0: syn.input = 0
      self.output += syn.input * syn.weight
      self.output /= synapses.count

  def __init__(self, id, synapses):
    self.id = id
    self.synapses = synapses
    ReLU()

  def iterate_children_recursive(self, callback):
    def iterate(neuron):
      for syn in neuron.synapses:
        for next_neuron in syn.next_neurons:
          callback(neuron, next_neuron, syn)
        iterate(neuron)
      
    iterate(self)

  def mutate(self):
    synapse_index = random()
    self.synapses[synapse_index].weight = random()
    ReLU()

  def mutate_struct(self):
    neurons = [self]

    iterate_children_recursive(lambda neuron, next_neuron: 
      neurons.append(next_neuron) if not any(neuron.id == next_neuron.id for neuron in neurons) else neuron)
    
    neurons.sort(id)
    last_neuron_id = neurons[neurons.count].id

    gen_index = randrange(1, neurons.count - 2)

    def gen_neuron():
      neuron = Neuron(gen_index, [Synapse(random(), random(), [neurons[gen_index + 1]])])
      neurons.insert(gen_index, gen_neuron)
      neurons[gen_index - 1].synapses.append(Synapse(random(), random(), [neuron]))
      for i in range(gen_index, neurons.count): neurons[i].id += 1

    def gen_synapse():
      ascending_exploration = random() > 0.5

      def explore(ascending):
        for i in range(gen_index if ascending else 0, neurons.count if ascending else gen_index):
          for syn in neurons[i].synapses:
            for next_neuron in syn.next_neurons:
              new_neuron_bond = filter(lambda neuron: 
                not any(neuron.id == next_neuron.id for neuron in neurons), neurons)
              if new_neuron_bond.count > 0:
                next_neuron.synapses.append(Synapse(random(), random(), [new_neuron_bond]))
                return True
        return False

      success = explore(ascending_exploration)
      if not success: explore(not ascending_exploration)

      return success

    should_gen_node = random() > 0.5
    if should_gen_node:
      gen_neuron()
    else:
      if not gen_synapse():
        gen_neuron()