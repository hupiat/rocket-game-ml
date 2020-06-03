from genetic import Genetic

genetic = Genetic()

genetic.generation_gen()

def ask_model(datas):
  return genetic.generation_train(datas)

def step_generation(final_datas):
  genetic.generation_crossover(final_datas)