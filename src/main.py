from genetic import Genetic

genetic = Genetic()

genetic.generation_gen()

def normalize(datas):
  for data in datas:
    data.rocket_top = 100 / data.rocket_top
    data.rocket_left = 100 / data.rocket_left
    data.rocket_speed = 100 / data.rocket_speed
    data.wall_top = 100 / data.wall_top
    data.wall_left = 100 / data.wall_left

def ask_model(datas):
  normalize(datas)
  for output in genetic.generation_train(datas): yield output > 0.5

def step_generation(final_datas):
  normalize(final_datas)
  genetic.generation_crossover(final_datas)