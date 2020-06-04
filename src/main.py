from genetic import Genetic

genetic = Genetic()

genetic.generation_gen()

def normalize(datas):
  def compute(val, max, min): return (val - min) / max - min
  for data in datas:
    data.rocket_top = compute(data.rocket_top, 1080, 0)
    data.rocket_left = compute(data.rocket_left, 1920, 0)
    data.wall_top = compute(data.wall_top, 1080, 0)
    data.wall_left = compute(data.wall_left, 1920, 0)

def ask_model(datas):
  normalize(datas)
  for output in genetic.generation_train(datas): yield output > 0.5

def step_generation(final_datas):
  normalize(final_datas)
  genetic.generation_crossover(final_datas)