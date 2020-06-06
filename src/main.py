from genetic import Genetic

genetic = Genetic()

genetic.generation_gen()


def normalize(datas, max_height, max_width):
    def compute(val, max, min): return (val - min) / max - min
    for data in datas:
        data.rocket_top = compute(data.rocket_top, max_height, 0)
        data.wall_left = compute(data.wall_left, max_width, 0)
        data.wall_length = compute(data.wall_left, max_height, 0)
        if data.wall_direction > 1:
            data.wall_direction = 1
        if data.wall_direction < 0:
            data.wall_direction = 0


def ask_model(datas, max_height, max_width):
    normalize(datas, max_height, max_width)
    for output in genetic.generation_train(datas):
        print(output)
        yield output > 0.5


def step_generation(final_datas):
    normalize(final_datas)
    genetic.generation_crossover(final_datas)
