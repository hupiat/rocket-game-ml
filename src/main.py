from genetic import Genetic

genetic = Genetic()

genetic.generation_gen()


def normalize(datas):
    MAX_HEIGHT_PX = 1080
    MAX_WIDTH_PX = 1920
    def compute(val, max, min): return (val - min) / max - min
    for data in datas:
        data.rocket_top = compute(data.rocket_top, MAX_HEIGHT_PX, 0)
        data.wall_left = compute(data.wall_left, MAX_WIDTH_PX, 0)
        data.wall_length = compute(data.wall_left, MAX_HEIGHT_PX, 0)
        if data.wall_direction > 1:
            data.wall_direction = 1
        if data.wall_direction < 0:
            data.wall_direction = 0


def ask_model(datas):
    normalize(datas)
    for output in genetic.generation_train(datas):
        print(output)
        yield output > 0.5


def step_generation(final_datas):
    normalize(final_datas)
    genetic.generation_crossover(final_datas)
