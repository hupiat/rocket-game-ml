class Data:

  def __init__(self, score, rocket_top, rocket_left, rocket_speed, wall_left, wall_top):
    self.score = score
    self.rocket_top = rocket_top
    self.rocket_left = rocket_left
    self.rocket_speed = rocket_speed
    self.wall_left = wall_left
    self.wall_top = wall_top

  def __iter__(self):
    for attr in self.__dict__.iteritems():
        yield attr
