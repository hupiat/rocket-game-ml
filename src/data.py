class Data:

    def __init__(self, id, score, rocket_top, wall_direction, wall_left, wall_length, monster_top, monster_left):
        self.id = id
        self.score = score
        self.rocket_top = rocket_top
        self.wall_direction = wall_direction
        self.wall_left = wall_left
        self.wall_length = wall_length
        self.monster_top = monster_top
        self.monster_left = monster_left

    def get_attributes(self):
        attrs = [a for a in dir(self) if not a.startswith('__')]
        return [a for a in attrs if a != 'id' and a != 'score']
