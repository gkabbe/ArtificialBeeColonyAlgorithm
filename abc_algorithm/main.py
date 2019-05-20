import logging

import numpy as np


logger = logging.getLogger(__name__)


class Bee:
    def __init__(self, pos, *, func, dim, limit, search_radius=1, lower_bound, upper_bound):
        self.pos = np.atleast_1d(pos)
        self.func = func
        self.dim = dim
        self.limit = limit
        self.search_radius = search_radius
        self._fitness = None
        self.tries = 0
        self.bounds = (lower_bound, upper_bound)

    def global_update(self):
        self.pos = np.random.uniform(lower_bound, upper_bound, dim)

    @classmethod
    def random_init(cls, *, func, dim, limit, search_radius=1, lower_bound=-100,
                    upper_bound=100):
        bee = cls(pos, func=func, dim=dim, limit=limit, search_radius=search_radius,
                  lower_bound=lower_bound, upper_bound=upper_bound)
        bee.global_update()
        return bee

    def calculate_fitness(self, pos):
        fx = self.func(pos)
        if fx >= 0:
            self._fitness = 1 / (1 + fx)
        else:
            self._fitness = 1 - fx
        return self._fitness

    @property
    def fitness(self):
        return self.calculate_fitness(self.pos)


class EmployedBee(Bee):
    def local_update(self, positions):
        logger.debug("Doing local update")
        random_pos = np.random.choice(positions)
        logger.debug("Testing new position %s", random_pos)
        i = np.random.randint(0, self.dim)
        new_pos_i = (self.pos[i] +
            np.random.uniform(-self.search_radius, self.search_radius, 2) * (random_pos[i] - self.pos[i]))
        new_pos = np.copy(self.pos)
        new_pos[i] = new_pos_i
        old_fitness = self._fitness
        logger.debug("Old fitness value: %.2e", old_fitness)
        new_fitness = self.calculate_fitness(new_pos)
        logger.debug("New fitness value: %.2e", new_fitness)
        if old_fitness is None or new_fitness > old_fitness:
            logger.debug("Updating position")
            self.pos[:] = new_pos
        else:
            logger.debug("Keeping old position")
            self.tries += 1


class OnlookerBee(EmployedBee):
    def choose_food_source(self, employed_bees: list):
        fitness_vals = [bee.fitness for bee in employed_bees]
        cumsum = np.cumsum(fitness_vals)
        idx = np.searchsorted(cumsum, np.random.uniform(0, cumsum[-1]))
        self.pos = employed_bees[idx].pos


class Scout(Bee):
    pass

