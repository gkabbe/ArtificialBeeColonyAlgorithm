import logging

import numpy as np


logger = logging.getLogger(__name__)


class Bee:
    def __init__(self, pos, *, func, limit, search_radius=1):
        self.pos = np.atleast_1d(pos)
        self.func = func
        self.limit = limit
        self.search_radius = search_radius
        self._fitness = None
        self.tries = 0

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
        new_pos = (self.pos +
            np.random.uniform(-self.search_radius, self.search_radius, 2) * (random_pos - self.pos))
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
    pass
