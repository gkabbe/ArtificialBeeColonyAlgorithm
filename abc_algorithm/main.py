import logging
import random

import fire
import numpy as np
from scipy.optimize import rosen


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

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self.pos})"

    def global_update(self):
        logger.debug("Old position: %s", self.pos)
        lower_bound, upper_bound = self.bounds
        self.pos = np.random.uniform(lower_bound, upper_bound, self.dim)
        logger.debug("New position: %s", self.pos)

    @classmethod
    def random_init(cls, *, func, dim, limit, search_radius=1, lower_bound=-100,
                    upper_bound=100):
        bee = cls([0] * dim, func=func, dim=dim, limit=limit, search_radius=search_radius,
                  lower_bound=lower_bound, upper_bound=upper_bound)
        bee.global_update()
        return bee

    @classmethod
    def init_from_other_bee(cls, bee):
        new_bee = cls(bee.pos, func=bee.func, dim=bee.dim, limit=bee.limit,
                      search_radius=bee.search_radius, lower_bound=bee.lower_bound,
                      upper_bound=bee.upper_bound)
        return new_bee

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
    def local_update(self, bees):
        logger.debug("Old position: %s", self.pos)
        logger.debug("Doing local update")
        random_pos = random.choice([bee.pos for bee in bees])
        logger.debug("Testing new position %s", random_pos)
        i = np.random.randint(0, self.dim)
        new_pos_i = (self.pos[i] +
            np.random.uniform(-self.search_radius, self.search_radius) * (random_pos[i] - self.pos[i]))
        new_pos = np.copy(self.pos)
        new_pos[i] = new_pos_i
        old_fitness = self._fitness
        logger.debug("Old fitness value: %.4e", old_fitness)
        new_fitness = self.calculate_fitness(new_pos)
        logger.debug("New fitness value: %.4e", new_fitness)
        if old_fitness is None or new_fitness > old_fitness:
            logger.debug("Updating position")
            self.pos[:] = new_pos
            self.tries = 0
        else:
            logger.debug("Keeping old position")
            self.tries += 1

    run = local_update


class OnlookerBee(EmployedBee):
    def choose_food_source(self, bees: list):
        fitness_vals = [bee.fitness for bee in bees]
        cumsum = np.cumsum(fitness_vals)
        idx = np.searchsorted(cumsum, np.random.uniform(0, cumsum[-1]))
        self.pos = bees[idx].pos

    def run(self, bees: list):
        self.choose_food_source(bees)
        self.local_update(bees)


class ScoutBee(Bee):
    def run(self, bees: list):
        self.global_update()


def square_well(x):
    return np.sum(x*x)


class Swarm:
    def __init__(self, func, dim, *, n_employed, n_onlooker, limit=10, max_cycles=100):
        self.employed_bees = [EmployedBee.random_init(func=func, limit=limit, dim=dim)
                              for _ in range(n_employed)]
        self.onlooker_bees = [OnlookerBee.random_init(func=func, limit=limit, dim=dim)
                              for _ in range(n_onlooker)]

        self.bees = self.employed_bees + self.onlooker_bees

        self.limit = limit
        self.max_cycles = max_cycles

    def run(self):
        for _ in range(self.max_cycles):
            self.step()

    def step(self):
        logger.info("Employed bee phase")
        for bee in self.employed_bees:
            bee.run(self.bees)

        logger.info("Onlooker bee phase")
        for bee in self.onlooker_bees:
            bee.run(self.bees)

        logger.info("Scout bee phase")
        for bee in self.employed_bees:
            if bee.tries > self.limit:
                logger.debug("Tries above limit")
                bee.global_update()
                bee.tries = 0

        # get bee with best fitness
        best_bee = max(self.bees, key=lambda bee: bee.fitness)
        logger.info("Best position so far: %s with fitness %s", best_bee.pos,
                    best_bee.fitness)


def main(n_employed, n_onlooker, limit=10, max_cycles=100, log_level="info"):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    func = rosen
    dim = 3

    swarm = Swarm(func, dim, n_employed=n_employed, n_onlooker=n_onlooker, limit=10,
                  max_cycles=max_cycles)
    swarm.run()


if __name__ == "__main__":
    fire.Fire(main)