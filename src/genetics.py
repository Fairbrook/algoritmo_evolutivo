from ast import List
from msilib.schema import Error
from numbers import Number
import random
from typing import Callable, Tuple
import functools


class Genetics:
    def __init__(self, domain: tuple, population_len: int, mutation_rate: float = 0, evaluation_cb: Callable[[float, float], float] = lambda x, y: 0, minima=False, seed=None) -> None:
        self.domain = domain
        self.length = population_len
        self.mutation_rate = mutation_rate
        self.evaluate = evaluation_cb
        self.population = []
        self.minima = minima
        if seed == None:
            _seed = random.random()
            print("Semilla:", _seed)
            random.seed(_seed)
            return

        random.seed(seed)

    def gen_initial_population(self) -> List:
        self.population = [(random.uniform(self.domain[0], self.domain[1]), (random.uniform(
            self.domain[0], self.domain[1]))) for i in range(self.length)]
        return self.population

    def __evaluate(self) -> List:
        minima = -1 if self.minima else 1
        evaluations = [(i, minima*self.evaluate(i[0], i[1]))
                       for i in self.population]
        evaluations.sort(key=lambda i: i[1])
        return evaluations

    def evaluate_population(self) -> List:
        evaluations = self.__evaluate()
        return [(ev[0], -ev[1] if self.minima else ev[1]) for ev in evaluations]

    def format_evaluations(evaluations: List) -> str:
        # evaluations = self.evaluate_population()
        res = ""
        for eval in evaluations:
            res += f"({eval[0][0]},{eval[0][1]})\t->\t{eval[1]}\n"
        return res

    def __normalize_evaluations(evaluations: List) -> List:
        total = functools.reduce(lambda res, e: res+e[1], evaluations, 0.0)
        return [(e[0], e[1]/total) for e in evaluations]

    def __accumulate_evaluations(normalized: List) -> List:
        accumulator = 0
        res = []
        for norm in normalized:
            accumulator += norm[1]
            res.append((norm[0], accumulator))
        return res

    def __choose_parents(accumulated: List) -> List:
        parents = []
        for n in range(0, len(accumulated), 2):
            tolerance = 10
            while tolerance > 0:
                r_a = random.uniform(0, 1)
                r_b = random.uniform(0, 1)
                parent_a = None
                parent_b = None
                for parent in accumulated:
                    if parent_a == None and parent[1] >= r_a:
                        parent_a = parent
                    if parent_b == None and parent[1] >= r_b and parent != parent_a:
                        parent_b = parent
                    if parent_a != None and parent_b != None:
                        break
                if parent_a != None and parent_b != None:
                    break
                tolerance -= 1
            if parent_b == None or parent_a == None:
                raise Exception(
                    "No hay suficiente material genético. Se están creando clones!!")
            parents.append(parent_a)
            parents.append(parent_b)
        return parents

    def __mutate(self, individual: Tuple) -> Tuple:
        r = random.uniform(0, 1)
        if r < self.mutation_rate:
            left = random.randrange(0, 1)
            if left == 1:
                return (random.uniform(self.domain[0], self.domain[1]), individual[1])
            return (individual[0], random.uniform(self.domain[0], self.domain[1]))
        return individual

    def __combine_and_mutate(self, parents: List) -> List:
        children = []
        for i in range(0, len(parents), 2):
            parent_a = parents[i][0]
            parent_b = parents[i+1][0]
            child_a = self.__mutate((parent_a[0], parent_b[1]))
            child_b = self.__mutate((parent_b[0], parent_a[1]))
            children.append(child_a)
            children.append(child_b)
        return children

    def next_generation(self) -> List:
        evaluations = self.__evaluate()
        evaluations = Genetics.__normalize_evaluations(evaluations)
        evaluations = Genetics.__accumulate_evaluations(evaluations)
        parents = Genetics.__choose_parents(evaluations)
        children = self.__combine_and_mutate(parents)
        self.population = children
        return children
