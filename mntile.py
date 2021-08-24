from dataclasses import dataclass, field
from typing import Tuple, List, TextIO
from enum import Enum
import sys


@dataclass(frozen=True, order=True)
class TilePuzzleState:
    """Represents the state of an MxN tile puzzle."""

    puzzle: Tuple[int, ...]  # An MxN-sized list
    blank: int = field(default=-1, hash=False, compare=False)

    @property  # Not a classmethod property because, sadly, pypy doesn't support classmethod properties yet
    # (at least not on dataclasses)
    def width(self):
        return self.__class__._width

    @property  # Not a classmethod property because, sadly, pypy doesn't support classmethod properties yet
    # (at least not on dataclasses)
    def height(self):
        return self.__class__._height

    @property  # Not a classmethod property because, sadly, pypy doesn't support classmethod properties yet
    # (at least not on dataclasses)
    def size(self):
        return self.__class__._width * self.__class__._height

    def __post_init__(self):
        if self.blank == -1:
            object.__setattr__(self, "blank", self.puzzle.index(0))


TilePuzzleState._width = 4  # For now
TilePuzzleState._height = 4


class TileWeight(Enum):
    unit_weight = 0
    squared = 1


class SlideDirection(Enum):
    left = 0
    up = 1
    down = 2
    right = 3
    no_slide = 4


class TilePuzzle:
    _h_increment: List[List[float]]

    def __init__(
        self,
        operator_order=(
            SlideDirection.right,
            SlideDirection.left,
            SlideDirection.down,
            SlideDirection.up,
        ),
        goal_state=None,
        weight=TileWeight.unit_weight,
        use_manhattan=True,
        error_rate=0,
        error_lower=0,
        error_upper=0,
    ):
        self.weight = weight
        self.use_manhattan = use_manhattan
        self.error_rate = error_rate
        self.error_lower = error_lower
        self.error_upper = error_upper

        if set(operator_order) != set(SlideDirection) - {SlideDirection.no_slide}:
            raise Exception("All 4 operators need to be provided to specify their order")
        self.operators_in_order = operator_order

        w = self.width
        h = self.height
        self.applicable_operators = [[]] * self.size
        for blank in range(self.size):
            applicable = []
            for op in self.operators_in_order:
                if op == SlideDirection.up and blank > w - 1:
                    applicable.append(op)
                elif op == SlideDirection.left and blank % w > 0:
                    applicable.append(op)
                elif op == SlideDirection.right and blank % w < w - 1:
                    applicable.append(op)
                elif op == SlideDirection.down and blank < w * h - w:
                    applicable.append(op)
            self.applicable_operators[blank] = applicable

        if goal_state is None:
            goal_state = TilePuzzleState(
                tuple(range(TilePuzzle._height * TilePuzzle._width)), 0
            )  # TilePuzzle.size is not known when
            # __init__ is defined so I can't set this
            # value as the default in the declaration
        self.set_goal(goal_state)

    def set_goal(self, goal_state: TilePuzzleState):
        """Also initializes _h_increment,
        where _h_increment[x][y] is how much to add to h when seeing tile x in position y"""

        if set(goal_state.puzzle) != set(range(self.size)):
            raise Exception(f"Bad goal state {goal_state}")

        self._goal = goal_state

        self._h_increment = [None] * self.size
        for i in range(1, self.size):
            self._h_increment[i] = [0] * self.size
        for goal_pos in range(self.size):
            tile = goal_state.puzzle[goal_pos]
            if tile == 0:  # Blank doesn't contribute to h because it's moved in every operator
                # and we count the movement of the other tile each time
                continue
            for pos in range(self.size):
                self._h_increment[tile][pos] = abs(goal_pos % self.width - pos % self.width) + abs(
                    goal_pos // self.width - pos // self.width
                )  # # difference in column + difference in row

    def h(self, state: TilePuzzleState) -> int:
        min_dist = 0

        if self.weight == TileWeight.unit_weight:
            for i, tile in enumerate(state.puzzle):
                if tile == 0:
                    continue  # Don't count for blank
                min_dist += self._h_increment[tile][i]
        elif self.weight == TileWeight.squared:
            for i, tile in enumerate(state.puzzle):
                if tile == 0:
                    continue  # Don't count for blank
                min_dist += self._h_increment[tile][i] * tile ** 2

        if self.error_rate != 0:
            # insert a uniform additive error according to the hash of the state
            state_hash = hash(state)
            random_from_hash = (
                state_hash * 1.0 / 2 ** sys.hash_info.width + 0.5
            )  # +0.5 because half the hashes are negative integers
            if random_from_hash < self.error_rate:
                second_random_from_hash = random_from_hash / self.error_rate  # 0 <= random_from_hash < error_rate so
                # 0 <= second_random_from_hash < 1
                min_dist += round(second_random_from_hash * (self.error_upper - self.error_lower)) + self.error_lower
                if min_dist < 0:
                    min_dist = 0

        return min_dist

    def goal_test(self, state: TilePuzzleState):
        return self._goal == state

    def get_successors_and_op_cost(self, state: TilePuzzleState) -> List[Tuple[TilePuzzleState, int]]:
        neighbors_and_op_costs = []
        if self.weight == TileWeight.unit_weight:
            for op in self.applicable_operators[state.blank]:
                new_state_puzzle, new_state_blank = self.apply_op(op, state.puzzle, state.blank)
                neighbors_and_op_costs.append((TilePuzzleState(new_state_puzzle, new_state_blank), 1))
        elif self.weight == TileWeight.squared:
            for op in self.applicable_operators[state.blank]:
                new_state_puzzle, new_state_blank = self.apply_op(op, state.puzzle, state.blank)
                neighbors_and_op_costs.append(
                    (
                        TilePuzzleState(new_state_puzzle, new_state_blank),
                        new_state_puzzle[state.blank] ** 2,
                    )
                )
        return neighbors_and_op_costs

    def get_actions(self, state: TilePuzzleState):
        """Returns the applicable operators from <state>, in correct order"""
        return list(self.applicable_operators[state.blank])

    def get_action(self, state1: TilePuzzleState, state2: TilePuzzleState):
        """Returns the operator to apply to state1 to get to state2"""
        row1 = state1.blank % self.width
        col1 = state1.blank // self.height
        row2 = state2.blank % self.width
        col2 = state2.blank // self.height

        if row1 == row2:
            if col1 > col2:
                return SlideDirection.left
            return SlideDirection.right
        else:
            if row1 > row2:
                return SlideDirection.up
            return SlideDirection.down

    def apply_op(self, op: SlideDirection, orig_puzzle: Tuple[int, ...], blank: int) -> Tuple[Tuple[int, ...], int]:
        #  We actually do the swap to maintain consistency when using abstract states
        #  (these contain -1 in some positions, including possibly the blank position.)
        puzzle = list(orig_puzzle)  # timeit showed copying and swapping is faster than constructing
        # a tuple with slice unpacking and two swapped items. Changing the variable name to help mypy
        # cope with the type change.
        w = self.width
        h = self.height
        if op == SlideDirection.up:
            if blank >= w:
                puzzle[blank], puzzle[blank - w] = puzzle[blank - w], puzzle[blank]
                blank -= w
            else:
                raise Exception(f"Up operator is invalid for {puzzle}")
        elif op == SlideDirection.down:
            if blank < self.size - w:
                puzzle[blank], puzzle[blank + w] = puzzle[blank + w], puzzle[blank]
                blank += w
            else:
                raise Exception(f"Down operator is invalid for {puzzle}")

        elif op == SlideDirection.right:
            if blank % w < w - 1:
                puzzle[blank], puzzle[blank + 1] = puzzle[blank + 1], puzzle[blank]
                blank += 1
            else:
                raise Exception(f"Right operator is invalid for {puzzle}")
        elif op == SlideDirection.left:
            if blank % w > 0:
                puzzle[blank], puzzle[blank - 1] = puzzle[blank - 1], puzzle[blank]
                blank -= 1
            else:
                raise Exception(f"Left operator is invalid for {puzzle}")

        return tuple(puzzle), blank

    @staticmethod
    def invert_op(op: SlideDirection):
        return {
            SlideDirection.up: SlideDirection.down,
            SlideDirection.down: SlideDirection.up,
            SlideDirection.right: SlideDirection.left,
            SlideDirection.left: SlideDirection.right,
        }[op]

    @staticmethod
    def read(instances_file: TextIO) -> List[TilePuzzleState]:
        ret = []
        for line in instances_file:
            start_puzzle = tuple(int(item) for item in line.split())
            if set(start_puzzle) != set(range(TilePuzzle._height * TilePuzzleState._width)):
                raise Exception(f"Bad start puzzle {start_puzzle}")
            blank = start_puzzle.index(0)
            ret.append(TilePuzzleState(start_puzzle, blank))

        return ret

    def generate_instances(self, instances_path, suboptimal_solutions_path, num_instances, min_ops, max_ops, seed=123):
        """The optimal number of operations to solve the generated instances may be smaller than the number
        that was use for generating them."""
        import random

        random.seed(seed)
        with open(instances_path, "w") as instances, open(suboptimal_solutions_path, "w") as suboptimal_solutions:
            for instance_num in range(num_instances):
                num_ops = random.randrange(min_ops, max_ops + 1)
                puzzle = tuple(range(TilePuzzle._height * TilePuzzleState._width))  # Start from the solved state
                blank = 0
                ops = []
                for i in range(num_ops):
                    op = random.choice(self.applicable_operators[blank])
                    puzzle, blank = self.apply_op(op, puzzle, blank)
                    ops.append(op)
                instances.write(f'{" ".join(str(p) for p in puzzle)}\n')
                suboptimal_solutions.write(f'{",".join(reversed([op.name for op in ops]))}\n')

    @property  # Not a classmethod property because, sadly, pypy doesn't support classmethod properties yet
    # (at least not on dataclasses)
    def width(self):
        return self.__class__._width

    @property  # Not a classmethod property because, sadly, pypy doesn't support classmethod properties yet
    # (at least not on dataclasses)
    def height(self):
        return self.__class__._height

    @property  # Not a classmethod property because, sadly, pypy doesn't support classmethod properties yet
    # (at least not on dataclasses)
    def size(self):
        return self.__class__._width * self.__class__._height


TilePuzzle._width = 4  # For now
TilePuzzle._height = 4


if __name__ == "__main__":
    import sys
    import re

    height, width = re.match(r"(\d+)x(\d+)", sys.argv[0]).groups()
    TilePuzzleState._width = TilePuzzle._width = int(height)
    TilePuzzleState._height = TilePuzzle._height = int(width)

    with open(sys.argv[0]) as f:
        start_states = TilePuzzle.read(f)
