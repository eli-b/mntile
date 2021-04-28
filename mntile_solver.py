import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, TextIO
import time

from mntile import TilePuzzleState, TilePuzzle, SlideDirection, TileWeight


@dataclass(unsafe_hash=True, order=True)  # Not frozen to allow changing h and, consequently, f.
# The members that participate in the hash are immutable.
class AStarNode:
    f: float = field(hash=False, init=False)  # Order of fields determines comparison order
    h: float = field(hash=False)  # Tie-break in favor of smaller h -> higher g.
    g: float = field(hash=False)
    state: TilePuzzleState  # It does participate in __cmp__ (and TilePuzzleState is defined with the default
    # order=True) only to make tie-breaking deterministic
    parent: "AStarNode" = field(
        default=None, repr=False, hash=False, compare=False
    )  # The string is a forward reference
    in_open: bool = field(default=True, repr=False, hash=False, compare=False)

    def __post_init__(self):
        self.f = self.g + self.h


def replace_in_heap(heap, index_in_heap, item):
    # Replace in_closed with new_node in open
    parent_index = (index_in_heap - 1) >> 1
    while index_in_heap != 0:
        heap[parent_index], heap[index_in_heap] = (
            heap[index_in_heap],
            heap[parent_index],
        )
        index_in_heap = parent_index
        parent_index = (index_in_heap - 1) >> 1
    # in_closed is now at 0
    heapq.heapreplace(heap, item)


class NoSolution(Exception):
    pass


class Timeout(Exception):
    pass


class AStar:
    def __init__(self, puzzle: TilePuzzle):
        self._tile_puzzle = puzzle
        self._reset_stats()

    def _reset_stats(self):
        self.expanded = 0
        self.generated = 0
        self.reopened = 0
        self.cost = None
        self.cost_lower_bound = 0
        self.total_time = None

    def solve(self, state: TilePuzzleState, nodes_out: TextIO = None, timeout=60) -> List[SlideDirection]:
        """Returns the actions to reach the goal from the given state.
        Their combined cost is guaranteed to be optimal as long as
        the heuristic is admissible."""
        self._reset_stats()
        start_time = time.time()
        root_h = self._tile_puzzle.h(state)
        root = AStarNode(root_h, 0, state)
        self.generated += 1
        closed = {root.state: root}
        open_ = [root]

        nodes_out.write("{\n")

        while open_:
            if time.time() - start_time > timeout:
                nodes_out.write("}")
                self.total_time = time.time() - start_time
                raise Timeout(f"Timed out after {self.total_time} seconds.")

            node = heapq.heappop(open_)
            node.in_open = False

            self.cost_lower_bound = node.f

            if nodes_out is not None:
                nodes_out.write(
                    f"{hash(node)}: {{f: {node.f}, h: {node.h}, g: {node.g}, "
                    f"state: {list(node.state.puzzle)}, parent: {hash(node.parent)}}}\n"
                )

            if self._tile_puzzle.goal_test(node.state):
                nodes_out.write("}")
                self.total_time = time.time() - start_time
                self.cost = node.g
                return self.retrace(node), self.cost, self.total_time

            self.expanded += 1
            successor_states_and_op_costs = self._tile_puzzle.get_successors_and_op_cost(node.state)
            successor_hs = [self._tile_puzzle.h(state) for (state, delta_g) in successor_states_and_op_costs]
            # Reverse Path-Max (operators are invertible):
            max_inferred_h_of_parent = max(
                [h - delta_g for (h, (_, delta_g)) in zip(successor_hs, successor_states_and_op_costs)]
            )
            if node.h < max_inferred_h_of_parent:
                node.h = max_inferred_h_of_parent
            # One-Step Forward Path-Max (unlike in HOG2):
            successor_hs = [
                max(node.h - delta_g, h) for (h, (_, delta_g)) in zip(successor_hs, successor_states_and_op_costs)
            ]

            for (successor_state, delta_g), successor_h in zip(successor_states_and_op_costs, successor_hs):
                successor_g = node.g + delta_g
                if successor_state in closed:
                    in_closed = closed[successor_state]
                    successor_h = max(in_closed.h, successor_h)
                    if in_closed.g > node.g + delta_g:  # Found a better path to this node
                        new_node = AStarNode(
                            successor_h,
                            successor_g,
                            successor_state,
                            node,
                        )
                        closed[successor_state] = new_node
                        # Insert new_node into OPEN and remove in_closed from OPEN if it's in it
                        if in_closed.in_open:
                            index_in_open = open_.index(
                                in_closed
                            )  # TODO: Consider adding an index_in_open field to nodes
                            replace_in_heap(open_, index_in_open, new_node)
                            in_closed.in_open = False  # Just to be tidy
                        else:  # in_closed has already been expanded - reopen.
                            heapq.heappush(open_, new_node)
                            self.reopened += 1
                        self.generated += 1
                    elif in_closed.h < successor_h and in_closed.in_open:
                        in_closed.h = successor_h
                        index_in_open = open_.index(in_closed)  # TODO: Consider adding an index_in_open field to nodes
                        replace_in_heap(open_, index_in_open, in_closed)
                else:
                    new_node = AStarNode(
                        successor_h,
                        successor_g,
                        successor_state,
                        node,
                    )
                    closed[successor_state] = new_node
                    heapq.heappush(open_, new_node)
                    self.generated += 1

        nodes_out.write("}")
        self.total_time = time.time() - start_time
        raise NoSolution(f"No solution. Elapsed time: {self.total_time} seconds.")

    def retrace(self, node: AStarNode):
        reverse_order_actions = []

        while node.parent is not None:
            reverse_order_actions.append(self._tile_puzzle.get_action(node.parent.state, node.state))
            node = node.parent

        return list(reversed(reverse_order_actions))


if __name__ == "__main__":
    import pathlib
    import re
    import argparse

    parser = argparse.ArgumentParser(description="Sliding Tile Puzzle Solver", prog="mntile_solver")
    # parser.add_argument('input', metavar='INPUT_PATH', type=open, default=sys.stdin,
    #                    help='Input problem instances file path. If not provided, will read lines from stdin.')
    parser.add_argument(
        "input_path",
        metavar="INPUT_PATH",
        help="Input problem instances file path. "
        "File name must start with MxN where M and N are the dimensions of the puzzle.",
    )
    parser.add_argument("--nodes-dir", type=str, default="nodes", help="The directory to store nodes files in")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout per puzzle, in seconds")
    parser.add_argument("--error-rate", type=float, default=0, help="Error rate of the heuristic")
    parser.add_argument("--error-lower-bound", type=float, default=0, help="Error lower bound")
    parser.add_argument("--error-upper-bound", type=float, default=0, help="Error upper bound")
    parser.add_argument("--weighted", type=bool, default=False, help="Does moving a tile cost its value squared?")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")
    args = parser.parse_args()

    try:
        height, width = re.match(r"(?:.*/)?(\d+)x(\d+)", args.input_path).groups()
    except Exception as e:
        raise Exception("Input file name not of correct format (MxN_###)") from e
    TilePuzzleState._width = TilePuzzle._width = int(height)
    TilePuzzleState._height = TilePuzzle._height = int(width)

    tile_puzzle = TilePuzzle(
        weight=TileWeight.unit_weight if not args.weighted else TileWeight.squared,
        error_rate=args.error_rate,
        error_lower=args.error_lower_bound,
        error_upper=args.error_upper_bound,
    )
    a_star = AStar(tile_puzzle)

    with open(args.input_path) as f:
        start_states = TilePuzzle.read(f)
    for i, start_state in enumerate(start_states):
        print(f"Solving instance {i}: ", end="", flush=True)
        with open(f"{args.nodes_dir}/{pathlib.Path(args.input_path).name}_{i}", "w") as nodes:
            try:
                solution = a_star.solve(start_state, nodes_out=nodes, timeout=args.timeout)
                print(
                    f"solved with cost {a_star.cost} in {a_star.total_time:.3f} seconds "
                    f"({a_star.expanded} nodes expanded, {a_star.generated} nodes generated, "
                    f"{a_star.reopened} nodes reopened)"
                )
            except Timeout as e:
                print(
                    f"timed out at {a_star.total_time:.3f} seconds with cost lower bound {a_star.cost_lower_bound} "
                    f"({a_star.expanded} nodes expanded, {a_star.generated} nodes generated, "
                    f"{a_star.reopened} nodes reopened)"
                )
            except NoSolution as e:
                print(
                    f"no solution exists proven after {a_star.total_time:.3f} seconds "
                    f"({a_star.expanded} nodes expanded, {a_star.generated} nodes generated, "
                    f"{a_star.reopened} nodes reopened)"
                )
