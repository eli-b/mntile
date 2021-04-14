Tile Puzzle
---

This project implement a Sliding Tile Puzzle solver in Python.
Two cost functions are supports: unit cost, and squared tile value cost.

The solver currently uses A*, and implements one-step Bidirectional Path-Max. 
The heuristic we use is Manhattan distance (taking the cost function into
account).

The solver optionally takes an error rate and additive error bounds.
The rate indicates the portion of search nodes (based on their hash)
whose h will receive a deterministic additive integer error
(based again on their hash) within the given bounds. This simulates bugs
in the implementation of the heuristic function. Setting the error rate to 1
and the error bounds to be negative and equal to each other can simulate a weak
heuristic (like GAP-1).

The search tree may be output to a JSON file. This allows comparing the search
trees of different solvers with different error rates. 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
