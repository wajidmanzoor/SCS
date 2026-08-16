# Serial CPU implementation

## Overview

This directory contains the single-process, single-threaded CPU implementation
of size-bounded community search (SCS). Each invocation loads one graph and
solves one query.

Given an undirected graph, a query vertex `QID`, and a size range `[N1, N2]`,
the program searches for a connected subgraph that:

1. contains `QID`;
2. has between `N1` and `N2` vertices; and
3. maximizes the minimum degree of the induced subgraph.

## 1. Requirements

- GNU Make
- A C++14 compiler (`g++` by default)

This implementation does not require CUDA, MPI, or other external libraries.

## 2. Compilation

From the repository root, run:

```bash
make -C cpu/serial
```

Alternatively, from this directory, run:

```bash
make
```

Both commands create `cpu/serial/my_program`. To set different compiler flags:

```bash
make -C cpu/serial CXXFLAGS="-O0 -g -std=c++14 -Wall -Wextra"
```

To remove the executable:

```bash
make -C cpu/serial clean
```

## 3. Input graph format

The graph must be an undirected edge list. The first line gives the number of
vertices and edges; each remaining line describes one edge:

```text
<#Vertices> <#Edges>
<u0> <v0>
<u1> <v1>
...
```

Vertex identifiers are zero-based and must be in the range `0` through
`<#Vertices> - 1`.

## 4. Running a query

From the repository root, use:

```text
./cpu/serial/my_program GRAPH N1 N2 QID \
  HEU2 HEU3 HEU4 \
  UB1 UB2 UB3 UB3_OPT \
  CORE_MAINT NEW2VI DEL_VR \
  DOM DOM_THRESHOLD MAX_TIME_SECONDS SEARCH_ORDER VERBOSE
```

For example:

```bash
./cpu/serial/my_program \
  ./data/hepPH_SCS \
  3 6 2485 \
  1 1 1 \
  1 1 1 0 \
  1 1 1 \
  0 0 36000 1 0
```

### Query arguments

| Argument | Description |
| --- | --- |
| `GRAPH` | Path to the graph edge-list file |
| `N1` | Minimum allowed solution size |
| `N2` | Maximum allowed solution size |
| `QID` | Query vertex identifier |

### Algorithm options

Most algorithm options are flags: use `1` to enable an option and `0` to
disable it.

| Argument | Description |
| --- | --- |
| `HEU2`, `HEU3`, `HEU4` | Three heuristic strategies |
| `UB1`, `UB2`, `UB3` | Three upper-bound pruning rules |
| `UB3_OPT` | UB3 optimization |
| `CORE_MAINT` | Core-maintenance reduction |
| `NEW2VI` | Reduction that moves vertices into `VI` |
| `DEL_VR` | Removal of vertices from `VR` |
| `DOM` | Domination-based branching (`0` selects ordinary exact branching) |
| `DOM_THRESHOLD` | Domination-pair threshold; normally `0` when `DOM=0` |
| `MAX_TIME_SECONDS` | Per-query time limit in seconds |
| `SEARCH_ORDER` | Branching/search-order strategy |
| `VERBOSE` | Print the search tree when set to `1` |

## 5. Output

The program writes the following information to standard output:

- the graph, query, and enabled algorithm options;
- the best minimum degree (`mindeg`);
- the number of vertices in the solution (`H.size`);
- elapsed time in microseconds; and
- the solution vertex identifiers.

If the time limit is reached, the program prints `overtime` followed by the
best solution found before the timeout.
