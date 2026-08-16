# Parallel CPU implementation

## Overview

This directory contains the MPI-based parallel CPU implementation of
size-bounded community search (SCS). It executes multiple queries at the same
time and partitions each query's exact binary branch-and-bound tree across a
fixed number of worker ranks.

Rank 0 acts only as the coordinator. Every other MPI rank is assigned to one
query and one search shard. Consequently, the required number of ranks is:

```text
1 + (number of queries x workers per query)
```

For example, seven queries with nine workers per query require 64 ranks:

```text
1 coordinator + (7 queries x 9 workers) = 64 ranks
```

## 1. Requirements

- GNU Make
- An MPI implementation that provides `mpic++` and `mpirun`
- A C++14 compiler supported by the MPI installation

This implementation does not require CUDA.

## 2. Compilation

From the repository root, run:

```bash
make -C cpu/parallel
```

Alternatively, from this directory, run:

```bash
make
```

Both commands create `cpu/parallel/cpu_hybrid`. To set different compiler
flags:

```bash
make -C cpu/parallel CXXFLAGS="-O0 -g -std=c++14 -Wall -Wextra"
```

To remove the executable:

```bash
make -C cpu/parallel clean
```

## 3. Input data formats

### Graph file

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

### Query file

Each nonempty query line must begin with three integers:

```text
<N1> <N2> <QID>
```

- `N1` is the minimum allowed solution size.
- `N2` is the maximum allowed solution size.
- `QID` is the query vertex identifier.

Blank lines and lines beginning with `#` are ignored. Extra columns after
`QID` are allowed, and a line beginning with `server_exit` stops query-file
parsing.

Example:

```text
# N1 N2 QID
3 6 2485
4 10 120
```

## 4. Running a batch

The complete command syntax is:

```text
mpirun -np RANKS ./cpu/parallel/cpu_hybrid \
  GRAPH QUERIES OUTPUT_CSV \
  HEU2 HEU3 HEU4 \
  UB1 UB2 UB3 UB3_OPT \
  CORE_MAINT NEW2VI DEL_VR \
  DOM DOM_THRESHOLD MAX_TIME_SECONDS SEARCH_ORDER VERBOSE \
  WORKERS_PER_QUERY SPLIT_DEPTH
```

For a query file containing seven queries and using nine workers per query,
`RANKS` must be 64. For example:

```bash
mpirun -np 64 ./cpu/parallel/cpu_hybrid \
  ./data/hepPH_SCS ./queries.txt ./results.csv \
  1 1 1 \
  1 1 1 0 \
  1 1 1 \
  0 0 36000 1 0 \
  9 4
```

### Files and query options

| Argument | Description |
| --- | --- |
| `GRAPH` | Path to the graph edge-list file |
| `QUERIES` | Path to the query file |
| `OUTPUT_CSV` | Path for the aggregate result CSV |
| `HEU2`, `HEU3`, `HEU4` | Three heuristic strategies (`0` or `1`) |
| `UB1`, `UB2`, `UB3` | Three upper-bound pruning rules (`0` or `1`) |
| `UB3_OPT` | UB3 optimization (`0` or `1`) |
| `CORE_MAINT` | Core-maintenance reduction (`0` or `1`) |
| `NEW2VI` | Reduction that moves vertices into `VI` (`0` or `1`) |
| `DEL_VR` | Removal of vertices from `VR` (`0` or `1`) |
| `DOM` | Must be `0`; this runner supports exact binary branching only |
| `DOM_THRESHOLD` | Normally `0` when `DOM=0` |
| `MAX_TIME_SECONDS` | Time limit for each shard; must be positive |
| `SEARCH_ORDER` | Branching/search-order strategy |
| `VERBOSE` | Print search-tree details when set to `1` |

### Parallel options

| Argument | Description |
| --- | --- |
| `WORKERS_PER_QUERY` | Number of worker ranks assigned to each query |
| `SPLIT_DEPTH` | Binary-tree depth at which prefixes are distributed; valid range is 1 to 63 |

The number of prefixes, `2^SPLIT_DEPTH`, must be at least
`WORKERS_PER_QUERY`. The program exits with an error if the rank count or split
depth is inconsistent with the query file.

## 5. Output

The coordinator creates two files:

- `OUTPUT_CSV` contains one aggregate row per query. The reported degree is the
  maximum found across its workers, and `time_us` is the maximum internal
  worker time for that query.
- `OUTPUT_CSV.shards.csv` contains the degree, elapsed time, timeout state, and
  worker rank for every shard. Use it to inspect load balance.

The batch-level `batch_search_wall_us` field measures concurrent search wall
time. `batch_total_wall_us` also includes graph loading and core decomposition.
A query is marked as overtime if any of its shards reaches the time limit.

## 6. How the search is partitioned

At `SPLIT_DEPTH`, binary branch prefixes are assigned to workers modulo
`WORKERS_PER_QUERY`. These subtrees are disjoint and together cover the normal
`DOM=0` search tree. All workers start for every query, although they can finish
at different times because of pruning, heuristic solutions, and uneven
subtrees.
