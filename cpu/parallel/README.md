# Hybrid exact CPU runner

This target partitions each `dom=0` binary branch-and-bound tree across MPI
ranks. Rank 0 is coordinator-only. With seven queries and nine shards per query,
the required layout is 64 ranks:

```text
1 coordinator + (7 queries x 9 search shards) = 64 physical cores
```

Build:

```bash
make
```

The repository-level batch launcher is:

```bash
../run_cpu_hybrid_batches.sh
```

Each shard owns binary prefixes at the configured split depth. Prefixes are
assigned modulo nine, so their subtrees are disjoint and jointly cover the
ordinary `dom=0` search. The aggregate query degree is the maximum returned by
the nine shards. A query is marked overtime if any shard reaches its limit.

`time_us` is the maximum program-internal timer among the nine concurrently
running shards. It is not shell wall time. The companion `shard_times.csv`
contains all 63 worker timings for checking load balance.

All nine cores start and run for every query. They may not stay busy for the
same duration when a query is solved by the heuristic, prunes before the split
frontier, or has uneven subtrees.
