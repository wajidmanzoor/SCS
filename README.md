# size-bounded community search (SCS)

###Note : Intial notes added will add more details 


## Problem Statement

Given:
1. graph $G = (V,E)$
2. query $q \in V$ 
3. size constraint $[l,h]$

Find subgraph $H$ of $G$ that satisfies the bellow conditions
1. Connected: $H$ is connected and contains $q$
2. Size bound : $l < |V(H)| < h $.
3. Minimum degree of H is maximum among all sub graphs that satisfy above conditions

## Progress

1. Implemented draft version, with reduction rule 1,2,3. (Returns right answer)
2. Need to implement time efficient version. (Fixed length array at the beginning) -Implementation compeleted 
3. Need to implement size efficient version. (Shared memory version ).
4. Compare results. 

## Notes

### [Serial Code ](SerialCode.md)


### [Parallel Code ](ParallelCode.md)


# Current Version 
1. dominating_version : 
    - Dominating branching version 
    - Separate expand and Dominating Expand
    - Separate reduction and degree update
    - Update degree of all vertices
    - issue: Dominating write takes a lot of time to write if doms set is large.
    - issue: As each wrap writes all new task from doms set in one buffer, memory becomes a issue.

2. new_version
    - All reduction rules
    - Branch pruning rule 1
    - No dominating branching 
    - reduction and degree update in same kernel, only updating the degree of those elements who will be affected by removed vertex
    - As each task will generate two new tasks, C+ ustar will be written by changing the status of old task
    - C-ustar will be written in a new location.
    - Jump strategy used get the partition where new task will be written. 
    - Issue: Isn't faster than CPU version 
    - Issue: Lots of wraps will have no task to process
    - Issue: Task wrap imbalance
3. sep_deg_update
    - All reduction rules
    - Branch pruning rule 1
    - No dominating branching 
    - reduction and degree update in different kernel, degree of all elements who will be affected by removed vertex
    - As each task will generate two new tasks, C+ ustar will be written by changing the status of old task
    - C-ustar will be written in a new location.
    - Jump strategy used get the partition where new task will be written. 
    - Issue: Isn't faster than CPU version 
    - Issue: Lots of wraps will have no task to process
    - Issue: Task wrap imbalance

3. old_version
    - Memory efficient version 
  







