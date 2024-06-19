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







