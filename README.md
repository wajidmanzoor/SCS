# size-bounded community search (SCS)

### Note : Intial notes added will add more details 


## Problem Statement

Given: 
1. graph $G = (V,E)$
2. query $q \in V$ 
3. size constraint $[l,h]$

Find subgraph $H$ of $G$ that satisfies the bellow conditions
1. Connected: $H$ is connected and contains $q$
2. Size bound : $l < |V(H)| < h $.
3. Minimum degree of H is maximum among all sub graphs that satisfy above conditions


## Notes


### Progress

1. Implemented draft version, with reduction rule 1,2,3. (Returns right answer)
2. Need to implement time efficient version. (Fixed length array at the beginning) -Implementation compeleted 
3. Need to implement size efficient version. (Shared memory version ).
4. Compare results. 

### Our code 

**Explaination**

1. Read Graphs : The Graph should be edge list (seperated by tab ) with the first line represents numVerticies and numEdges
    The Data is stored in 
        1. pstart : Neighbors offset
        2. edges : Neighbor list
        3. degree : Degree of each vertex 
        4. q_dist : Distance from query vertex of each vertex
        5. core : Core value of each vertex

2. Get Core values of each vertex (currently on CPU): Calculated the core values of each vertex and stores that in Core array. 
TODO: Move to GPU

3. Calculate distance from query vertex (currenlty on CPU) : Calculates the distance of all vertices from query vertex and stores that info in q_dist

4. IntialReductionRules (GPU): copies vertices with core values > Klower and distance from QID <  upperBoundSize.

5. CompressTask (GPU) : Compress array returned by IntialReductionRules Kernel. 

6. SCSSpeedEff (GPU) :  This kernel applies a reduction rule to prune the set $\( R \)$. It also compares and updates the minimum degree, calculates $\( u^* \)$, and determines the potential maximum minimum degree of each subgraph. Using the current task (subgraph) and $\( u^* \)$, it generates new tasks (subgraphs) if the potential maximum degree exceeds $\( k_{\text{lower}} \)$. These new tasks are then added to the task array.


