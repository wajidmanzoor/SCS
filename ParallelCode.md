# Our code 

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

<img src="pics\Intial Reduction Rules.png"/>

5. CompressTask (GPU) : Compress array returned by IntialReductionRules Kernel. 

<img src="pics/Compress Task.png"/>

6. Process Task (GPU) :  This kernel applies a reduction rule to prune the set $\( R \)$. It also compares and updates the minimum degree, calculates $\( u^* \)$, and determines the potential maximum minimum degree of each subgraph. Using the current task (subgraph) and $\( u^* \)$, it generates new tasks (subgraphs) if the potential maximum degree exceeds $\( k_{\text{lower}} \)$. These new tasks are then added to the task array.

<img src="pics/ustar and size.png"/>


7. Expand Task (GPU): 

<img src="pics/Expand Task.png"/>
