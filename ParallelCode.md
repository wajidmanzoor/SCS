# Our code 

## 1. Read Graphs
The Graph should be edge list (seperated by tab ) with the first line represents numVerticies and numEdges
    The Data is stored in 
        1. pstart : Neighbors offset
        2. edges : Neighbor list
        3. degree : Degree of each vertex 
        4. q_dist : Distance from query vertex of each vertex
        5. core : Core value of each vertex

## 2. Get Core values of each vertex (currently on CPU)
Calculated the core values of each vertex and stores that in Core array. 
TODO: Move to GPU

## 3. Calculate distance from query vertex (currenlty on CPU)
Calculates the distance of all vertices from query vertex and stores that info in q_dist

## 4. IntialReductionRules (GPU)
Removes vertices that have core values less than or equal to the initial maximum minimum degree and a distance from the query vertex is greater than or equal to the upper bound size minus 1.

Suppose we have $n$ vertices, whose core values are stored in an array called *core* and distances from the query vertex are stored in an array called *dist_q*. We divide the vertices into partitions of size $ psize = \frac{n}{TOTAL_WARPS}+1$. This means the total number of partitions will be equal to *TOTAL_WARPS*.

We allocate memory for two arrays of size $pSize \times TOTAL_WARPS$. These arrays will store the vertices and their status if they satisfy certain conditions. Additionally, we allocate an array called *NumCounter* of size *TOTAL_WARPS* to keep track of the number of vertices written in each partition.

The processing unit in this context is a warp. The threads of each warp will read *psize* vertices and check if each vertex satisfies the given conditions. If a vertex satisfies the conditions, the thread will write the vertex and its status to its corresponding partitions in the output arrays. To avoid race conditions, threads inside a wrap use atomic add operation to determine the next location where the vertex will be written. the status of an vertex will be set to zero except if it is a query vertex.

Once a warp has processed all *psize* vertices, the first thread of the warp will write the local counter (i.e., the total number of vertices written in the partition) to the *NumCounter* array at the location corresponding to the warp's ID. Additionally, the first thread of each warp will atomically add the local counter to a global counter to keep track of the total number of vertices written.

**Input**: Core Value Array, Distance from Query Vertex Array

```cpp
// Assumption: local_counter is in shared memory
1: if laneId == 0 then
2:     local_counter[warpId] ← 0
3: end if
4: __syncwarp()
5: start,writeOffset ← warpId * pSize
6: end ← min((warpId + 1) * pSize, size)
7: total ← end - start
8: for i ← laneId to total - 1 step 32 do
9:     vertex ← start + i
10:    if coreValues[vertex] > lowerBoundDegree and distanceFromQID[vertex] < (upperBoundSize - 1) then
11:        loc ← atomicAdd(local_counter[warpId], 1)
12:        taskList[loc + writeOffset] ← vertex
13:        if vertex == queryVertex then
14:            taskStatus[loc + writeOffset] ← 1
15:        else
16:            taskStatus[loc + writeOffset] ← 0
17:        end if
18:    end if
19: end for
20: __syncwarp()
21: if laneId == 0 then
22:    numEntries[warpId] ← local_counter[warpId]
23:    atomicAdd(globalCounter, local_counter[warpId])
24: end if
```

**Output**: Vertex Array, Status Array, Num Counter Array, Global Count

<img src="pics\Intial Reduction Rules.png"/>

## 5. CompressTask (GPU)
Compress vertex and status array returned by IntialReductionRules Kernel. 

**Input**: Vertex Array, Status Array, Num Counter Array, Global Count

<img src="pics/Compress Task.png"/>

## 6. Process Task (GPU)  
This kernel applies a reduction rule to prune the set $\( R \)$. It also compares and updates the minimum degree, calculates $\( u^* \)$, and determines the potential maximum minimum degree of each subgraph. Using the current task (subgraph) and $\( u^* \)$, it generates new tasks (subgraphs) if the potential maximum degree exceeds $\( k_{\text{lower}} \)$. These new tasks are then added to the task array.

<img src="pics/ustar and size.png"/>


## 7. Expand Task (GPU)

<img src="pics/Expand Task.png"/>
