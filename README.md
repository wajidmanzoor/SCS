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


# Current Versions 

**1. branchPrunNeigRem:**
   - **Initial Reduction and Filtering**: The algorithm begins by applying initial reduction rules based on two criteria:
     - The core value of each vertex must be greater than lower bound of maximun minimum degree.
     - The distance of each vertex from Querry Index must be less than or equal to upper bound distance from query vertex.
    If verticies fail these condition they are removed from R.
    To avoid contention among threads for write locations, the array is divided into partitions of size `pSize`, and local counters are stored in shared memory. This way, only threads within a warp compete for write locations, minimizing the contention compared to a global approach.
    
   - **Vertex Compression and Neighbor Calculation**: After the initial reduction, the algorithm compresses the vertex data, updating their status and calculates degrees in C and degree in R. It computes new neighbor lists by considering only those neighbors with non-zero degrees after vertex elimination. neighbors of a single vertex are processed by a wrap so only they can compete for write locations, reducing contetion. 

   - **Reduction Rules and Task Processing**: Apply  reduction rules
        - If the sum of degrees in C and R is less than the current maximum minimum degree, the vertex is removed from R (status set to 2), and the degrees of its neighbors are updated accordingly.
        - If the sum of degrees in C and R equals the maximum minimum degree, all neighbors of the vertex are added to C (status set to 1), and the degrees of both the vertex and its neighbors are updated.
        - Calculate the ustar and minimum degree.
        - Calculate the upper bound using both algorithm, this minimum of those is used for branch pruning.

   - **Dominating Set Identification**: For each task, vertices dominated by `ustar` are identified, and their connection scores are calculated. The algorithm retains only the top-scoring dominating vertices, ensuring that the most influential vertices are considered in subsequent steps.

   - **Task Expansion and Buffer Management**: The algorithm then creates new tasks using `ustar` and the dominating set vertices. It manages buffer access with locks to ensure efficient read and write operations, especially when partitions are full or when tasks need to be retrieved from the buffer. Simultaneous read and write operations are supported, with specific warps handling buffer access to maintain data integrity.
    - The `C + ustar` task is written at the same location by updating the status of `ustar` and the dominating set vertices.
    - The `C - ustar` and `C + dominating set + ustar` tasks are written in task array with offset determined by `jump` partitions.
    - If a partition is full, the task is written to the buffer.
    - If a task doesn't have a dominating set and its partition is at max at `copyLimit`% capacity, one task is read from the buffer and written into the task array with offset determined by `jump` partitions.
    - If a warp has no tasks to process, it reads up to `readLimit` tasks from the buffer and writes them into the task array with      offset determined by `jump` partitions.
    - Simultaneous read and write operations from the buffer are enabled using locks:
    - One warp accesses the buffer offset to obtain locations for writing, releasing the lock after obtaining the locations.
    - One warp accesses the buffer offset to obtain locations for reading, releasing the lock after obtaining the locations.

   - **Overall Optimization**: Throughout the process, the algorithm optimizes task handling by removing tasks without relevant `ustar` and enabling warps with no tasks to process to read from the buffer. This approach ensures that the algorithm efficiently distributes tasks and reduces the overall number of levels in the search process, leading to faster and more accurate community detection.
**2. branchPrun:**
    Same as the (1) except removing the verticies from neighbor list
**3. New Version Implementation**
   - **Reduction Rules**: Implemented all reduction rules.
   - **Branch Pruning**:
     - Applied Branch Pruning Rule 1.
     - No dominating branching.
   - **Kernel Updates**:
     - Combined reduction and degree update in a single kernel.
     - Only updating the degree of elements affected by the removed vertex.
   - **Task Generation**:
     - Each task generates two new tasks:
       - The \( C + u^* \) task is updated by changing the status of the old task.
       - The \( C - u^* \) task is written to a new location.
   - **Jump Strategy**: Used to get the partition where the new task will be written.
   - **Issues**:
     - The new implementation isn't faster than the CPU version.
     - Many warps have no tasks to process.
     - Task warp imbalance.

**4. domsLimitVersion:**
   - **Reduction Rules**: Implemented all reduction rules.
   - **Branch Pruning**:
     - Applied Branch Pruning Rule 1.
     - Implemented dominating branching with a user-defined limit on the number of dominating branches.
   - **Kernel Updates**:
     - Combined reduction and degree updates into a single kernel, updating only the degrees of elements affected by the removed vertex.
     - Merged the write operations for \( C - u^* \) tasks and dominating tasks into one kernel.
   - **Buffer Management**:
     - Implemented a buffer to write tasks if a partition is full.
     - Enabled simultaneous read and write operations from the buffer.
     - Used locks to manage buffer access:
       - One warp accesses the buffer offset to get locations for writing and releases the lock after obtaining the locations.
       - Another warp accesses the buffer offset to get locations for reading and releases the lock after obtaining the locations.
     - TODO: Can be utilized to distribute tasks effectively.
   - **Task Optimization**:
     - Removed tasks without \( u^* \) from the task list.
     - Warps with no tasks to process read tasks from the buffer and write them into the task list.
     - Warps with task that have no dominating set read from the buffer and write the into the task list. 

**5. Dominating Version:**
   - **Reduction Rules**: Implemented all reduction rules.
   - **Branch Pruning**:
     - Applied Branch Pruning Rule 1.
     - Implemented dominating branching.
   - **Kernel Updates**:
     - Combined reduction and degree updates into a single kernel, updating only the degrees of elements affected by the removed vertex.
     - Merged the write operations for \( C - u^* \) tasks and dominating tasks into one kernel.
   - **Issues**:
     - Dominating write operations take a considerable amount of time if the dominating set is large.
     - Memory becomes an issue as each warp writes all new tasks from the dominating set into one buffer.

**6. sep_deg_update:**
   - **Reduction Rules**: Implemented all reduction rules.
   - **Branch Pruning**:
     - Applied Branch Pruning Rule 1.
     - No dominating branching.
   - **Kernel Updates**:
     - Separated reduction and degree update into different kernels.
     - Updated the degree of all elements affected by the removed vertex.
   - **Task Generation**:
     - Each task generates two new tasks:
       - \( C + u^* \) is updated by changing the status of the old task.
       - \( C - u^* \) is written to a new location.
   - **Partitioning Strategy**: 
     - Jump strategy used to determine the partition where the new task will be written.
   - **Issues**:
     - The implementation isn't faster than the CPU version.
     - Many warps have no tasks to process.
     - Imbalance in task distribution across warps.

7. old_version
    - Memory efficient version 


## Progress

1. Implemented the one graph multiple querry implementation. 

## Notes

### [Serial Code ](SerialCode.md)


### [Parallel Code ](ParallelCode.md)


