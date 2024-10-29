__device__ __forceinline__ void calculateMinimumDegree(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,ui* sharedDegree, ui startIndex, ui start, ui total, int laneId);
__device__ __forceinline__ void reductionRule3(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, ui size, ui totalEdges, ui startIndex, ui otherStartIndex, ui start,ui end, ui total , ui queryId,ui iter, int laneId);
__device__ __forceinline__ void reductionRule1and2(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, ui size, ui totalEdges, ui startIndex, ui otherStartIndex, ui start,ui end, ui total , ui queryId, ui upperBoundSize,ui iter, ui red1, ui red2,int laneId);
__device__ __forceinline__ void calculateUpperBoundDegree(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,ui* sharedUBDegree, ui* sharedC_, ui* sharedCounterC, ui* sharedCounterR, ui maxN2, ui startIndex, ui otherStartIndex, ui start, ui total,ui upperBoundSize,ui iter, ui prun1,ui prun2, int laneId);
__device__ __forceinline__ void selectUstar(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,double* sharedScore, int* sharedUstar,ui size, ui totalEdges, ui dmax, ui maxN2, ui startIndex, ui otherStartIndex,ui start,ui end, ui total, ui queryId, int laneId);

__device__ __forceinline__ void writeTasks(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,ui* sharedCounter, ui ustar, ui writePartition,ui writeOffset,ui totalTasksWrite, ui startIndex,ui start, ui end, ui otherStartIndex,ui pSize,ui offsetPsize, ui otherPsize, ui size, ui totalEdges,ui queryId,ui total, ui iter, ui warpId, ui laneId);
__device__ __forceinline__ void writeDomTasks(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B, ui* sharedCounter,ui ustar, ui writePartition, ui overFlow, ui totalDoms, ui writeOffset,ui domsWriteOffset,ui newTotalTasksWrite, ui startIndex,ui start, ui end, ui pSize, ui  offsetPsize, ui otherPsize, ui size, ui totalEdges,ui totalWrite,ui queryId, ui iter,ui warpId, ui laneId);
__device__ __forceinline__ void writeDomsBuffer(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B,ui ustar, ui overFlow, ui totalDoms, ui writeOffset, ui startIndex,ui start, ui size,ui totalWrite,ui queryId,ui bufferSize, ui otherPsize,ui iter,ui warpId,ui laneId); 
__device__ __forceinline__ void readFromBuffer(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B, double copyLimit,ui lastWritten, ui startIndex, ui start, ui total, ui iter, ui pSize,ui offsetPsize,ui otherPsize, ui size, ui totalEdges, ui bufferSize,ui warpId, ui laneId);
__device__ __forceinline__ void writeToBuffer(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B, ui writePartition, ui totalTasksWrite, ui writeOffset, ui ustar, ui startIndex, ui start,ui pSize,ui otherPsize, ui bufferSize, ui size, ui totalEdges,ui total,ui queryId,ui iter,ui warpId,ui laneId);



__device__ ui minn(ui a, ui b) {
  /**
   * Returns the minimum of two numbers.

   */
  return (a < b) ? a : b;
}

__device__ int findIndexKernel(ui * arr, ui start, ui end, ui target)

{
  /**
   * Performs a linear search to find the index of the target vertex in the task array.
   *
   * @param arr      Pointer to the array of unsigned integers where the search is performed.
   * @param start    Starting index of the array segment to search within.
   * @param end      Ending index of the array segment to search within.
   * @param target   Vertex to locate in the array.
   *
   * This function searches for the index of the specified `target` vertex within a specified range of the `arr` array,
   * from index `start` to `end`. It returns the index of the `target` if found, or an indicator of not found. A single
   * thread will perform the search.
   */

  int resultIndex = -1;
  for (ui index = start; index < end; index++) {
    if (arr[index] == target) {
      resultIndex = index;
      break;
    }
  }
  return resultIndex;
}

__device__ void warpBubbleSort(ui * arr, ui start, ui end, ui laneID, ui reverse) {

  /**
   * Performs a bubble sort within a single warp.
   *
   * @param arr      Pointer to the array of unsigned integers to be sorted.
   * @param start    Starting index of the partition of the array to be sorted.
   * @param end      Ending index of the partition of the array to be sorted.
   * @param laneID   The lane ID of the current thread within the warp.
   * @param reverse  A flag indicating the sort order: 0 for ascending and 1 for descending.
   *
   * This function sorts a segment of the array within a single warp using bubble sort. The sorting is performed
   * by comparing and potentially swapping values between different lanes of the warp. The `reverse` flag determines
   * the sort order. If `reverse` is 0, the function sorts in ascending order; if 1, it sorts in descending order.
   */
  ui size = end - start;

  for (ui i = 0; i < size; i++) {
    for (ui j = i + 1; j < size; j++) {
      ui val_i = __shfl_sync(0xFFFFFFFF, arr[start + laneID], i);
      ui val_j = __shfl_sync(0xFFFFFFFF, arr[start + laneID], j);
      if (!reverse) {
        if (laneID == i && val_i > val_j) {
          // Swap values within the warp
          arr[start + i] = val_j;
          arr[start + j] = val_i;
        }

      } else {
        if (laneID == i && val_i < val_j) {
          // Swap values within the warp
          arr[start + i] = val_j;
          arr[start + j] = val_i;
        }
      }

    }
  }
}

__device__ void selectionSort(ui * values, ui start, ui end, ui laneId) {

  /**
   * Performs selection sort within a single warp.
   *
   * @param values   Pointer to the array of unsigned integers to be sorted.
   * @param start    Starting index of the partition of the array to be sorted.
   * @param end      Ending index of the partition of the array to be sorted.
   * @param laneId   The lane ID of the current thread within the warp.
   *
   * This function sorts a segment of the array within a single warp using the selection sort algorithm.
   * The sorting is performed as follows:
   *
   * 1. Each thread identifies the maximum value in its portion of the array segment.
   * 2. A warp-wide reduction is used to determine the global maximum value and its index within the warp.
   * 3. The global maximum value is swapped with the value at the current index if they are different.
   * 4. The process is repeated for each index in the segment to achieve sorted order.
   *
   * The sorting is performed in-place and only within the specified segment of the array. The lane with `laneId` 0
   * performs the swap to ensure that the maximum value is placed in the correct position.
   */

  int n = end - start + 1;

  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    ui max_val = values[start + i];

    for (int j = laneId + i + 1; j < n; j += 32) {
      if (values[start + j] > max_val) {
        max_val = values[start + j];
        max_idx = j;
      }
    }

    // Reduce to find global max within the warp
    for (int offset = 16; offset > 0; offset /= 2) {
      ui other_val = __shfl_down_sync(0xffffffff, max_val, offset);
      int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
      if (other_val > max_val) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    // Lane 0 performs the swap
    if (laneId == 0 && max_idx != i) {
      ui temp_value = values[start + i];
      values[start + i] = values[start + max_idx];
      values[start + max_idx] = temp_value;
    }

    __syncwarp();
  }
}

__device__ void warpSelectionSort(double * keys, ui * values, ui start, ui end,
  ui laneId) {

  /**
   * Performs selection sort within a single warp, sorting an array of values based on their corresponding keys in decreasing order.
   *
   * @param keys     Pointer to the array of double values to be sorted.
   * @param values   Pointer to the array of unsigned integers associated with the keys.
   * @param start    Starting index of the partition of the arrays to be sorted.
   * @param end      Ending index of the partition of the arrays to be sorted.
   * @param laneId   The lane ID of the current thread within the warp.
   *
   * This function sorts a segment of the `keys` array and reorders the corresponding `values` array within a single warp using the selection sort algorithm.
   * The sorting process is as follows:
   *
   * 1. Each thread identifies the maximum key value within its portion of the array segment.
   * 2. A warp-wide reduction is used to determine the global maximum key value and its index within the warp.
   * 3. The maximum key value and its corresponding value are swapped with the value at the current index if they are different.
   * 4. The process is repeated for each index in the segment to achieve sorted order.
   *
   * The sorting is performed in-place and only within the specified segment of the `keys` and `values` arrays. The lane with `laneId` 0 performs the swap to
   * ensure that the maximum value is placed in the correct position.
   */

  int n = end - start + 1;

  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    double max_val = keys[start + i];

    for (int j = laneId + i + 1; j < n; j += 32) {
      if (keys[start + j] > max_val) {
        max_val = keys[start + j];
        max_idx = j;
      }
    }

    // Reduce to find global max within the warp
    for (int offset = 16; offset > 0; offset /= 2) {
      double other_val = __shfl_down_sync(0xffffffff, max_val, offset);
      int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
      if (other_val > max_val) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    // Lane 0 performs the swap
    if (laneId == 0 && max_idx != i) {
      double temp_key = keys[start + i];
      keys[start + i] = keys[start + max_idx];
      keys[start + max_idx] = temp_key;

      ui temp_value = values[start + i];
      values[start + i] = values[start + max_idx];
      values[start + max_idx] = temp_value;
    }

    __syncwarp();
  }
}

__global__ void initialReductionRules(deviceGraphGenPointers G, deviceGraphPointers G_,
  deviceInterPointers P, ui size,
  ui upperBoundDistance, ui pSize, ui queryId) {

  extern __shared__ ui shared_memory1[];

  ui * local_counter = shared_memory1;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui start = minn(warpId * pSize, size);
  ui end = minn((warpId + 1) * pSize, size);

  ui writeOffset = start;
  ui total = end - start;

  if (laneId == 0) {
    local_counter[threadIdx.x / warpSize] = 0;
  }
  __syncwarp();

  for (ui i = laneId; i < total; i += warpSize) {
    ui vertex = start + i;
    if ((G.core[vertex] > G_.lowerBoundDegree[queryId]) &&
      (G_.distance[(queryId * size) + vertex] <= upperBoundDistance)) {
      ui loc = atomicAdd( & local_counter[threadIdx.x / warpSize], 1);
      P.initialTaskList[loc + writeOffset] = vertex;

    } else {
      G_.degree[(queryId * size) + vertex] = 0;
    }
  }

  __syncwarp();
  if (laneId == 0) {
    P.entries[warpId] = local_counter[threadIdx.x / warpSize];

    atomicAdd(P.globalCounter, local_counter[threadIdx.x / warpSize]);
  }
}

__global__ void CompressTask(deviceGraphGenPointers G, deviceGraphPointers G_, deviceInterPointers P,
  deviceTaskPointers T, ui pSize, ui queryVertex, ui queryId, ui size, ui taskPSize, ui totalWarps, ui factor) {

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui start = warpId * pSize;
  int temp = warpId - 1;
  ui total = P.entries[warpId];
  ui offsetPsize = taskPSize / factor;
  ui otherPsize = * T.limitTasks;
  // add some of mechanism
  //ui minTasks = UINT_MAX;
  ui writeWarp = T.sortedIndex[0];
  //ui spaceLeft;
  ui numTasks;

  numTasks = T.numTasks[writeWarp];
  ui writeOffset = taskPSize * writeWarp + T.taskOffset[offsetPsize * writeWarp + numTasks];

  if (idx == 0) {
    T.size[otherPsize * (writeWarp) + numTasks] = 1;
    T.queryIndicator[otherPsize * (writeWarp) + numTasks] = queryId;
    T.taskOffset[offsetPsize * (writeWarp) + numTasks + 1] = T.taskOffset[offsetPsize * (writeWarp) + numTasks] + * P.globalCounter;
    T.ustar[otherPsize * (writeWarp) + numTasks] = -1;
    T.numTasks[writeWarp]++;

  }
  __syncwarp();


  for (ui i = laneId; i < total; i += warpSize) {
    while (temp >= 0) {
      writeOffset += P.entries[temp];
      temp--;
    }
    int vertex = P.initialTaskList[start + i];
    T.taskList[writeOffset + i] = vertex;
    T.statusList[writeOffset + i] = (vertex == queryVertex) ? 1 : 0;

    ui totalNeigh = 0;
    ui nStart = G.offset[vertex];
    ui nEnd = G.offset[vertex + 1];
    ui degInR = 0;
    ui degInc = 0;
    for (ui k = nStart; k < nEnd; k++) {
      if ((G.neighbors[k] != queryVertex) && (G_.degree[(queryId * size) + G.neighbors[k]] != 0)) {
        degInR++;
      }
      if (G.neighbors[k] == queryVertex) {
        degInc++;
      }
      if (G_.degree[(queryId * size) + G.neighbors[k]] != 0) {
        totalNeigh++;
      }
    }

    G_.newOffset[(queryId * (size + 1)) + vertex + 1] = totalNeigh;
    T.degreeInR[writeOffset + i] = degInR;
    T.degreeInC[writeOffset + i] = degInc;
  }

}



__global__ void NeighborUpdate(deviceGraphGenPointers G, deviceGraphPointers G_,deviceTaskPointers T, ui totalWarps, ui queryId, ui size, ui totalEdges, ui pSize, ui factor) {

  extern __shared__ ui sharedMem[];
  ui * localCounter = sharedMem;

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < size; i += totalWarps) {
    if (laneId == 0) {
      localCounter[threadIdx.x / warpSize] = 0;
    }
    __syncwarp();
    ui degree = G_.degree[(queryId * size) + i];

    if (degree > 0) {
      ui writeOffset = G_.newOffset[(queryId * (size + 1)) + i];

      ui start = G.offset[i];
      ui end = G.offset[i + 1];
      ui total = end - start;
      ui neighbor;
      ui neighborDegree;
      for (ui j = laneId; j < total; j += warpSize) {
        neighbor = G.neighbors[start + j];
        neighborDegree = G_.degree[(queryId * size) + neighbor];
        if (neighborDegree > 0) {
          ui loc = atomicAdd( & localCounter[threadIdx.x / warpSize], 1);
          G_.newNeighbors[(2 * totalEdges * queryId) + writeOffset + loc] = neighbor;

        }

      }
      __syncwarp();
      if (laneId == 0) {
        localCounter[threadIdx.x / warpSize] = 0;
      }

    }
    __syncwarp();
  }

  __syncwarp();
  if (laneId == 0) {
      ui offsetPsize = pSize / factor;
      ui totalTasks = T.numTasks[warpId];
      ui offsetStartIndex = warpId * offsetPsize;
      T.sortedIndex[warpId] = T.taskOffset[offsetStartIndex + totalTasks];
    }

}


__global__ void ProcessTask(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,
  ui pSize, ui factor, ui maxN2, ui size, ui totalEdges, ui dmax, ui limitQueries, ui red1, ui red2, ui red3 , ui prun1,ui prun2) {
  extern __shared__ char shared_memory[];
  ui sizeOffset = 0;

  ui * sharedUBDegree = (ui * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  ui * sharedDegree = (ui * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int * sharedUstar = (int * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(int);

  ui * sharedCounterC = (ui * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  ui * sharedCounterR = (ui * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  ui * sharedC_ = (ui * )(shared_memory + sizeOffset);
  sizeOffset += maxN2 * WARPS_EACH_BLK * sizeof(ui);
  sizeOffset = (sizeOffset + alignof(double) - 1) & ~(alignof(double) - 1);

  double * sharedScore = (double * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(double);

  // minimum degree initialized to zero.

  // Connection score used to calculate the ustar.
  ui offsetPsize = pSize / factor;
  ui otherPsize = * T.limitTasks;
  

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui startIndex = warpId * pSize;
  ui offsetStartIndex = warpId * offsetPsize;
  ui otherStartIndex = warpId * otherPsize;

  ui totalTasks = T.numTasks[warpId];

  if (laneId == 0) {

    sharedUBDegree[threadIdx.x / warpSize] = UINT_MAX;
    sharedDegree[threadIdx.x / warpSize] = UINT_MAX;
    sharedUstar[threadIdx.x / warpSize] = -1;
    sharedCounterC[threadIdx.x / warpSize] = 0;
    sharedCounterR[threadIdx.x / warpSize] = 0;
    sharedScore[threadIdx.x / warpSize] = 0;

  }

  __syncwarp();
  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();

    ui start = T.taskOffset[offsetStartIndex + iter];
    ui end = T.taskOffset[offsetStartIndex + iter + 1];
    ui total = end - start;
    ui queryId = T.queryIndicator[otherStartIndex + iter];
    ui lowerBoundSize = 0;
    ui upperBoundSize = 0;

    if (queryId != UINT_MAX) {
      lowerBoundSize = G_.lowerBoundSize[queryId];
      upperBoundSize = G_.upperBoundSize[queryId];

    }

  if ((T.ustar[otherStartIndex + iter] != INT_MAX) && (queryId != UINT_MAX) && (T.size[otherStartIndex + iter] <= upperBoundSize) && (queryId != limitQueries)) {
      
      ui currentSize = T.size[otherStartIndex + iter];
      if (lowerBoundSize <= currentSize ) {
        calculateMinimumDegree(G, G_, T, sharedDegree, startIndex, start, total, laneId);
        if (laneId == 0) {
            if (sharedDegree[threadIdx.x / warpSize] != UINT_MAX) {
            atomicMax(&G_.lowerBoundDegree[queryId], sharedDegree[threadIdx.x / warpSize]);
            sharedDegree[threadIdx.x / warpSize] = UINT_MAX;
            }
        }
        __syncwarp();
      }

      if(red3){
        reductionRule3(G, G_, T, size, totalEdges, startIndex, otherStartIndex, start, end,total , queryId, iter, laneId);

        if ((lowerBoundSize <= T.size[otherStartIndex + iter]) && (T.size[otherStartIndex + iter] <= upperBoundSize) && (currentSize!=T.size[otherStartIndex + iter])) {
          calculateMinimumDegree(G, G_, T, sharedDegree, startIndex, start, total, laneId);
          if (laneId == 0) {
              if (sharedDegree[threadIdx.x / warpSize] != UINT_MAX) {
              atomicMax(&G_.lowerBoundDegree[queryId], sharedDegree[threadIdx.x / warpSize]);
              sharedDegree[threadIdx.x / warpSize] = UINT_MAX;
              }
          }
          __syncwarp();
        }
      }
      
      if (T.size[otherStartIndex + iter] < upperBoundSize) {

        reductionRule1and2(G, G_, T,size, totalEdges, startIndex, otherStartIndex, start, end,total , queryId, upperBoundSize,iter,red1, red2, laneId);
        calculateUpperBoundDegree(G, G_, T, sharedUBDegree, sharedC_, sharedCounterC, sharedCounterR, maxN2, startIndex, otherStartIndex, start, total,  upperBoundSize, iter,prun1,prun2, laneId);
        ui upperBoundDegreeLimit;
        if(prun1 && prun2 ){
          upperBoundDegreeLimit = minn(sharedC_[(threadIdx.x / warpSize) * maxN2], sharedUBDegree[threadIdx.x / warpSize]);
        }else{
          if(prun1){
            upperBoundDegreeLimit = sharedUBDegree[threadIdx.x / warpSize];
          }
          if(prun2){
            upperBoundDegreeLimit = sharedC_[(threadIdx.x / warpSize) * maxN2];

          }
        }
        if (( upperBoundDegreeLimit > G_.lowerBoundDegree[queryId]) && (upperBoundDegreeLimit != UINT_MAX)){
        selectUstar(G, G_, T,sharedScore, sharedUstar,size, totalEdges, dmax,maxN2, startIndex, otherStartIndex, start, end,total, queryId, laneId);
        }
        
        if (laneId == 0) {
            int writeOffset = warpId * otherPsize;
            if ((sharedScore[threadIdx.x / warpSize] > 0) &&
                (upperBoundDegreeLimit != UINT_MAX) &&
                (total >= lowerBoundSize) && (upperBoundDegreeLimit > G_.lowerBoundDegree[queryId])) {
            T.ustar[writeOffset + iter] = sharedUstar[threadIdx.x / warpSize];
            } else {
            T.ustar[writeOffset + iter] = INT_MAX;
            }
        }
        __syncwarp();
        

      }
      else {
        if (laneId == 0) {
          int writeOffset = warpId * otherPsize;
          T.ustar[writeOffset + iter] = INT_MAX;
        }
        __syncwarp();
      }

      if (laneId == 0) {
        sharedUBDegree[threadIdx.x / warpSize] = UINT_MAX;
        sharedScore[threadIdx.x / warpSize] = 0;
        sharedUstar[threadIdx.x / warpSize] = -1;
        sharedDegree[threadIdx.x / warpSize] = UINT_MAX;
        sharedCounterC[threadIdx.x / warpSize] = 0;
        sharedCounterR[threadIdx.x / warpSize] = 0;
      }
    }
  }

  __syncwarp();

  for (int iter = totalTasks - 1; iter >= 0; iter--) {

    if (laneId == 0) {
    bool shouldDecrement = (T.queryIndicator[warpId * otherPsize + iter] == UINT_MAX) ||
      (T.ustar[warpId * otherPsize + iter] == INT_MAX);

    if (shouldDecrement) {
      T.numTasks[warpId]--;
      T.ustar[warpId * otherPsize + iter] = -1;
      T.queryIndicator[warpId * otherPsize + iter] = limitQueries;
      
    }
      
    }
  }
}


__global__ void FindDoms(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, ui pSize, ui factor, ui size, ui totalEdges, ui dmax,ui limitQueries) {
  /**
   * This kernel iterates through tasks assigned to each warp, and finds the verticies that are dominated by the ustar of that task.
   * Vertex v' is dominated by ustar if all of its neighbors are either neighbors of ustar or ustar itself.
   * Calculates the connection score of all verticies in dominating set and stores the dominating set decreasing order of connection score.
   * only a limited number are retained, controlled by `limitDoms`.
   *
   * @param G         Device graph pointers containing graph data.
   * @param T         Device task pointers containing task data.
   * @param pSize     Size of each partition assigned to a warp.
   * @param dmax      Maximum degree of a graph.
   * @param limitDoms Maximum number of dominating vertices to store per task.
   */
  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui * sharedCounter = (ui * )(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  int resultIndex;

  ui offsetPsize = pSize / factor;
  ui otherPsize = * T.limitTasks;

  ui startIndex = warpId * pSize;
  ui offsetStartIndex = warpId * offsetPsize;
  ui otherStartIndex = warpId * otherPsize;

  ui totalTasks = T.numTasks[warpId];
  if (laneId == 0) {
    sharedCounter[threadIdx.x / 32] = 0;
  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[offsetStartIndex + iter];
    ui end = T.taskOffset[offsetStartIndex + iter + 1];
    ui total = end - start;
    ui writeOffset = startIndex + start;
    ui queryId = T.queryIndicator[otherStartIndex + iter];
    ui limitDoms = 0;
    if (queryId != UINT_MAX)
      limitDoms = G_.limitDoms[queryId];

    if ((T.ustar[otherStartIndex + iter] != INT_MAX) &&
      (T.ustar[otherStartIndex + iter] != -1) && (queryId != UINT_MAX) && (limitDoms > 0) && (queryId != limitQueries)) {
      ui ustar = T.taskList[T.ustar[otherStartIndex + iter]];
      for (ui i = laneId; i < total; i += 32) {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];
        ui degC = T.degreeInC[ind];

        ui startNeighbor = G_.newOffset[(queryId * (size + 1)) + vertex];
        ui endNeighbor = G_.newOffset[(queryId * (size + 1)) + vertex + 1];
        bool is_doms = false;
        double score = 0;

        if ((status == 0) && (vertex != ustar) && (degC != 0)) {
          is_doms = true;
          ui neighbor;
          for (int j = startNeighbor; j < endNeighbor; j++) {
            neighbor = G_.newNeighbors[(2 * totalEdges * queryId) + j];

            bool found = false;
            if (neighbor != ustar) {
              for (ui k = G_.newOffset[(queryId * (size + 1)) + ustar]; k < G_.newOffset[(queryId * (size + 1)) + ustar + 1]; k++) {
                if (neighbor == G_.newNeighbors[(2 * totalEdges * queryId) + k]) {
                  found = true;
                  break;
                }else if (neighbor > G_.newNeighbors[(2 * totalEdges * queryId) + k]){
                  break;
                }
              }
            } else {
              found = true;
            }
            if (!found) {
              is_doms = false;
              break;
            }

            resultIndex = findIndexKernel(T.taskList, startIndex + start,
              startIndex + end, neighbor);
            if (resultIndex != -1) {
              if (T.statusList[resultIndex] == 1) {
                if (T.degreeInC[resultIndex] != 0) {
                  score += (double) 1 / T.degreeInC[resultIndex];
                }
              }
            }

          }
          score += (double) T.degreeInR[ind] / dmax;
        }

        if (is_doms) {
          ui loc = atomicAdd( & sharedCounter[threadIdx.x / 32], 1);
          T.doms[writeOffset + loc] = vertex;
          T.cons[writeOffset + loc] = score;

        }
      }
    
    __syncwarp();

    if (sharedCounter[threadIdx.x / 32] > 1) {
      warpSelectionSort(T.cons, T.doms, startIndex + start,
        startIndex + start + sharedCounter[threadIdx.x / 32],
        laneId);
    }
    __syncwarp();
    if (laneId == 0) {
      T.doms[startIndex + end - 1] =
        (sharedCounter[threadIdx.x / 32] > limitDoms) ?
        limitDoms :
        sharedCounter[threadIdx.x / 32];
      sharedCounter[threadIdx.x / 32] = 0;
    }
    }
  }
}

 
  __global__ void Expand(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,
  deviceBufferPointers B, ui pSize, ui factor, double copyLimit,
  ui bufferSize, ui lastWritten, ui readLimit, ui size, ui totalEdges, ui dmax, ui limitQueries) {

  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui * sharedCounter = (ui * )(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui offsetPsize = pSize / factor;
  ui otherPsize = *T.limitTasks;

  ui startIndex = warpId * pSize;
  ui offsetStartIndex = warpId * offsetPsize;
  ui otherStartIndex = warpId * otherPsize;

  ui totalTasks = T.numTasks[warpId];

  if (laneId == 0) {
    sharedCounter[threadIdx.x / warpSize] = 0;
  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[offsetStartIndex + iter];
    ui end = T.taskOffset[offsetStartIndex + iter + 1];
    ui total = end - start;
    ui queryId = T.queryIndicator[otherStartIndex + iter];

    ui upperBoundSize = 0;
    if (queryId != UINT_MAX) {
      upperBoundSize = G_.upperBoundSize[queryId];
    }

    if ((T.ustar[otherStartIndex + iter] != INT_MAX) &&
      (T.ustar[otherStartIndex + iter] != -1) &&
      (T.size[otherStartIndex + iter] < upperBoundSize) && (queryId != UINT_MAX) &&  (queryId != limitQueries)) {
      ui writePartition = T.sortedIndex[T.mapping[warpId]];
      ui totalTasksWrite = T.numTasks[writePartition];
      ui writeOffset = ((writePartition) * pSize) +
        T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite];
      ui ustar = T.taskList[T.ustar[otherStartIndex + iter]];
      ui totalWrite;

      if (((T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite] + total) < (pSize - 1)) && (totalTasksWrite < * T.limitTasks)) {
        writeTasks( G,  G_,  T,sharedCounter,  ustar,  writePartition, writeOffset,totalTasksWrite,  startIndex,start, end,  otherStartIndex, pSize,offsetPsize, otherPsize,  size,  totalEdges, queryId, total,  iter, warpId, laneId);
        
        ui totalDoms = T.doms[startIndex + end - 1];
        ui newTotalTasksWrite = totalTasksWrite + 1;
        if (totalDoms == 0) {
          if (laneId == 0) {
            sharedCounter[threadIdx.x / warpSize] = 0;
          }
        }
        if (totalDoms != 0) {
          totalWrite = sharedCounter[threadIdx.x / 32];

          if (laneId == 0) {
            sharedCounter[threadIdx.x / warpSize] = 0;
          }
          __syncwarp();
          ui domsWriteOffset = writeOffset + totalWrite;
          totalWrite += 1;
          int leftSpace;
          int overFlow;
          int leftTasks = * T.limitTasks - newTotalTasksWrite;
          leftSpace = pSize - T.taskOffset[(writePartition) * offsetPsize + newTotalTasksWrite] -1 ;
          leftSpace = max(leftSpace, 0); // not required as it will be alway positive
          overFlow = leftSpace / totalWrite;
          overFlow = min(leftTasks, overFlow);
          overFlow = min(overFlow, totalDoms);
          writeDomTasks( G,  G_,  T,  B, sharedCounter, ustar,  writePartition,  overFlow,  totalDoms,  writeOffset, domsWriteOffset, newTotalTasksWrite,  startIndex, start,  end,  pSize,   offsetPsize,  otherPsize,  size,  totalEdges, totalWrite, queryId,  iter, warpId,  laneId);


          writeDomsBuffer( G,  G_,  T, B,ustar, overFlow, totalDoms, writeOffset, startIndex,start, size,totalWrite,queryId,bufferSize, otherPsize,iter,warpId, laneId);

          
                 } else {
          if (( * B.numTask > 0)) {

            readFromBuffer( G,  G_,  T,  B,  copyLimit, lastWritten, startIndex,  start,  total,  iter,  pSize, offsetPsize, otherPsize,  size,  totalEdges,  bufferSize, warpId, laneId);
            
        }
        }
      } else {
            writeToBuffer(G,G_,T,B,  writePartition,  totalTasksWrite,  writeOffset,ustar, startIndex, start,  pSize, otherPsize, bufferSize,size,  totalEdges, total,queryId,iter,warpId, laneId);

        
      }
    }
  }

  __syncwarp();
  if ((T.numTasks[warpId] == 0) && ( * B.numTask > 0)) {
    for (ui iterRead = 0; iterRead < readLimit; iterRead++) {
      __syncwarp();

      ui writePartition = T.sortedIndex[T.mapping[warpId]];
      ui totalTasksWrite = T.numTasks[writePartition];
      ui writeOffset = ((writePartition) * pSize) +
        T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite];
      if ((T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite] < (ui)(pSize * copyLimit)) && (totalTasksWrite < * T.limitTasks)) {
        ui numRead = UINT_MAX;
        ui readFlag = 0;
        __syncwarp();
        if ((laneId == 0)) {
          while (true) {
            if (atomicCAS(B.readMutex, 0, 1) == 0) {
              if ( * B.numReadTasks < lastWritten) {
                numRead = atomicAdd(B.numReadTasks, 1);
                readFlag = 1;
              }

              atomicExch(B.readMutex, 0);
              break;
            }
          }
        }
        __syncwarp();
        numRead = __shfl_sync(0xffffffff, numRead, 0);
        readFlag = __shfl_sync(0xffffffff, readFlag, 0);

        if (readFlag) {

          ui readStart = B.taskOffset[numRead];
          ui readEnd = B.taskOffset[numRead + 1];
          ui totalRead = readEnd - readStart;
          ui readQueryId = B.queryIndicator[numRead];

          for (ui i = laneId; i < totalRead; i += warpSize) {
            ui ind = readStart + i;
            ui vertex = B.taskList[ind];
            ui status = B.statusList[ind];

            T.taskList[writeOffset + i] = vertex;
            T.statusList[writeOffset + i] = status;
            int keyTask;
            ui degC = 0;
            ui degR = 0;
            if (status != 2) {
              for (ui k = G_.newOffset[(readQueryId * (size + 1)) + vertex]; k < G_.newOffset[(readQueryId * (size + 1)) + vertex + 1]; k++) {
                keyTask =
                  findIndexKernel(B.taskList, readStart, readEnd, G_.newNeighbors[(2 * totalEdges * readQueryId) + k]);

                if (keyTask != -1) {
                  if (B.statusList[keyTask] == 1) {
                    degC++;

                  } else if (B.statusList[keyTask] == 0) {
                    degR++;
                  }
                }

              }

            }

            T.degreeInC[writeOffset + i] = degC;
            T.degreeInR[writeOffset + i] = degR;
          }
          __syncwarp();

          if (laneId == 0) {
            T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite + 1] =
              T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite] + totalRead;
            T.size[(writePartition) * otherPsize + totalTasksWrite] = B.size[numRead];
            T.numTasks[writePartition]++;
            T.queryIndicator[(writePartition) * otherPsize + totalTasksWrite] = readQueryId;
            T.ustar[(writePartition) * otherPsize + totalTasksWrite] = -1;

            G_.numRead[readQueryId]++;
            G_.flag[readQueryId] = 1;

          }
        }
      }
    }
  }
}



__global__ void RemoveCompletedTasks(deviceGraphPointers G_, deviceTaskPointers T, ui pSize, ui factor) {


  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  ui otherPsize = * T.limitTasks;

  ui otherStartIndex = warpId * otherPsize;
  ui totalTasks = T.numTasks[warpId];
  __syncwarp();

  for (ui iter = laneId; iter < totalTasks; iter += warpSize) {
    __syncwarp();
    ui queryId = T.queryIndicator[otherStartIndex + iter];
    if (queryId != UINT_MAX) {
      ui stopflag = G_.flag[queryId];
      if (stopflag == 0) {
        T.queryIndicator[otherStartIndex + iter] = UINT_MAX;
      }
    }
  }
  if (laneId == 0) {
    ui offsetPsize = pSize / factor;

    ui offsetStartIndex = warpId * offsetPsize;
    T.sortedIndex[warpId] = T.taskOffset[offsetStartIndex + totalTasks];
  }
}



__device__ __forceinline__ void calculateMinimumDegree(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,ui* sharedDegree, ui startIndex, ui start, ui total, int laneId) {
  ui maskGen = total;
  for (ui i = laneId; i < total; i += warpSize) {
    ui ind = startIndex + start + i;
    ui currentMinDegree = UINT_MAX;

    unsigned int mask;
    ui cond;

    if (maskGen > warpSize) {
      mask = 0xFFFFFFFF;
      maskGen -= warpSize;
      cond = warpSize;
    } else {
      mask = (1u << maskGen) - 1;
      cond = maskGen;
    }
    if (T.statusList[ind]== 1) {
      currentMinDegree = T.degreeInC[ind];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      ui temp2 = __shfl_down_sync(mask, currentMinDegree, offset);
      if (laneId + offset < cond) {
        currentMinDegree = minn(currentMinDegree, temp2);
      }
    }

    currentMinDegree = __shfl_sync(mask, currentMinDegree, 0);

    if (i % warpSize == 0) {
      if (currentMinDegree < sharedDegree[threadIdx.x / warpSize]) {
        sharedDegree[threadIdx.x / warpSize] = currentMinDegree;
      }
    }
  }
  __syncwarp();
 
}

__device__ __forceinline__ void reductionRule3(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, ui size, ui totalEdges, ui startIndex, ui otherStartIndex, ui start,ui end,  ui total , ui queryId, ui iter, int laneId) {
  for (ui i = laneId; i < total; i += warpSize) {
    ui ind = startIndex + start + i;
    ui vertex = T.taskList[ind];
    ui kl = G_.lowerBoundDegree[queryId];


    if ((T.statusList[ind] == 1) && ((T.degreeInC[ind] + T.degreeInR[ind]) == (kl + 1))) {
      for (int j = G_.newOffset[(queryId * (size + 1)) + vertex]; j < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; j++) {
        int resultIndex = findIndexKernel(T.taskList, startIndex + start,
          startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + j]);
        if (resultIndex != -1) {
          if (atomicCAS(&T.statusList[resultIndex], 0, 1) == 0) {
            atomicAdd(&T.size[otherStartIndex + iter], 1);

            ui neig = T.taskList[resultIndex];

            for (int k = G_.newOffset[(queryId * (size + 1)) + neig]; k < G_.newOffset[(queryId * (size + 1)) + neig + 1]; k++) {
              int resInd = findIndexKernel(T.taskList, startIndex + start,
                startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + k]);
              if (resInd != -1) {
                atomicAdd(&T.degreeInC[resInd], 1);
                if (T.degreeInR[resInd] != 0) {
                  atomicSub(&T.degreeInR[resInd], 1);
                  if (T.degreeInR[resInd] == 4294967295) {
                    T.degreeInR[resInd] = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  __syncwarp();
}

__device__ __forceinline__ void reductionRule1and2(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, ui size, ui totalEdges, ui startIndex, ui otherStartIndex, ui start,ui end, ui total , ui queryId, ui upperBoundSize,ui iter, ui red1, ui red2, int laneId ) {
  for (ui i = laneId; i < total; i += warpSize) {
    ui ind = startIndex + start + i;
    ui vertex = T.taskList[ind];
    ui ubD = upperBoundSize - 1;
    ui kl = G_.lowerBoundDegree[queryId];

    for (ui d = 1; d <= upperBoundSize; d++) {
      if (d == 1 || d == 2) {
        if (kl + d > upperBoundSize) {
          ubD = d - 1;
          break;
        }
      } else {
        ui min_n = kl + d + 1 + (d / 3) * (kl - 2);
        if (upperBoundSize < min_n) {
          ubD = d - 1;
          break;
        }
      }
    }

    bool check;
    if(red1 && red2 ){
      check = (minn((T.degreeInR[ind] + T.degreeInC[ind]), (T.degreeInC[ind] + upperBoundSize - T.size[otherStartIndex + iter] - 1)) <= G_.lowerBoundDegree[queryId]) || (ubD < G_.distance[(queryId * size) + vertex]);
    }else{
      if(red1){
        check = (minn((T.degreeInR[ind] + T.degreeInC[ind]), (T.degreeInC[ind] + upperBoundSize - T.size[otherStartIndex + iter] - 1)) <= G_.lowerBoundDegree[queryId]);
      }
      if(red2){
        check = (ubD < G_.distance[(queryId * size) + vertex]);
      }

    }
    


    if (T.statusList[ind] == 0) {
      if ( check ) {
        T.statusList[ind] = 2;
        T.degreeInR[ind] = 0;
        T.degreeInC[ind] = 0;
        for (int j = G_.newOffset[(queryId * (size + 1)) + vertex]; j < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; j++) {
          int resultIndex = findIndexKernel(T.taskList, startIndex + start,
            startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + j]);
          if (resultIndex != -1) {
            if (T.degreeInR[resultIndex] != 0) {
              atomicSub(&T.degreeInR[resultIndex], 1);
              if (T.degreeInR[resultIndex] == 4294967295) {
                T.degreeInR[resultIndex] = 0;
              }
            }
          }
        }
      }
    }
  }
  __syncwarp();
}

__device__ __forceinline__ void calculateUpperBoundDegree(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,ui* sharedUBDegree, ui* sharedC_, ui* sharedCounterC, ui* sharedCounterR, ui maxN2, ui startIndex, ui otherStartIndex, ui start, ui total, ui upperBoundSize,ui iter,ui prun1,ui prun2, int laneId) {
  ui maskGen = total;
  for (ui i = laneId; i < total; i += warpSize) {
    ui ind = startIndex + start + i;
    ui status = T.statusList[ind];
    ui degInC = T.degreeInC[ind];
    ui degInR = T.degreeInR[ind];
    ui degreeBasedUpperBound = UINT_MAX;
    if(prun1){
      

      unsigned int mask;
      ui cond;

      if (maskGen > warpSize) {
        mask = 0xFFFFFFFF;
        maskGen -= warpSize;
        cond = warpSize;
      } else {
        mask = (1u << maskGen) - 1;
        cond = maskGen;
      }

      if (status == 1) {
        ui oneside = degInC + upperBoundSize - T.size[otherStartIndex + iter];
        degreeBasedUpperBound = minn(oneside, degInR + degInC);
      }

      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        ui temp3 = __shfl_down_sync(mask, degreeBasedUpperBound, offset);
        if (laneId + offset < cond) {
          degreeBasedUpperBound = min(degreeBasedUpperBound, temp3);
        }
      }
      degreeBasedUpperBound = __shfl_sync(mask, degreeBasedUpperBound, 0);
    }
    if(prun2){
      if (status == 0) {
        ui locR = atomicAdd(&sharedCounterR[threadIdx.x / warpSize], 1);
        T.doms[startIndex + start + locR] = degInC;
      }
      if (status == 1) {
        ui locC = atomicAdd(&sharedCounterC[threadIdx.x / warpSize], 1);
        sharedC_[(threadIdx.x / warpSize) * maxN2 + locC] = degInC;
      }
    }
    if(prun1){
      if (i % warpSize == 0) {
        if (degreeBasedUpperBound < sharedUBDegree[threadIdx.x / warpSize]) {
          sharedUBDegree[threadIdx.x / warpSize] = degreeBasedUpperBound;
        }
      }
    }
  }
  __syncwarp();
  if(prun2){
    selectionSort(T.doms, startIndex + start, startIndex + start + sharedCounterR[threadIdx.x / warpSize], laneId);
    __syncwarp();
    warpBubbleSort(sharedC_, (threadIdx.x / warpSize) * maxN2, (threadIdx.x / warpSize) * maxN2 + sharedCounterC[threadIdx.x / warpSize], laneId, 0);
    ui currentSize = T.size[otherStartIndex + iter];
    ui totalIter_ = minn((upperBoundSize - currentSize), sharedCounterR[threadIdx.x / warpSize]);
    for (ui iter_ = 0; iter_ < totalIter_; iter_++) {
      __syncwarp();
      ui value = T.doms[startIndex + start + iter_];
      for (ui i = laneId; i < minn(value, sharedCounterC[threadIdx.x / warpSize]); i += 32) {
        sharedC_[(threadIdx.x / warpSize) * maxN2 + i]++;
      }
      __syncwarp();
      warpBubbleSort(sharedC_, (threadIdx.x / warpSize) * maxN2, (threadIdx.x / warpSize) * maxN2 + sharedCounterC[threadIdx.x / warpSize], laneId, 0);
    }
    __syncwarp();
  }
}

__device__ __forceinline__ void selectUstar(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,double* sharedScore, int* sharedUstar,ui size, ui totalEdges, ui dmax, ui maxN2, ui startIndex, ui otherStartIndex,ui start, ui end, ui total, ui queryId, int laneId) {
    ui maskGen = total;
    for (ui i = laneId; i < total; i += warpSize) {
      int index = startIndex + start + i;
      ui ind = startIndex + start + i;
      ui vertex = T.taskList[ind];

      double score = 0;
      int ustar = -1;

      unsigned int mask;
      ui cond;

      if (T.statusList[ind] == 0) {
        for (int j = G_.newOffset[(queryId * (size + 1)) + vertex]; j <  G_.newOffset[(queryId * (size + 1)) + vertex + 1]; j++) {
          int resultIndex = findIndexKernel(T.taskList, startIndex + start,
            startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + j]);
          if (resultIndex != -1) {
            if (T.statusList[resultIndex] == 1) {
              if (T.degreeInC[resultIndex] != 0) {
                score += (double) 1 / T.degreeInC[resultIndex];
              } else {
                score += 1;
              }
            }
          }
        }
      }
      if (score > 0) {
        score += (double) T.degreeInR[ind] / dmax;
      }

      

      if (maskGen > warpSize) {
        mask = 0xFFFFFFFF;
        maskGen -= warpSize;
        cond = warpSize;
      } else {
        mask = (1u << maskGen) - 1;
        cond = maskGen;
      }

      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        double temp = __shfl_down_sync(mask, score, offset);
        int otherId = __shfl_down_sync(mask, index, offset);

        if (temp > score && temp > 0 && (laneId + offset < cond)) {
          score = temp;
          index = otherId;
        }
      }

      ustar = __shfl_sync(mask, index, 0);
      score = __shfl_sync(mask, score, 0);

      if (i % warpSize == 0) {
        if (score > sharedScore[threadIdx.x / warpSize]) {
          sharedScore[threadIdx.x / warpSize] = score;
          sharedUstar[threadIdx.x / warpSize] = ustar;
        }
      }
    }
  __syncwarp();
}
 

__device__ __forceinline__ void writeTasks(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,ui* sharedCounter, ui ustar, ui writePartition,ui writeOffset,ui totalTasksWrite, ui startIndex,ui start, ui end, ui otherStartIndex,ui pSize,ui offsetPsize, ui otherPsize, ui size, ui totalEdges,ui queryId,ui total, ui iter, ui warpId, ui laneId) {
        ui maskGen = total;
        for (ui i = laneId; i < total; i += warpSize) {
          unsigned int mask;

          if (maskGen > warpSize) {
            mask = 0xFFFFFFFF;
            maskGen -= warpSize;

          } else {
            mask = (1u << maskGen) - 1;
          }
          __syncwarp(mask);
          ui ind = startIndex + start + i;
          ui vertex = T.taskList[ind];
          ui status = T.statusList[ind];

          ui degInR;
          ui degInC;

          if ((vertex != ustar) && (status != 2)) {
            ui loc = atomicAdd( & sharedCounter[threadIdx.x / warpSize], 1);
            T.taskList[writeOffset + loc] = vertex;
            T.statusList[writeOffset + loc] = status;
            degInR = T.degreeInR[ind];
            degInC = T.degreeInC[ind];

            for (ui k = G_.newOffset[(queryId * (size + 1)) + vertex]; k < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; k++) {
              if (G_.newNeighbors[(2 * totalEdges * queryId) + k] == ustar) {
                T.degreeInC[ind]++;

                if (degInR != 0) {
                  degInR--;
                }
              }
            }

            T.degreeInR[writeOffset + loc] = degInR;
            T.degreeInC[writeOffset + loc] = degInC;
            T.degreeInR[ind] = degInR;
            if (T.doms[startIndex + end - 1] != 0) {
              int key = findIndexKernel(
                T.doms, startIndex + start,
                startIndex + start + T.doms[startIndex + end - 1], vertex);

              if (key != -1) {
                T.statusList[ind] = 2;
                T.statusList[writeOffset + loc] = 2;
                T.degreeInC[ind] = 0;
                T.degreeInR[ind] = 0;
                T.degreeInC[writeOffset + loc] = 0;
                T.degreeInR[writeOffset + loc] = 0;
              }
             
                ui tDoms =  T.doms[startIndex + end - 1];
                for (int j = G_.newOffset[(queryId * (size + 1)) + vertex]; j < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; j++) {
                    int isIndoms = findIndexKernel(T.doms, startIndex + start,startIndex + start + T.doms[startIndex + end - 1],G_.newNeighbors[(2 * totalEdges * queryId) + j]);
                    if(tDoms == 0){
                        break;
                    }
                    if (isIndoms != -1) {
                     tDoms--;
                    if (T.degreeInR[ind] != 0) {
                        T.degreeInR[ind]--;
                        T.degreeInR[writeOffset + loc]--;

                    }
                    }
                }

              

            }
          }
          if (vertex == ustar) {
            ui tDoms =  T.doms[startIndex + end - 1];
            for (int j = G_.newOffset[(queryId * (size + 1)) + vertex]; j < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; j++) {
              int isIndoms = findIndexKernel(
                T.doms, startIndex + start,
                startIndex + start + T.doms[startIndex + end - 1],
                G_.newNeighbors[(2 * totalEdges * queryId) + j]);
                if(tDoms == 0){
                      break;
                }
              if (isIndoms != -1) {
                tDoms--;
                if (T.degreeInR[ind] != 0) {
                  T.degreeInR[ind]--;

                }
              }
            }
          }

        }
        __syncwarp();
        if (laneId == 0) {
            G_.flag[queryId] = 1;
            T.statusList[T.ustar[otherStartIndex + iter]] = 1;

            T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite + 1] =
              T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite] +
              sharedCounter[threadIdx.x / warpSize];
            T.numTasks[writePartition]++;

            T.size[(writePartition) * otherPsize + totalTasksWrite] =
              T.size[warpId * otherPsize + iter];
            T.size[warpId * otherPsize + iter] += 1;
            T.queryIndicator[(writePartition) * otherPsize + totalTasksWrite] = queryId;
            T.ustar[(writePartition) * otherPsize + totalTasksWrite] = -1;
            T.ustar[warpId * otherPsize + iter] = -1;
          
        }
        __syncwarp();
}

__device__ __forceinline__ void writeDomTasks(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B, ui* sharedCounter,ui ustar, ui writePartition, ui overFlow, ui totalDoms, ui writeOffset,ui domsWriteOffset,ui newTotalTasksWrite, ui startIndex,ui start, ui end, ui pSize, ui  offsetPsize, ui otherPsize, ui size, ui totalEdges,ui totalWrite,ui queryId, ui iter,ui warpId, ui laneId) {
   __syncwarp();
   for (ui domIndex = 0; domIndex < overFlow; domIndex++) {
            for (ui i = laneId; i < totalWrite - 1; i += 32) {

              ui srcIndex = writeOffset + i;
              int key2 = findIndexKernel(T.doms, startIndex + start + domIndex,
                startIndex + start + totalDoms,
                T.taskList[srcIndex]);
              ui domVertex = T.doms[startIndex + start + domIndex];

              ui dstIndex;
              dstIndex = domsWriteOffset + (totalWrite * domIndex) + i;
              ui vertex = T.taskList[srcIndex];
              T.taskList[dstIndex] = vertex;
              T.statusList[dstIndex] =
                (key2 != -1) ? 0 : T.statusList[srcIndex];
              T.degreeInC[dstIndex] = T.degreeInC[srcIndex];
              T.degreeInR[dstIndex] = T.degreeInR[srcIndex];
              if (T.statusList[dstIndex] != 2) {
                for (ui k = G_.newOffset[(queryId * (size + 1)) + vertex]; k < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; k++) {
                  if (G_.newNeighbors[(2 * totalEdges * queryId) + k] == ustar) {
                    T.degreeInC[dstIndex]++;
                    break;
                  }
                  
                }

                 for (ui k = G_.newOffset[(queryId * (size + 1)) + vertex]; k < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; k++) {
                  if (G_.newNeighbors[(2 * totalEdges * queryId) + k] == domVertex) {
                    T.degreeInC[dstIndex]++;
                    break;
                  }
                }
              }

              if (T.taskList[srcIndex] == domVertex) {
                T.statusList[dstIndex] = 1;
              }
            }

            __syncwarp();

            if (laneId == 0) {
              T.taskList[domsWriteOffset + (totalWrite * domIndex) +
                totalWrite - 1] = ustar;
              T.statusList[domsWriteOffset + (totalWrite * domIndex) +
                totalWrite - 1] = 1;
              T.size[(writePartition) * otherPsize + newTotalTasksWrite + domIndex] =
                T.size[warpId * otherPsize + iter] + 1;
              T.taskOffset[(writePartition) * offsetPsize + newTotalTasksWrite +
                  domIndex + 1] =
                T.taskOffset[(writePartition) * offsetPsize + newTotalTasksWrite] +
                ((domIndex + 1) * totalWrite);
              T.numTasks[writePartition]++;
              T.doms[startIndex + end - 1] = 0;
              T.queryIndicator[(writePartition) * otherPsize + newTotalTasksWrite + domIndex] = queryId;
              T.ustar[(writePartition) * otherPsize + newTotalTasksWrite + domIndex] = -1;


            }
          }

          __syncwarp();
          for (ui domIndex = 0; domIndex < overFlow; domIndex++) {
            __syncwarp();
            for (ui i = laneId; i < totalWrite; i += 32) {
              ui ind = domsWriteOffset + (totalWrite * domIndex) + i;
              ui vertex = T.taskList[ind];
              int addedDomKey =
                findIndexKernel(T.doms, startIndex + start + domIndex + 1,
                  startIndex + start + totalDoms, vertex);
              ui domVertex = T.doms[startIndex + start + domIndex];

              if ((addedDomKey != -1) || (vertex == domVertex) ||
                (vertex == ustar)) {
                ui dC = 0;
                ui dR = 0;
                int neighKey;
                for (ui k = G_.newOffset[(queryId * (size + 1)) + vertex]; k < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; k++) {
                  neighKey = findIndexKernel(
                    T.taskList, domsWriteOffset + (totalWrite * domIndex),
                    domsWriteOffset + (totalWrite * (domIndex + 1)),
                    G_.newNeighbors[(2 * totalEdges * queryId) + k]);
                  if (neighKey != -1) {
                    if (T.statusList[neighKey] == 0) {
                      dR++;
                    }
                    if (T.statusList[neighKey] == 1) {
                      dC++;
                    }
                    if (addedDomKey != -1) {
                      atomicAdd( & T.degreeInR[neighKey], 1);
                    }
                  }
                }

                T.degreeInC[ind] = dC;
                T.degreeInR[ind] = dR;
              }
            }
          }
          __syncwarp();
        
}

__device__ __forceinline__ void writeDomsBuffer(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B,ui ustar, ui overFlow, ui totalDoms, ui writeOffset, ui startIndex,ui start, ui size,ui totalWrite,ui queryId,ui bufferSize, ui otherPsize,ui iter,ui warpId,ui laneId) {
           ui numTaskBuffer;
          bool writeFlag;
          for (ui domIndex = overFlow; domIndex < totalDoms; domIndex++) {
            __syncwarp();
            numTaskBuffer = UINT_MAX;
            writeFlag = 0;
            if (laneId == 0) {
              while (1) {
                if (atomicCAS(B.writeMutex, 0, 1) == 0) {
                  if (((bufferSize - B.taskOffset[ * B.temp]) > totalWrite) && ( * B.temp < * B.limitTasks)) {
                    numTaskBuffer = atomicAdd(B.temp, 1);
                    __threadfence();
                    B.taskOffset[numTaskBuffer + 1] =
                      B.taskOffset[numTaskBuffer] + totalWrite;

                    __threadfence();
                    writeFlag = 1;
                    atomicExch(B.writeMutex, 0);
                    break;
                  } else {
                    *B.outOfMemoryFlag = 1;
                    atomicExch(B.writeMutex, 0);
                    break;

                  }

                }
              }
            }
            __syncwarp();

            numTaskBuffer = __shfl_sync(0xFFFFFFFF, numTaskBuffer, 0);
            writeFlag = __shfl_sync(0xFFFFFFFF, writeFlag, 0);
            if (writeFlag) {
              for (ui i = laneId; i < totalWrite - 1; i += 32) {
                ui srcIndex = writeOffset + i;
                int key2 = findIndexKernel(T.doms, startIndex + start + domIndex,
                  startIndex + start + totalDoms,
                  T.taskList[srcIndex]);
                ui domVertex = T.doms[startIndex + start + domIndex];
                ui dstIndex = B.taskOffset[numTaskBuffer] + i;
                B.taskList[dstIndex] = T.taskList[srcIndex];
                B.statusList[dstIndex] =
                  (key2 != -1) ? 0 : T.statusList[srcIndex];
                if (T.taskList[srcIndex] == domVertex) {
                  B.statusList[dstIndex] = 1;
                }
              }
              __syncwarp();

              if (laneId == 0) {
                atomicAdd(B.numTask, 1);
                B.size[numTaskBuffer] = T.size[warpId * otherPsize + iter] + 1;
                B.taskList[B.taskOffset[numTaskBuffer + 1] - 1] = ustar;
                B.statusList[B.taskOffset[numTaskBuffer + 1] - 1] = 1;
                B.queryIndicator[numTaskBuffer] = queryId;
                G_.numWrite[queryId]++;
                G_.flag[queryId] = 1;
              }
            }
          }


}

__device__ __forceinline__ void readFromBuffer(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B, double copyLimit,ui lastWritten, ui startIndex, ui start, ui total, ui iter, ui pSize,ui offsetPsize,ui otherPsize, ui size, ui totalEdges, ui bufferSize,ui warpId, ui laneId) {
            ui writePartition = T.sortedIndex[T.mapping[warpId]];
            ui totalTasksWrite = T.numTasks[writePartition];
            ui writeOffset = ((writePartition) * pSize) + T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite];
            if ((T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite] < (ui)(pSize * copyLimit)) && (totalTasksWrite < * T.limitTasks)) {
              ui numRead = UINT_MAX;
              ui readFlag = 0;
              __syncwarp();
              if (laneId == 0) {

                while (true) {
                  if (atomicCAS(B.readMutex, 0, 1) == 0) {
                    if ( * B.numReadTasks < lastWritten) {
                      numRead = atomicAdd(B.numReadTasks, 1);
                      readFlag = 1;

                    }

                    atomicExch(B.readMutex, 0);
                    break;
                  }
                }
              }
              __syncwarp();
              numRead = __shfl_sync(0xffffffff, numRead, 0);
              readFlag = __shfl_sync(0xffffffff, readFlag, 0);

              if (readFlag == 1) {
                ui readStart = B.taskOffset[numRead];
                ui readEnd = B.taskOffset[numRead + 1];
                ui totalRead = readEnd - readStart;
                ui readQueryId1 = B.queryIndicator[numRead];

                for (ui i = laneId; i < totalRead; i += warpSize) {
                  ui ind = readStart + i;
                  ui vertex = B.taskList[ind];
                  ui status = B.statusList[ind];
                  T.taskList[writeOffset + i] = vertex;
                  T.statusList[writeOffset + i] = status;
                  int keyTask;
                  ui degC = 0;
                  ui degR = 0;
                  if (status != 2) {
                    for (ui k = G_.newOffset[(readQueryId1 * (size + 1)) + vertex]; k < G_.newOffset[(readQueryId1 * (size + 1)) + vertex + 1]; k++) {
                      keyTask = findIndexKernel(B.taskList, readStart, readEnd, G_.newNeighbors[(2 * totalEdges * readQueryId1) + k]);

                      if (keyTask != -1) {
                        if (B.statusList[keyTask] == 1) {
                          degC++;

                        } else if (B.statusList[keyTask] == 0) {
                          degR++;

                        }

                      }

                    }

                  }

                  T.degreeInC[writeOffset + i] = degC;
                  T.degreeInR[writeOffset + i] = degR;
                }
                __syncwarp();

                if (laneId == 0) {
                  T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite + 1] = T.taskOffset[(writePartition) * offsetPsize + totalTasksWrite] + totalRead;
                  T.size[(writePartition) * otherPsize + totalTasksWrite] = B.size[numRead];
                  T.numTasks[writePartition]++;
                  T.queryIndicator[(writePartition) * otherPsize + totalTasksWrite] = readQueryId1;
                  T.ustar[(writePartition) * otherPsize + totalTasksWrite] = -1;
                  
                  G_.numRead[readQueryId1]++;
                  G_.flag[readQueryId1] = 1;

                }

              }
            }
          }

__device__ __forceinline__ void writeToBuffer(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, deviceBufferPointers B, ui writePartition, ui totalTasksWrite, ui writeOffset,ui ustar,  ui startIndex, ui start, ui pSize,ui otherPsize, ui bufferSize, ui size, ui totalEdges, ui total, ui queryId, ui iter, ui warpId,ui laneId) {
  ui numTaskBuffer;

    numTaskBuffer = UINT_MAX;
    ui availableMemory;
    bool writeFlag = 0;

    if (laneId == 0) {
        while (1) {
        if (atomicCAS(B.writeMutex, 0, 1) == 0) {
            availableMemory = bufferSize - B.taskOffset[ * B.temp];
            if (((availableMemory) > total) && ( * B.temp < * B.limitTasks)) {
            numTaskBuffer = atomicAdd(B.temp, 1);
            __threadfence();
            B.taskOffset[numTaskBuffer + 1] =
                B.taskOffset[numTaskBuffer] + total;
            __threadfence();
            writeFlag = 1;
            atomicExch(B.writeMutex, 0);
            break;

            } else {
            * B.outOfMemoryFlag = 1;
            atomicExch(B.writeMutex, 0);
            break;
            }

        }
        }
    }
    __syncwarp();
    numTaskBuffer = __shfl_sync(0xFFFFFFFF, numTaskBuffer, 0);
    writeFlag = __shfl_sync(0xFFFFFFFF, writeFlag, 0);

    if (writeFlag) {
        ui bufferWriteOffset = B.taskOffset[numTaskBuffer];

        for (ui i = laneId; i < total; i += warpSize) {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];

        ui degInR;

        B.taskList[bufferWriteOffset + i] = vertex;
        B.statusList[bufferWriteOffset + i] =
            (vertex != ustar) ? status : 2;

        degInR = T.degreeInR[ind];

        for (ui k = G_.newOffset[(queryId * (size + 1)) + vertex]; k < G_.newOffset[(queryId * (size + 1)) + vertex + 1]; k++) {
            if (G_.newNeighbors[(2 * totalEdges * queryId) + k] == ustar) {
            T.degreeInC[ind]++;

            if (degInR != 0) {
                degInR--;
            }
            }
        }

        T.degreeInR[ind] = degInR;
        }

        __syncwarp();
        if (laneId == 0) {
        B.size[numTaskBuffer] = T.size[warpId * otherPsize + iter];
        T.size[warpId * otherPsize + iter]++;
        T.statusList[T.ustar[warpId * otherPsize + iter]] = 1;
        atomicAdd(B.numTask, 1);
        B.queryIndicator[numTaskBuffer] = queryId;
        G_.flag[queryId] = 1;
        G_.numWrite[queryId]++;
        G_.flag[queryId] = 1;
        T.ustar[warpId * otherPsize + iter] = -1;

        }
    }     
}

