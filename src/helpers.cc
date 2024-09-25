__device__ ui minn(ui a, ui b) {
  /**
   * Returns the minimum of two numbers.

   */
  return (a < b) ? a : b;
  }



__device__ int findIndexKernel(ui* arr, ui start, ui end, ui target)

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

__device__ void warpBubbleSort(ui* arr, ui start, ui end, ui laneID, ui reverse ) {

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
            if(!reverse){
               if (laneID == i && val_i > val_j) {
                // Swap values within the warp
                arr[start + i] = val_j;
                arr[start + j] = val_i;
            }

            }else{
               if (laneID == i && val_i < val_j) {
                // Swap values within the warp
                arr[start + i] = val_j;
                arr[start + j] = val_i;
            }
            }


        }
    }
}

__device__ void selectionSort(ui* values, ui start, ui end, ui laneId) {

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

  int n = end - start +1;

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


__device__ void warpSelectionSort(double* keys, ui* values, ui start, ui end,
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

  ui* local_counter = shared_memory1;

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
    if ((G.core[vertex] >= G_.lowerBoundDegree[queryId]) &&
        (G_.distance[(queryId*size)+vertex] <= upperBoundDistance)) {
      ui loc = atomicAdd(&local_counter[threadIdx.x / warpSize], 1);
      P.initialTaskList[loc + writeOffset] = vertex;

    } else {
      G_.degree[(queryId*size) + vertex] = 0;
    }
  }

  __syncwarp();
  if (laneId == 0) {
    P.entries[warpId] = local_counter[threadIdx.x / warpSize];

    atomicAdd(P.globalCounter, local_counter[threadIdx.x / warpSize]);
  }
}

__global__ void CompressTask(deviceGraphGenPointers G, deviceGraphPointers G_, deviceInterPointers P,
  deviceTaskPointers T, ui pSize, ui queryVertex, ui queryId, ui size, ui taskPSize, ui totalWarps,ui factor ) {

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui start = warpId * pSize;
  int temp = warpId - 1;
  ui total = P.entries[warpId];
  ui offsetPsize = taskPSize/factor;
  ui otherPsize = *T.limitTasks;
  // add some of mechanism
  ui minTasks = UINT_MAX;
  ui writeWarp = 0;
  ui spaceLeft;
  ui numTasks;
  for(ui i =0;i<totalWarps;i++){
    numTasks = T.numTasks[i];
    spaceLeft = taskPSize - T.taskOffset[offsetPsize*i + numTasks];
    if( (spaceLeft > *P.globalCounter) && (numTasks < *T.limitTasks) ){
      if( numTasks < minTasks){
        minTasks = numTasks;
        writeWarp = i;
      }

    }

  }
  __syncwarp();
  numTasks = T.numTasks[writeWarp];
  ui writeOffset = taskPSize*writeWarp + T.taskOffset[offsetPsize*writeWarp + numTasks];

  if (idx == 0) {
    T.size[otherPsize*(writeWarp) + numTasks ] = 1;
    T.queryIndicator[otherPsize*(writeWarp) + numTasks ] = queryId;
    T.taskOffset[offsetPsize*(writeWarp) + numTasks + 1] =  T.taskOffset[offsetPsize*(writeWarp) + numTasks] + *P.globalCounter;
    T.ustar[otherPsize*(writeWarp) +numTasks ] = -1;
    T.numTasks[writeWarp]++;


  }


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
      if ((G.neighbors[k] != queryVertex) && (G_.degree[(queryId*size)+G.neighbors[k]] != 0)) {
        degInR++;
      }
      if (G.neighbors[k] == queryVertex) {
        degInc++;
      }
      if (G_.degree[(queryId*size)+G.neighbors[k]] != 0) {
        totalNeigh++;
      }
    }

    G_.newOffset[(queryId*(size+1))+ vertex + 1] = totalNeigh;
    T.degreeInR[writeOffset + i] = degInR;
    T.degreeInC[writeOffset + i] = degInc;
  }
}

__global__ void NeighborUpdate(deviceGraphGenPointers G, deviceGraphPointers G_, ui INTOTAL_WARPS,ui queryId, ui size, ui totalEdges ) {

  extern __shared__ ui sharedMem[];
  ui * localCounter = sharedMem;

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  for (ui i = warpId; i < size; i += INTOTAL_WARPS) {
    if (laneId == 0) {
      localCounter[threadIdx.x / warpSize] = 0;
    }
    __syncwarp();
    ui degree = G_.degree[(queryId*size) + i];

    if (degree > 0) {
      ui writeOffset = G_.newOffset[(queryId*(size+1)) + i];

      ui start = G.offset[i];
      ui end = G.offset[i + 1];
      ui total = end - start;
      ui neighbor;
      ui neighborDegree;
      for (ui j = laneId; j < total; j += warpSize) {
        neighbor = G.neighbors[start + j];
        neighborDegree = G_.degree[(queryId*size) + neighbor];
        if (neighborDegree > 0) {
          ui loc = atomicAdd( & localCounter[threadIdx.x / warpSize], 1);
          G_.newNeighbors[(2*totalEdges*queryId) + writeOffset + loc] = neighbor;


        }

      }
      __syncwarp();
      if (laneId == 0) {
        localCounter[threadIdx.x / warpSize] = 0;
      }

    }
    __syncwarp();
  }
}

__global__ void ProcessTask(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,
  ui pSize, ui factor, ui maxN2, ui size, ui totalEdges, ui dmax, ui limitQueries) {
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
  ui offsetPsize = pSize/factor;
  ui otherPsize = *T.limitTasks;
  double score;
  int ustar;
  ui currentMinDegree;


  double temp;
  ui temp3;
  ui degreeBasedUpperBound;

  int otherId;

  int resultIndex, resInd;
  ui currentSize;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui startIndex = warpId * pSize;
  ui offsetStartIndex = warpId*offsetPsize;
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

    ui maskGen = total;
    if ((T.ustar[otherStartIndex + iter] != INT_MAX) && (queryId != UINT_MAX) && (T.size[otherStartIndex + iter]<=upperBoundSize)) {
      for (ui i = laneId; i < total; i += warpSize)

      {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];
        ui degR = T.degreeInR[ind];
        ui degC = T.degreeInC[ind];
        ui hSize = T.size[otherStartIndex + iter];

        ui startNeighbor = G_.newOffset[(queryId * (size + 1)) + vertex];
        ui endNeighbor = G_.newOffset[(queryId * (size + 1)) + vertex + 1];
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
        unsigned int mask;

        if (maskGen > warpSize) {
          mask = 0xFFFFFFFF;
          maskGen -= warpSize;

        } else {
          mask = (1u << maskGen) - 1;
        }
        __syncwarp(mask);

        if (status == 0) {
          if ((minn((degR + degC), (degC + upperBoundSize - hSize - 1)) <=
              G_.lowerBoundDegree[queryId]) ||
            (ubD < G_.distance[(queryId * size) + vertex])) {
            T.statusList[ind] = 2;
            T.degreeInR[ind] = 0;
            T.degreeInC[ind] = 0;
            for (int j = startNeighbor; j < endNeighbor; j++) {
              resultIndex = findIndexKernel(T.taskList, startIndex + start,
                startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + j]);
              if (resultIndex != -1) {
                if (T.degreeInR[resultIndex] != 0) {
                  atomicSub( & T.degreeInR[resultIndex], 1);
                  if (T.degreeInR[resultIndex] == 4294967295) {
                    T.degreeInR[resultIndex] = 0;
                  }
                }
              }
            }
          }
        }

        ui neig;
        if ((status == 1) && ((degC + degR) == (G_.lowerBoundDegree[queryId] + 1))) {
          // printf("R3 iter %u wrap %u lane %u index %u vertex %u \n",
          // iter,warpId,laneId,i,vertex);

          for (int j = startNeighbor; j < endNeighbor; j++) {
            resultIndex = findIndexKernel(T.taskList, startIndex + start,
              startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + j]);
            if (resultIndex != -1) {
              if ((atomicCAS( & T.statusList[resultIndex], 0, 1) == 0) &&
                (atomicAdd( & T.size[otherStartIndex + iter], 0) < upperBoundSize)) {
                atomicAdd( & T.size[otherStartIndex + iter], 1);
                neig = T.taskList[resultIndex];

                for (int k = G_.newOffset[(queryId * (size + 1)) + neig]; k < G_.newOffset[(queryId * (size + 1)) + neig + 1]; k++) {
                  resInd = findIndexKernel(T.taskList, startIndex + start,
                    startIndex + end, G_.newNeighbors[(2 * totalEdges * queryId) + k]);
                  if (resInd != -1) {
                    atomicAdd( & T.degreeInC[resInd], 1);
                    if (T.degreeInR[resInd] != 0) {
                      atomicSub( & T.degreeInR[resInd], 1);
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

       ui maskGen = total;
       for (ui i = laneId; i < total; i += warpSize){
        ui ind = startIndex + start + i;
        ui status = T.statusList[ind];
        ui degInC = T.degreeInC[ind];
        ui degInR = T.degreeInR[ind];
        currentMinDegree = UINT_MAX;
        degreeBasedUpperBound = UINT_MAX;
        ui oneside;

        unsigned int mask;
        ui cond;

        if (maskGen > warpSize) {
          mask = 0xFFFFFFFF;
          maskGen -= warpSize;
          cond = warpSize;
          // printf("here");

        } else {
          mask = (1u << maskGen) - 1;
          cond = maskGen;
          // printf("2nd here");
        }

        if (status == 1) {
          oneside =
            degInC + upperBoundSize - T.size[otherStartIndex + iter] - 1;
          degreeBasedUpperBound =
            minn(oneside, degInR +degInC );
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          temp3 = __shfl_down_sync(mask, degreeBasedUpperBound, offset);

          if (laneId + offset < cond) {
            degreeBasedUpperBound = min(degreeBasedUpperBound, temp3);
          }
        }
        degreeBasedUpperBound = __shfl_sync(mask, degreeBasedUpperBound, 0);

        if (status == 1) {
          currentMinDegree = T.degreeInC[ind];
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          ui temp2 = __shfl_down_sync(mask, currentMinDegree, offset);

          if (laneId + offset < cond) {
            currentMinDegree = minn(currentMinDegree, temp2);
          }
        }

        currentMinDegree = __shfl_sync(mask, currentMinDegree, 0);

        if (status == 0) {
          ui locR = atomicAdd( & sharedCounterR[threadIdx.x / warpSize], 1);
          T.doms[startIndex + start + locR] = degInC;

        }
        if (status == 1) {
          ui locC = atomicAdd( & sharedCounterC[threadIdx.x / warpSize], 1);
          sharedC_[(threadIdx.x / warpSize) * maxN2 + locC] = degInC;

        }
        if (i % warpSize == 0) {
           if (currentMinDegree < sharedDegree[threadIdx.x / warpSize]) {
            sharedDegree[threadIdx.x / warpSize] = currentMinDegree;
          }
          if (degreeBasedUpperBound < sharedUBDegree[threadIdx.x / warpSize]) {
            sharedUBDegree[threadIdx.x / warpSize] = degreeBasedUpperBound;
          }
        }

       }
      __syncwarp();
      selectionSort(T.doms, startIndex + start, startIndex + start + sharedCounterR[threadIdx.x / warpSize], laneId);

      __syncwarp();
      warpBubbleSort(sharedC_, (threadIdx.x / warpSize) * maxN2, (threadIdx.x / warpSize) * maxN2 + sharedCounterC[threadIdx.x / warpSize], laneId, 0);
      currentSize = T.size[otherStartIndex + iter];
      ui justCMD = sharedC_[(threadIdx.x / warpSize) * maxN2];
      ui totalIter_ = minn((upperBoundSize - currentSize), sharedCounterR[threadIdx.x / warpSize]);
      for (ui iter_ = 0; iter_ < totalIter_; iter_++) {
        __syncwarp();
        ui value = T.doms[startIndex + start + iter_];
        for (ui i = laneId; i < minn(value,sharedCounterC[threadIdx.x / warpSize]); i += 32) {
          sharedC_[(threadIdx.x / warpSize) * maxN2 + i]++;
        }
        __syncwarp();
        warpBubbleSort(sharedC_, (threadIdx.x / warpSize) * maxN2, (threadIdx.x / warpSize) * maxN2 + sharedCounterC[threadIdx.x / warpSize], laneId, 0);

      }

      __syncwarp();


      if(laneId==0){

        //printf("warp Id %u iter %u query %u current size %u current %u shared  % u \n",warpId,iter,queryId,currentSize,G_.lowerBoundDegree[queryId],sharedDegree[threadIdx.x / warpSize]);
        if ((lowerBoundSize <= currentSize) &&
          (currentSize <= upperBoundSize)) {
          if (sharedDegree[threadIdx.x / warpSize] != UINT_MAX) {

            ui old = atomicMax( & G_.lowerBoundDegree[queryId],sharedDegree[threadIdx.x / warpSize]);
            /*if(old<G_.lowerBoundDegree[queryId]){
              printf(" solution query %u warp %u iter %u old %u new %u cmd %u ub2 %u ub1 %u  \n", queryId,warpId,iter,old,G_.lowerBoundDegree[queryId],justCMD,sharedC_[(threadIdx.x / warpSize) * maxN2],sharedUBDegree[threadIdx.x / warpSize]);
            }*/


          }
        }
      }

      __syncwarp();
       ui upperBoundDegreeLimit = minn(sharedC_[(threadIdx.x / warpSize) * maxN2], sharedUBDegree[threadIdx.x / warpSize]);
       ui kl = G_.lowerBoundDegree[queryId];
       if (upperBoundDegreeLimit > kl ){

             maskGen = total;
            for (ui i = laneId; i < total; i += warpSize) {
                int index = startIndex + start + i;
                ui ind = startIndex + start + i;
                ui vertex = T.taskList[ind];
                ui status = T.statusList[ind];

                ui startNeighbor = G_.newOffset[(queryId * (size + 1)) + vertex];
                ui endNeighbor = G_.newOffset[(queryId * (size + 1)) + vertex + 1];

                //ui hSize = T.size[startIndex + iter];

                score = 0;
                ustar = -1;

                if (status == 0) {
                for (int j = startNeighbor; j < endNeighbor; j++) {
                    resultIndex = findIndexKernel(T.taskList, startIndex + start,
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

                unsigned int mask;
                ui cond;

                if (maskGen > warpSize) {
                mask = 0xFFFFFFFF;
                maskGen -= warpSize;
                cond = warpSize;
                // printf("here");

                } else {
                mask = (1u << maskGen) - 1;
                cond = maskGen;
                // printf("2nd here");
                }

                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                temp = __shfl_down_sync(mask, score, offset);
                otherId = __shfl_down_sync(mask, index, offset);

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

       }
         __syncwarp();
      if (laneId == 0) {
        int writeOffset = warpId * otherPsize;
        if ((sharedScore[threadIdx.x / warpSize] > 0) &&
          (sharedUBDegree[threadIdx.x / warpSize] != UINT_MAX) &&
          (total >= lowerBoundSize) && (upperBoundDegreeLimit > G_.lowerBoundDegree[queryId])) {
          T.ustar[writeOffset + iter] = sharedUstar[threadIdx.x / warpSize];

        } else {
          T.ustar[writeOffset + iter] = INT_MAX;
        }
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
    bool shouldDecrement = (T.queryIndicator[warpId * otherPsize + iter] == UINT_MAX) ||
      (T.ustar[warpId * otherPsize + iter] == INT_MAX);
    shouldDecrement = __all_sync(0xFFFFFFFF, shouldDecrement);

    if (!shouldDecrement) {
      break;
    }

    if (laneId == 0) {
      T.numTasks[warpId]--;
      T.ustar[warpId * otherPsize + iter] = -1;
      T.queryIndicator[warpId * otherPsize + iter] = limitQueries;
    }
  }
}
__global__ void Expand(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T,
                       deviceBufferPointers B, ui pSize, ui factor, ui jump, double copyLimit,
                       ui bufferSize, ui lastWritten,ui readLimit, ui size, ui totalEdges, ui dmax ) {

  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui* sharedCounter = (ui*)(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui offsetPsize = pSize/factor;
  ui otherPsize = *T.limitTasks;

  ui startIndex = warpId * pSize;
  ui offsetStartIndex = warpId*offsetPsize;
  ui otherStartIndex = warpId * otherPsize;


  ui totalTasks = T.numTasks[warpId];
  int key, resultIndex;

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

    ui upperBoundSize =0;
    if(queryId!=UINT_MAX){
      upperBoundSize = G_.upperBoundSize[queryId];
    }


    if ((T.ustar[otherStartIndex + iter] != INT_MAX) &&
        (T.ustar[otherStartIndex + iter] != -1) &&
        (T.size[otherStartIndex + iter] < upperBoundSize)  && ( queryId != UINT_MAX)) {
      ui bufferNum = T.sortedIndex[T.mapping[warpId]];
      ui totalTasksWrite = T.numTasks[bufferNum];
      ui writeOffset = ((bufferNum) * pSize) +
                       T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite];
      ui ustar = T.taskList[T.ustar[otherStartIndex + iter]];
      ui totalWrite;

      if (((writeOffset + total) < ((bufferNum+1) * pSize - 1))&&(totalTasksWrite< *T.limitTasks)) {
        for (ui i = laneId; i < total; i += warpSize) {
          ui ind = startIndex + start + i;
          ui vertex = T.taskList[ind];
          ui status = T.statusList[ind];

          ui degInR;
          ui degInC;
          if ((vertex != ustar) && (status != 2)) {
            ui loc = atomicAdd(&sharedCounter[threadIdx.x / warpSize], 1);
            T.taskList[writeOffset + loc] = vertex;
            T.statusList[writeOffset + loc] = status;
            degInR = T.degreeInR[ind];
            degInC = T.degreeInC[ind];

            for (ui k = G_.newOffset[(queryId*(size+1)) + vertex]; k < G_.newOffset[(queryId*(size+1)) + vertex + 1]; k++) {
              if (G_.newNeighbors[(2*totalEdges*queryId) + k] == ustar) {
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
              key = findIndexKernel(
                  T.doms, startIndex + start,
                  startIndex + start + T.doms[startIndex + end - 1], vertex);

              if (key != -1) {
                T.statusList[ind] = 2;
                T.statusList[writeOffset + loc] = 2;
                T.degreeInC[ind] = 0;
                T.degreeInR[ind] = 0;
                T.degreeInC[writeOffset + loc] = 0;
                T.degreeInR[writeOffset + loc] = 0;
                for (int j = G_.newOffset[(queryId*(size+1)) + vertex]; j < G_.newOffset[(queryId*(size+1)) + vertex + 1]; j++) {
                  resultIndex =
                      findIndexKernel(T.taskList, startIndex + start,
                                      startIndex + end, G_.newNeighbors[(2*totalEdges*queryId) + j]);
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
        }
        __syncwarp();
        ui totalDoms = T.doms[startIndex + end - 1];
        if (totalDoms != 0) {
          for (ui i = laneId; i < sharedCounter[threadIdx.x / warpSize];
               i += warpSize) {
            ui ind = writeOffset + i;
            ui vertex = T.taskList[ind];
            for (int j = G_.newOffset[(queryId*(size+1)) + vertex]; j < G_.newOffset[(queryId*(size+1)) + vertex + 1]; j++) {
              resultIndex = findIndexKernel(
                  T.doms, startIndex + start,
                  startIndex + start + T.doms[startIndex + end - 1],
                  G_.newNeighbors[(2*totalEdges*queryId) + j]);
              if (resultIndex != -1) {
                if (T.degreeInR[ind] != 0) {
                  T.degreeInR[ind]--;
                }
              }
            }
          }
        }
        __syncwarp();
        if (laneId == 0) {
          if ((T.size[otherStartIndex + iter] < upperBoundSize) &&
              (T.ustar[otherStartIndex + iter] != -1)) {
            G_.flag[queryId] = 1;
            T.statusList[T.ustar[otherStartIndex + iter]] = 1;

            T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite + 1] =
                T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite] +
                sharedCounter[threadIdx.x / warpSize];
            T.numTasks[bufferNum]++;

            T.size[(bufferNum) * otherPsize + totalTasksWrite] =
                T.size[warpId * otherPsize + iter];
            T.size[warpId * otherPsize + iter] += 1;
            T.queryIndicator[(bufferNum) * otherPsize + totalTasksWrite] = queryId;
          }
        }
        __syncwarp();

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
          int leftTasks = *T.limitTasks - newTotalTasksWrite;
          leftSpace = ((bufferNum+1) * pSize) - (domsWriteOffset + totalWrite);

          leftSpace = max(leftSpace, 0);
          overFlow = leftSpace / totalWrite;
          overFlow = min(leftTasks,overFlow);
          overFlow = min(overFlow, totalDoms);
          for (ui domIndex = 0; domIndex < overFlow; domIndex++) {
            __syncwarp();
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
                for (ui k = G_.newOffset[(queryId*(size+1)) + vertex]; k < G_.newOffset[(queryId*(size+1)) + vertex + 1]; k++) {
                  if (G_.newNeighbors[(2*totalEdges*queryId) + k] == ustar) {
                    T.degreeInC[dstIndex]++;
                  }
                  if (G_.newNeighbors[(2*totalEdges*queryId) + k] == domVertex) {
                    T.degreeInC[dstIndex]++;
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
              T.size[(bufferNum) * otherPsize + newTotalTasksWrite + domIndex] =
                  T.size[warpId * otherPsize + iter] + 1;
              T.taskOffset[(bufferNum) * offsetPsize + newTotalTasksWrite +
                           domIndex + 1] =
                  T.taskOffset[(bufferNum) * offsetPsize + newTotalTasksWrite] +
                  ((domIndex + 1) * totalWrite);
              T.numTasks[bufferNum]++;
              T.doms[startIndex + end - 1] = 0;
              T.queryIndicator[(bufferNum) * otherPsize + newTotalTasksWrite + domIndex] = queryId;

            }
          }

          __syncwarp();
          for (ui domIndex = 0; domIndex < overFlow; domIndex++){
            __syncwarp();
           for (ui i = laneId; i < totalWrite; i += 32) {
              ui ind = domsWriteOffset + (totalWrite * domIndex) + i;
              ui vertex = T.taskList[ind];
              int addedDomKey =
                  findIndexKernel(T.doms, startIndex + start + domIndex+1 ,
                                  startIndex + start + totalDoms, vertex);
              ui domVertex = T.doms[startIndex + start + domIndex];

              if ((addedDomKey != -1) || (vertex == domVertex) ||
                  (vertex == ustar)) {
                ui dC = 0;
                ui dR = 0;
                int neighKey;
                for (ui k = G_.newOffset[(queryId*(size+1)) + vertex]; k < G_.newOffset[(queryId*(size+1)) + vertex + 1]; k++) {
                  neighKey = findIndexKernel(
                      T.taskList, domsWriteOffset + (totalWrite * domIndex),
                      domsWriteOffset + (totalWrite * (domIndex + 1)),
                      G_.newNeighbors[(2*totalEdges*queryId) + k]);
                  if (neighKey != -1) {
                    if (T.statusList[neighKey] == 0) {
                      dR++;
                    }
                    if (T.statusList[neighKey] == 1) {
                      dC++;
                    }
                    if (addedDomKey != -1) {
                      atomicAdd(&T.degreeInR[neighKey], 1);
                    }
                  }
                }

                T.degreeInC[ind] = dC;
                T.degreeInR[ind] = dR;
              }
            }
            }


          __syncwarp();
          ui numTaskBuffer1;
           bool writeFlag;
          for (ui domIndex = overFlow; domIndex < totalDoms; domIndex++) {
            __syncwarp();
            // printf("here ");
            numTaskBuffer1 = UINT_MAX;
            writeFlag = 0;
            if (laneId == 0) {
              while (1) {
                if (atomicCAS(B.writeMutex, 0, 1) == 0) {
                  if(((bufferSize - B.taskOffset[*B.temp]) > totalWrite)&&(*B.temp<*B.limitTasks)){
                        numTaskBuffer1 = atomicAdd(B.temp, 1);
                        __threadfence();
                        B.taskOffset[numTaskBuffer1 + 1] =
                            B.taskOffset[numTaskBuffer1] + totalWrite;

                        __threadfence();
                        writeFlag = 1;
                        atomicExch(B.writeMutex, 0);
                        break;
                  }else{
                    *B.outOfMemoryFlag = 1;
                    atomicExch(B.writeMutex, 0);
                    break;

                  }



                }
              }
            }
            __syncwarp();

            numTaskBuffer1 = __shfl_sync(0xFFFFFFFF, numTaskBuffer1, 0);
            writeFlag = __shfl_sync(0xFFFFFFFF,writeFlag , 0);
            if (writeFlag) {
            for (ui i = laneId; i < totalWrite - 1; i += 32) {
              ui srcIndex = writeOffset + i;
              int key2 = findIndexKernel(T.doms, startIndex + start + domIndex,
                                         startIndex + start + totalDoms,
                                         T.taskList[srcIndex]);
              ui domVertex = T.doms[startIndex + start + domIndex];
              ui dstIndex = B.taskOffset[numTaskBuffer1] + i;
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
              B.size[numTaskBuffer1] = T.size[warpId * otherPsize + iter] + 1;
              B.taskList[B.taskOffset[numTaskBuffer1 + 1] - 1] = ustar;
              B.statusList[B.taskOffset[numTaskBuffer1 + 1] - 1] = 1;
              B.queryIndicator[numTaskBuffer1] = queryId;
              G_.numWrite[queryId] ++;
              G_.flag[queryId] = 1;

              // printf("doms write iter %u warp %u loc %u \n", iter, warpId,
              // bufferNum);
            }
            }
          }

        } else {
          if (( *B.numTask> 0)) {

            ui bufferNum = T.sortedIndex[T.mapping[warpId]];
            ui totalTasksWrite = T.numTasks[bufferNum];
            ui writeOffset = ((bufferNum) * pSize) + T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite];
            if((T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite] < (ui)(pSize*copyLimit)) && (totalTasksWrite< *T.limitTasks )){
           ui numRead1 = UINT_MAX; ui readFlag1 = 0;
            __syncwarp();
            if (laneId == 0) {

              while (true) {
                if (atomicCAS(B.readMutex, 0, 1) == 0) {
                  if ( * B.numReadTasks < lastWritten) {
                      numRead1 = atomicAdd(B.numReadTasks, 1); readFlag1 = 1;

                  }

                  atomicExch(B.readMutex, 0);
                  break;
                }
              }
            }
            __syncwarp();
            numRead1 = __shfl_sync(0xffffffff, numRead1, 0);
            readFlag1 = __shfl_sync(0xffffffff, readFlag1, 0);

            if (readFlag1 == 1) {
              //printf("here ");
              ui readStart = B.taskOffset[numRead1];
              ui readEnd = B.taskOffset[numRead1 + 1];
              ui totalRead = readEnd - readStart;
              ui readQueryId1 = B.queryIndicator[numRead1];

              for (ui i = laneId; i < totalRead; i += warpSize) {
                ui ind = readStart + i;
                ui vertex = B.taskList[ind];
                ui status = B.statusList[ind];
                T.taskList[writeOffset + i] = vertex;
                T.statusList[writeOffset + i] = status;
                int keyTask;
                ui degC = 0;
                ui degR = 0;
                if(status!=2){
                   for (ui k = G_.newOffset[(readQueryId1*(size+1)) + vertex]; k < G_.newOffset[(readQueryId1*(size+1)) + vertex + 1]; k++) {
                  keyTask = findIndexKernel(B.taskList, readStart, readEnd,G_.newNeighbors[(2*totalEdges*readQueryId1) + k]);

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
                T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite + 1] = T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite] + totalRead;
                T.size[(bufferNum) * otherPsize + totalTasksWrite] =B.size[numRead1];
                T.numTasks[bufferNum]++;
                 T.queryIndicator[(bufferNum) * otherPsize + totalTasksWrite] = readQueryId1;
                 G_.numRead[readQueryId1]++;
                G_.flag[readQueryId1] = 1;



              }

            }
          }
          }
        }
      } else {
        ui numTaskBuffer;

        numTaskBuffer = UINT_MAX;
        ui availableMemory;
        bool writeFlag1 = 0;

        if (laneId == 0) {
          // printf("Before iter %u warpId % u offset %u end %u numTaskBuffer %u
          // temp %u numTask %u \n",iter, warpId,(writeOffset + total) ,
          // (bufferNum * pSize - 1),numTaskBuffer,*B.temp,*B.numTask);


          while (1) {
            if (atomicCAS(B.writeMutex, 0, 1) == 0) {
              availableMemory = bufferSize - B.taskOffset[*B.temp];
              if(((availableMemory) > total) && (*B.temp<*B.limitTasks) ){
                numTaskBuffer = atomicAdd(B.temp, 1);
                __threadfence();
                B.taskOffset[numTaskBuffer + 1] =
                    B.taskOffset[numTaskBuffer] + total;
                __threadfence();
                writeFlag1=1;
                atomicExch(B.writeMutex, 0);
                break;

              }else{
                *B.outOfMemoryFlag = 1;
                atomicExch(B.writeMutex, 0);
                break;
              }

            }
          }
          // printf("Write expand \n ");
        }
        __syncwarp();
        numTaskBuffer = __shfl_sync(0xFFFFFFFF, numTaskBuffer, 0);
            writeFlag1 = __shfl_sync(0xFFFFFFFF,writeFlag1 , 0);


        if (writeFlag1) {
          ui bufferWriteOffset = B.taskOffset[numTaskBuffer];

          for (ui i = laneId; i < total; i += warpSize) {
            ui ind = startIndex + start + i;
            ui vertex = T.taskList[ind];
            ui status = T.statusList[ind];

            ui degInR;
            //ui degInC;

            B.taskList[bufferWriteOffset + i] = vertex;
            B.statusList[bufferWriteOffset + i] =
                (vertex != ustar) ? status : 2;

            degInR = T.degreeInR[ind];
            //degInC = T.degreeInC[ind];

            for (ui k = G_.newOffset[(queryId*(size+1)) + vertex]; k < G_.newOffset[(queryId*(size+1)) + vertex + 1]; k++) {
              if (G_.newNeighbors[(2*totalEdges*queryId) + k] == ustar) {
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



          }
        }
      }
    }
  }

  __syncwarp();
  if (T.numTasks[warpId] == 0) {
  for(ui iterRead = 0; iterRead < readLimit; iterRead ++){
  __syncwarp();

    ui bufferNum = T.sortedIndex[T.mapping[warpId]];
    ui totalTasksWrite = T.numTasks[bufferNum];
    ui writeOffset = ((bufferNum) * pSize) +
                     T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite];
    if((T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite]< (ui) (pSize*copyLimit)) && (totalTasksWrite< *T.limitTasks )){
    ui numRead = UINT_MAX;
    ui readFlag = 0;
    __syncwarp();
    if ((laneId == 0)) {
      while (true) {
        if (atomicCAS(B.readMutex, 0, 1) == 0) {
          if (*B.numReadTasks < lastWritten) {
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
       /*if(laneId==0)
              printf("wrap %u readQueryId %u \n",warpId,readQueryId);*/


      for (ui i = laneId; i < totalRead; i += warpSize) {
        ui ind = readStart + i;
        ui vertex = B.taskList[ind];
        ui status = B.statusList[ind];

        T.taskList[writeOffset + i] = vertex;
        T.statusList[writeOffset + i] = status;
        int keyTask;
        ui degC = 0;
        ui degR = 0;
        if(status!=2){
          for (ui k = G_.newOffset[(readQueryId*(size+1)) + vertex]; k < G_.newOffset[(readQueryId*(size+1)) + vertex + 1]; k++) {
          keyTask =
              findIndexKernel(B.taskList, readStart, readEnd, G_.newNeighbors[(2*totalEdges*readQueryId) + k]);

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
        T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite + 1] =
            T.taskOffset[(bufferNum) * offsetPsize + totalTasksWrite] + totalRead;
        T.size[(bufferNum) * otherPsize + totalTasksWrite] = B.size[numRead];
        T.numTasks[bufferNum]++;
        T.queryIndicator[(bufferNum) * otherPsize + totalTasksWrite] = readQueryId;
        G_.numRead[readQueryId]++;
        G_.flag[readQueryId] = 1;


      }
    }
  }
   }
  }
}

__global__ void FindDoms(deviceGraphGenPointers G, deviceGraphPointers G_, deviceTaskPointers T, ui pSize, ui factor, ui size, ui totalEdges, ui dmax) {
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

  ui* sharedCounter = (ui*)(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  int resultIndex;

  ui offsetPsize = pSize/factor;
  ui otherPsize = *T.limitTasks;

  ui startIndex = warpId * pSize;
  ui offsetStartIndex = warpId*offsetPsize;
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
    if(queryId!=UINT_MAX)
    limitDoms = G_.limitDoms[queryId];

    if ((T.ustar[otherStartIndex + iter] != INT_MAX) &&
        (T.ustar[otherStartIndex + iter] != -1) && ( queryId != UINT_MAX) && (limitDoms > 0)) {
      ui ustar = T.taskList[T.ustar[otherStartIndex + iter]];
      for (ui i = laneId; i < total; i += 32) {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];
        ui degC = T.degreeInC[ind];

        ui startNeighbor = G_.newOffset[(queryId*(size+1)) + vertex];
        ui endNeighbor = G_.newOffset[(queryId*(size+1)) + vertex + 1];
        bool is_doms = false;
        double score = 0;

        if ((status == 0) && (vertex != ustar) && (degC != 0)) {
          is_doms = true;
          ui neighbor;
          for (int j = startNeighbor; j < endNeighbor; j++) {
            neighbor = G_.newNeighbors[(2*totalEdges*queryId) + j];

            bool found = false;
            if (neighbor != ustar) {
              for (ui k = G_.newOffset[(queryId*(size+1)) + ustar]; k < G_.newOffset[(queryId*(size+1)) + ustar + 1]; k++) {
                if (neighbor == G_.newNeighbors[(2*totalEdges*queryId) + k]) {
                  found = true;
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
                  score += (double)1 / T.degreeInC[resultIndex];
                }
              }
            }
            // printf("iter %u wrap %u lane %u vertex %u status %u neighbor %u
            // index %d score %f
            // \n",iter,warpId,i,vertex,status,neighbor,resultIndex,score);
          }
          score += (double)T.degreeInR[ind] / dmax;
        }
        // printf("iter %u wrap %u lane %u vertex %u status %u ustar %u doms %u
        // \n",iter,warpId,i,vertex,status,ustar, is_doms);

        if (is_doms) {
          ui loc = atomicAdd(&sharedCounter[threadIdx.x / 32], 1);
          T.doms[writeOffset + loc] = vertex;
          T.cons[writeOffset + loc] = score;
          // printf("iter %u wrap %u lane %u vertex %u status %u loc %u score %f
          // loc %u \n",iter,warpId,laneId,T.doms[writeOffset +
          // loc],status,loc,T.cons[writeOffset + loc],writeOffset + loc);
        }
      }
    }
    __syncwarp();

    if ((sharedCounter[threadIdx.x / 32] > 1) &&
        (T.ustar[otherStartIndex + iter] != INT_MAX) &&
        (T.ustar[otherStartIndex + iter] != -1) && ( queryId != UINT_MAX) && (limitDoms > 0)) {
      warpSelectionSort(T.cons, T.doms, startIndex + start,
                        startIndex + start + sharedCounter[threadIdx.x / 32],
                        laneId);
    }
    __syncwarp();
    if ((laneId == 0) && (T.ustar[otherStartIndex + iter] != INT_MAX) &&
        (T.ustar[otherStartIndex+ iter] != -1) && ( queryId != UINT_MAX) && (limitDoms > 0) ) {
      T.doms[startIndex + end - 1] =
          (sharedCounter[threadIdx.x / 32] > limitDoms)
              ? limitDoms
              : sharedCounter[threadIdx.x / 32];
      sharedCounter[threadIdx.x / 32] = 0;
    }
  }
}


__global__ void RemoveCompletedTasks( deviceGraphPointers G_,deviceTaskPointers T, ui pSize, ui factor ){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  ui otherPsize = *T.limitTasks;

  ui otherStartIndex = warpId * otherPsize;
  ui totalTasks = T.numTasks[warpId];
  __syncwarp();

  for (ui iter = laneId; iter < totalTasks; iter+=warpSize) {
    __syncwarp();
    ui queryId = T.queryIndicator[otherStartIndex + iter];
    if(queryId != UINT_MAX){
      ui stopflag = G_.flag[queryId];
      if(stopflag==0){
        T.queryIndicator[otherStartIndex + iter] = UINT_MAX;
      }
      }
  }
   if(laneId==0){
     ui offsetPsize = pSize/factor;

     ui offsetStartIndex = warpId*offsetPsize;
     T.sortedIndex[warpId] = T.taskOffset[offsetStartIndex + totalTasks];
  }
}