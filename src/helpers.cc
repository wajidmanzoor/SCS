__device__ ui minn(ui a, ui b) {
  return (a < b) ? a : b;
}

__device__ int findIndexKernel(ui * arr, ui start, ui end, ui target)

{
  // Perform a linear search to find the index of the target vertex in the task array.
  // 'start' and 'end' are the indices of the task array defining the search range.
  // 'target' is the vertex we are searching for.

  int resultIndex = -1;
  for (ui index = start; index < end; index++) {
    if (arr[index] == target) {
      resultIndex = index;
      break;
    }
  }
  return resultIndex;

}

__global__ void IntialReductionRules(deviceGraphPointers G, deviceInterPointers P, ui size, ui upperBoundSize, ui lowerBoundDegree, ui pSize) {

  // Intial Reduction rule based on core value and distance.
  extern __shared__ ui shared_memory1[];

  // Store the counter
  ui * local_counter = shared_memory1;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui start = warpId * pSize;
  ui end = (warpId + 1) * pSize;
  ui writeOffset = warpId * pSize;
  if (end > size) {
    end = size;
  }
  if (start > size) {
    start = size;
  }
  ui total = end - start;

  if (laneId == 0) {
    local_counter[threadIdx.x / 32] = 0;
    //printf("wrapId %d start %u end %u  \n",warpId,start,end);

  }
  __syncwarp();

  for (ui i = laneId; i < total; i += 32) {

    ui vertex = start + i;
    if ((G.core[vertex] > lowerBoundDegree) && (G.distance[vertex] < (upperBoundSize - 1))) {
      ui loc = atomicAdd( & local_counter[threadIdx.x / 32], 1);
      P.intialTaskList[loc + writeOffset] = vertex;
      //printf("block id %u wrapId %u lane id %u vertex %u degree %u \n",blockIdx.x,warpId,i,vertex,G.degree[vertex]);

    } else {
      G.degree[vertex] = 0;
    }

  }

  __syncwarp();
  if (laneId == 0) {
    P.entries[warpId] = local_counter[threadIdx.x / 32];

    atomicAdd(P.globalCounter, local_counter[threadIdx.x / 32]);
    //printf("block %d wrapId %d entries %u g %u \n",blockIdx.x,warpId,P.entries[warpId],*(P.globalCounter));

  }

}

__global__ void CompressTask(deviceGraphPointers G, deviceInterPointers P, deviceTaskPointers T, ui pSize, ui queryVertex) {

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui start = warpId * pSize;
  int temp = warpId - 1;
  ui total = P.entries[warpId];
  ui writeOffset = 0;
  if (idx == 0) {
    T.size[0] = 1;
  }

  for (ui i = laneId; i < total; i += 32) {

    while (temp >= 0) {
      writeOffset += P.entries[temp];
      temp--;
    }
    int vertex = P.intialTaskList[start + i];
    T.taskList[writeOffset + i] = vertex;
    T.statusList[writeOffset + i] = (vertex == queryVertex) ? 1 : 0;
    ui nStart = G.offset[vertex];
    ui nEnd = G.offset[vertex + 1];
    ui degInR = 0;
    ui degInc = 0;
    for (ui k = nStart; k < nEnd; k++) {
      if ((G.neighbors[k] != queryVertex) && (G.degree[G.neighbors[k]] != 0)) {
        degInR++;
      }
      if (G.neighbors[k] == queryVertex) {
        degInc++;

      }

    }

    T.degreeInR[writeOffset + i] = degInR;
    T.degreeInC[writeOffset + i] = degInc;
    //printf("warp id %u lane id %u vertex %u status %u degree %u dR %u dC %u \n",warpId,i,T.taskList[writeOffset+i],T.statusList[writeOffset+i],G.degree[vertex], T.degreeInR[writeOffset+i],T.degreeInC[writeOffset+i]);

  }

}

__global__ void ProcessTask(deviceGraphPointers G, deviceTaskPointers T, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax) {
  extern __shared__ char shared_memory[];
  ui sizeOffset = 0;
  // Stores new tasks
  ui * sharedUBDegree = (ui * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  ui * sharedDegree = (ui * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int * sharedUstar = (int * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(int);

  double * sharedScore = (double * )(shared_memory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(double);

  // minimum degree intialized to zero.
  ui currentMinDegree;

  // Connection score used to calculate the ustar.
  double score;
  int ustar;

  double temp;
  ui temp2;
  ui temp3;
  ui degreeBasedUpperBound;

  int otherId;

  int resultIndex;
  ui currentSize;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  if (laneId == 0) {
    sharedUBDegree[threadIdx.x / 32] = UINT_MAX;
    sharedScore[threadIdx.x / 32] = 0;
    sharedUstar[threadIdx.x / 32] = -1;
    sharedDegree[threadIdx.x / 32] = UINT_MAX;

  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();

    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;
    //printf("outside");

    ui maskGen = total;
    if (T.ustar[warpId * pSize + iter] != -2) {
      for (ui i = laneId; i < total; i += 32) {
        int index = startIndex + start + i;
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];


        ui startNeighbor = G.offset[vertex];
        ui endNeighbor = G.offset[vertex + 1];

        score = 0;
        currentMinDegree = UINT_MAX;
        degreeBasedUpperBound = UINT_MAX;

        ustar = -1;

        if (status == 0) {
          for (int j = startNeighbor; j < endNeighbor; j++) {
            resultIndex = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[j]);
            if (resultIndex != -1) {
              if (T.statusList[resultIndex] == 1) {
                if (T.degreeInC[resultIndex] != 0) {
                  score += (double) 1 / T.degreeInC[resultIndex];

                } else {
                  score += 1;
                }
                // printf("Iter %u warp %u lane id % u neighbor %u found %u status %u degof neighbor %u score %f \n ",iter,warpId,i,G.neighbors[j],T.taskList[resultIndex],T.statusList[resultIndex],T.degreeInC[resultIndex],score);

              }
            }
          }
        }
        if (score > 0) {
          score += (double) T.degreeInR[vertex] / dmax;
        }

        unsigned int mask;
        ui cond;
        if (maskGen > 32) {
          mask = (1u << 31) - 1;
          maskGen -= 32;
          cond = 32;

        } else {
          mask = (1u << maskGen) - 1;
          cond = maskGen;

        }

        for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
          temp = __shfl_down_sync(mask, score, offset);
          otherId = __shfl_down_sync(mask, index, offset);

          if (temp > score && temp > 0 && (laneId + offset < cond)) {
            score = temp;
            index = otherId;
          }
        }
        ustar = __shfl_sync(mask, index, 0);
        score = __shfl_sync(mask, score, 0);

        if (status == 1) {
          currentMinDegree = T.degreeInC[ind];
        }

        for (int offset = 16; offset > 0; offset /= 2) {
          temp2 = __shfl_down_sync(mask, currentMinDegree, offset);

          if (laneId + offset < cond) {
            currentMinDegree = min(currentMinDegree, temp2);
          }
        }

        currentMinDegree = __shfl_sync(mask, currentMinDegree, 0);
        ui oneside;
        if (status == 1) {
          oneside = T.degreeInC[ind] + upperBoundSize - T.size[startIndex + iter] - 1;
          degreeBasedUpperBound = minn(oneside, T.degreeInR[ind] + T.degreeInC[ind]);
        }

        for (int offset = 16; offset > 0; offset /= 2) {
          temp3 = __shfl_down_sync(mask, degreeBasedUpperBound, offset);

          if (laneId + offset < cond) {
            degreeBasedUpperBound = min(degreeBasedUpperBound, temp3);
          }
        }
        degreeBasedUpperBound = __shfl_sync(mask, degreeBasedUpperBound, 0);
        /*if(status==1){
        printf("iter %u warp %u lane %u vertex %u status %u Dc %u dr %u oneside %u upper degree %u \n",iter, warpId,i,vertex,status,T.degreeInC[ind],T.degreeInR[ind],oneside,degreeBasedUpperBound);
        }*/
        //printf(" Process Task : iter %u wrapId %u lane id %u  vertex %u status %u dc %u dr %u size %u min degree %u ubd %u global %u ustar %d score %f \n", iter, warpId, i, vertex, status, T.degreeInC[ind], T.degreeInR[ind],T.size[warpId * pSize + iter],currentMinDegree,degreeBasedUpperBound,*G.lowerBoundDegree,T.taskList[ustar],score);


        if (i % 32 == 0) {
          if (currentMinDegree < sharedDegree[threadIdx.x / 32]) {
            sharedDegree[threadIdx.x / 32] = currentMinDegree;
          }
          if (score > sharedScore[threadIdx.x / 32]) {

            sharedScore[threadIdx.x / 32] = score;
            sharedUstar[threadIdx.x / 32] = ustar;
          }
          if (degreeBasedUpperBound < sharedUBDegree[threadIdx.x / 32]) {
            sharedUBDegree[threadIdx.x / 32] = degreeBasedUpperBound;
          }

        //printf(" Process Task : iter %u wrapId %u lane id %u  vertex %u status %u dc %u dr %u size %u min degree %u score %f ubd %u \n", iter, warpId, i, vertex, status, T.degreeInC[ind], T.degreeInR[ind],T.size[warpId * pSize + iter],sharedDegree[threadIdx.x / 32],sharedScore[threadIdx.x / 32],sharedUBDegree[threadIdx.x / 32]);


        }

      }
      __syncwarp();

      if (laneId == 0) {
        currentSize = T.size[startIndex + iter];

        if ((lowerBoundSize <= currentSize) && (currentSize <= upperBoundSize)) {
          if (sharedDegree[threadIdx.x / 32] != UINT_MAX) {


            ui oldvalue = atomicMax(G.lowerBoundDegree, sharedDegree[threadIdx.x / 32]);
        //printf(" Process update : iter %u wrapId %u lane id %u  size %u min degree %u globa %u score %f ubd %u \n", iter, warpId, laneId,currentSize,sharedDegree[threadIdx.x / 32],*G.lowerBoundDegree,sharedScore[threadIdx.x / 32],sharedUBDegree[threadIdx.x / 32]);

          }
        }
        int writeOffset = warpId * pSize;
        //printf("iter %u warp %u shared %u G.lower %u \n",iter, warpId,sharedUBDegree[threadIdx.x / 32],*G.lowerBoundDegree);
        if ((sharedScore[threadIdx.x / 32] > 0) && (sharedUBDegree[threadIdx.x / 32] > * G.lowerBoundDegree) && (sharedUBDegree[threadIdx.x / 32] != UINT_MAX) && (currentSize < upperBoundSize)) {
          T.ustar[writeOffset + iter] = sharedUstar[threadIdx.x / 32];
          //printf("iter %u wrap id %u  shared ustar %d ustar %d score %f \n",iter,warpId,sharedUstar[threadIdx.x / 32],T.ustar[writeOffset + iter],sharedScore[threadIdx.x / 32]);

        } else {
          T.ustar[writeOffset + iter] = -2;
          //printf("other iter %u wrap id %u lane id %u ustar %d score %f \n",iter,warpId,laneId,T.ustar[writeOffset + iter],sharedScore[threadIdx.x / 32]);

        }
        sharedUBDegree[threadIdx.x / 32] = UINT_MAX;
        sharedScore[threadIdx.x / 32] = 0;
        sharedUstar[threadIdx.x / 32] = -1;
        sharedDegree[threadIdx.x / 32] = UINT_MAX;

      }
    }

    //__syncwarp();

  }
}
__global__ void Expand(deviceGraphPointers G, deviceTaskPointers T, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax, ui jump) {

  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui * sharedCounter = (ui * )(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];

  if (laneId == 0) {
    sharedCounter[threadIdx.x / 32] = 0;
  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;
    if ((T.ustar[warpId * pSize + iter] != -2) && (T.ustar[warpId * pSize + iter] != -1)) {
      for (ui i = laneId; i < total; i += 32) {

        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];

        if ((T.size[warpId * pSize + iter] < upperBoundSize) && (T.ustar[warpId * pSize + iter] != -1) && (T.ustar[warpId * pSize + iter] != -2)) {
          ui bufferNum = warpId + jump;
          if (bufferNum > TOTAL_WARPS) {
            bufferNum = bufferNum % TOTAL_WARPS;
          }
          ui totalTasksWrite = T.taskOffset[bufferNum * pSize - 1];
          ui writeOffset = ((bufferNum - 1) * pSize) + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite];

          ui ustar = T.taskList[T.ustar[warpId * pSize + iter]];
          ui degInR;
          ui degInC;
          if ((vertex != ustar) && (status != 2)) {
            ui loc = atomicAdd( & sharedCounter[threadIdx.x / 32], 1);
            T.taskList[writeOffset + loc] = vertex;
            T.statusList[writeOffset + loc] = status;
            degInR = T.degreeInR[ind];
            degInC = T.degreeInC[ind];
            T.degreeInR[writeOffset + loc] = (degInR != 4294967295) ? degInR : 0;
            T.degreeInR[ind] = (degInR != 4294967295) ? degInR : 0;
            T.degreeInC[writeOffset + loc] = degInC;

          }

        }

      }
      if (laneId == 0) {
        if ((T.size[warpId * pSize + iter] < upperBoundSize) && (T.ustar[warpId * pSize + iter] != -1)) {
          //printf("iter %u wrap id %u ustar %d \n",iter, warpId, T.ustar[warpId * pSize + iter]);
          *(T.flag) = 0;
          ui bufferNum = warpId + jump;
          if (bufferNum > TOTAL_WARPS) {
            bufferNum = bufferNum % TOTAL_WARPS;
          }
          ui totalTasksWrite = T.taskOffset[bufferNum * pSize - 1];
          T.statusList[T.ustar[warpId * pSize + iter]] = 1;

          T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite + 1] = T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite] + sharedCounter[threadIdx.x / 32];
          T.taskOffset[bufferNum * pSize - 1]++;
          T.size[(bufferNum - 1) * pSize + totalTasksWrite] = T.size[warpId * pSize + iter];
          T.size[warpId * pSize + iter] += 1;
        }

        sharedCounter[threadIdx.x / 32] = 0;
      }
    }

  }

}

__global__ void DegreeUpdate(deviceGraphPointers G, deviceTaskPointers T, ui pSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  int resultIndex;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;
    if ((T.ustar[warpId * pSize + iter] != -2)) {
      for (ui i = laneId; i < total; i += 32) {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];

        ui startNeighbor = G.offset[vertex];
        ui endNeighbor = G.offset[vertex + 1];
        ui degR = 0;
        ui degC = 0;
        int key;
        if ((status == 0) || (status == 1)) {
          for (ui j = startNeighbor; j < endNeighbor; j++) {
            key = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[j]);
            if (key != -1) {
              if (T.statusList[key] == 0) {
                degR++;
              }
              if (T.statusList[key] == 1) {
                degC++;
              }

            }

          }
        }

        T.degreeInR[ind] = degR;
        T.degreeInC[ind] = degC;

      }

    }
  }

}

__global__ void ReductionRules(deviceGraphPointers G, deviceTaskPointers T, ui pSize, ui upperBoundSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  int resultIndex;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;
    if ((T.ustar[warpId * pSize + iter] != -2)) {
      for (ui i = laneId; i < total; i += 32)

      {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];
        ui degR = T.degreeInR[ind];
        ui degC = T.degreeInC[ind];
        ui hSize = T.size[startIndex + iter];
        ui startNeighbor = G.offset[vertex];
        ui endNeighbor = G.offset[vertex + 1];
        ui ubD;
        ui kl = * G.lowerBoundDegree;
        int resKey;
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

        if (status == 0) {
          if ((minn((degR + degC), (degC + upperBoundSize - hSize - 1)) <= * G.lowerBoundDegree) || (ubD < G.distance[vertex])) {
            T.statusList[ind] = 2;
            T.degreeInR[ind] = 0;
            T.degreeInC[ind] = 0;

          }
        }
        ui neig;
        int resInd;
        if ((status == 1) && ((degC + degR) == ( * G.lowerBoundDegree + 1))) {
          for (int j = startNeighbor; j < endNeighbor; j++) {
            resKey = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[j]);
            if (resKey != -1) {
              if (T.statusList[resKey] == 0) {

                T.statusList[resKey] = 1;
                atomicAdd( & T.size[startIndex + iter], 1);

              }
            }

          }

        }

      }

    }
  }

}


__global__ void FindDoms(deviceGraphPointers G, deviceTaskPointers T, ui pSize, ui dmax) {
  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui * sharedCounter = (ui * )(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;
  int resultIndex;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  if (laneId == 0) {
    sharedCounter[threadIdx.x / 32] = 0;
  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;
    if ((T.ustar[warpId * pSize + iter] != -2) && (T.ustar[warpId * pSize + iter] != -1)) {
      ui ustar = T.taskList[T.ustar[warpId * pSize + iter]];
      for (ui i = laneId; i < total; i += 32) {

        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];
        ui degC = T.degreeInC[ind];

        ui startNeighbor = G.offset[vertex];
        ui endNeighbor = G.offset[vertex + 1];
        bool is_doms = false;
        double score = 0;
        if ((status == 0) && (vertex != ustar) && (degC!=0)) {
          is_doms = true;
          ui neighbor;
          for (int j = startNeighbor; j < endNeighbor; j++) {
            neighbor = G.neighbors[j];

            bool found = false;
            if (neighbor != ustar) {
              for (ui k = G.offset[ustar]; k < G.offset[ustar + 1]; k++) {
                if (neighbor == G.neighbors[k]) {
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


            resultIndex = findIndexKernel(T.taskList, startIndex + start, startIndex + end, neighbor);
            if (resultIndex != -1) {
              if (T.statusList[resultIndex] == 1) {
                if (T.degreeInC[resultIndex] != 0) {
                  score += (double) 1 / T.degreeInC[resultIndex];

                } else {
                  score += 1;
                }
              }

            }
        //printf("iter %u wrap %u lane %u vertex %u status %u neighbor %u  index %d score %f \n",iter,warpId,i,vertex,status,neighbor,resultIndex,score);

          }
          score += (double) T.degreeInR[ind] / dmax;
        }
        //printf("iter %u wrap %u lane %u vertex %u status %u ustar %u doms %u  \n",iter,warpId,i,vertex,status,ustar, is_doms);

        if (is_doms) {
          ui writeOffset = startIndex + start;
          ui loc = atomicAdd( &sharedCounter[threadIdx.x / 32], 1);
          T.doms[writeOffset + loc] = vertex;
          T.cons[writeOffset + loc] = score;
          //printf("iter %u wrap %u lane %u vertex %u status %u loc %u score %f\n",iter,warpId,laneId,vertex,status,loc,T.cons[writeOffset + loc]);

        }

      }
      __syncwarp();
      if (laneId == 0) {
        T.doms[startIndex + end - 1] = sharedCounter[threadIdx.x / 32];
        if( sharedCounter[threadIdx.x / 32]>0){
          printf("Iter %u wrap %u total doms %u shard doms %u ind %u \n",iter,warpId,T.doms[startIndex + end - 1],sharedCounter[threadIdx.x / 32],startIndex + end - 1);
        }
        sharedCounter[threadIdx.x / 32] = 0;

      }

    }
  }

}

__global__ void ExpandNew(deviceGraphPointers G, deviceTaskPointers T, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax, ui jump) {

  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui * sharedCounter = (ui * )(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];

  if (laneId == 0) {
    sharedCounter[threadIdx.x / 32] = 0;
  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;
    if ((T.ustar[warpId * pSize + iter] != -2) && (T.ustar[warpId * pSize + iter] != -1)) {
      for (ui i = laneId; i < total; i += 32) {

        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];

        if ((T.size[warpId * pSize + iter] < upperBoundSize) && (T.ustar[warpId * pSize + iter] != -1) && (T.ustar[warpId * pSize + iter] != -2)) {
          ui bufferNum = warpId + jump;
          if (bufferNum > TOTAL_WARPS) {
            bufferNum = bufferNum % TOTAL_WARPS;
          }
          ui totalTasksWrite = T.taskOffset[bufferNum * pSize - 1];
          ui writeOffset = ((bufferNum - 1) * pSize) + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite];

          ui ustar = T.taskList[T.ustar[warpId * pSize + iter]];
          ui degInR;
          ui degInC;
          if ((vertex != ustar) && (status != 2)) {
            ui loc = atomicAdd( & sharedCounter[threadIdx.x / 32], 1);
            T.taskList[writeOffset + loc] = vertex;
            T.statusList[writeOffset + loc] = status;
            degInR = T.degreeInR[ind];
            degInC = T.degreeInC[ind];
            T.degreeInR[writeOffset + loc] = (degInR != 4294967295) ? degInR : 0;
            T.degreeInR[ind] = (degInR != 4294967295) ? degInR : 0;
            T.degreeInC[writeOffset + loc] = degInC;
            //printf(" first %u ",degInR);

          }

        }

      }
      __syncwarp();
      if (laneId == 0) {
        if ((T.size[warpId * pSize + iter] < upperBoundSize) && (T.ustar[warpId * pSize + iter] != -1)) {
          //printf("iter %u wrap id %u ustar %d \n",iter, warpId, T.ustar[warpId * pSize + iter]);
          *(T.flag) = 0;
          ui bufferNum = warpId + jump;
          if (bufferNum > TOTAL_WARPS) {
            bufferNum = bufferNum % TOTAL_WARPS;
          }
          ui totalTasksWrite = T.taskOffset[bufferNum * pSize - 1];
          T.statusList[T.ustar[warpId * pSize + iter]] = 1;
          T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite + 1] = T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite] + sharedCounter[threadIdx.x / 32];
          T.taskOffset[bufferNum * pSize - 1]++;
          T.size[(bufferNum - 1) * pSize + totalTasksWrite] = T.size[warpId * pSize + iter];
          T.size[warpId * pSize + iter] += 1;
          if(T.doms[warpId*pSize + end -1 ]!=0){
          printf("expand new : iter %u wrap %u doms %u index %u write in wrap % u \n",iter, warpId,T.doms[warpId*pSize + end -1 ],warpId*pSize + end -1,bufferNum-1);
          T.ustar[(bufferNum - 1) * pSize + totalTasksWrite] = -3;
          T.doms[(bufferNum - 1) * pSize + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite ]] = warpId;
          T.doms[(bufferNum - 1) * pSize + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite ] + 1] = iter;
          //T.doms[(bufferNum - 1) * pSize + totalTasksWrite + 2] = sharedCounter[threadIdx.x / 32] + 1;
          //printf("ustar %d ", T.ustar[(bufferNum - 1) * pSize + totalTasksWrite]);
          //printf("Total  %d ", T.taskOffset[bufferNum * pSize - 1]);
          }

        }

        sharedCounter[threadIdx.x / 32] = 0;
      }

    }

  }

}

__global__ void ExpandDoms(deviceGraphPointers G, deviceTaskPointers T, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax, ui jump) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  //printf("totalTasks %u ",totalTasks);

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;

    if (T.ustar[warpId * pSize + iter] == -3) {

      ui prevWrap = T.doms[startIndex + start];
      ui prevIter = T.doms[startIndex + start + 1];
      ui domStart = T.taskOffset[prevWrap * pSize + prevIter];
      ui totalDoms = T.doms[prevWrap * pSize + T.taskOffset[prevWrap * pSize + prevIter + 1] - 1];
      ui prevUstar = T.taskList[T.ustar[prevWrap * pSize + prevIter]];
      ui newWriteOffset = startIndex + T.taskOffset[startIndex+totalTasks];
      ui totalWrite = total + 1;
      //if(laneId==0)
      //printf("wrap id %u lane id %u totalTasks %u ustar %d total %u ustar prev %u domsstart %u total doms %u prev iter %u prev wrap %u \n", warpId, laneId, totalTasks, T.ustar[warpId * pSize + iter], total, prevUstar, domStart, totalDoms,prevIter,prevWrap);
      ui maskGen = total;
      for (ui i = laneId; i < total; i += 32) {
        //printf("Inside");
        ui ind = laneId;
        ui index = startIndex + start + laneId;
        //printf("Inside ");


        ui vertex = T.taskList[index];
        //printf("inside %u \n", index);

        ui status = T.statusList[index];
        int processed = 1;

        ui degR = T.degreeInR[index];
        ui degC = T.degreeInC[index];
        ui currentDomIndex = 0;
        ui domVertex;
        //printf("vertex %u \n",vertex);
        //printf("iter %u wrap %u lane %u vertex %u status %u ustar %u doms \n", iter, warpId, i, vertex, status, prevUstar);

        while (currentDomIndex < totalDoms) {
          domVertex = T.doms[domStart + currentDomIndex];
          T.taskList[newWriteOffset + (totalWrite * currentDomIndex) + ind] = vertex;
          int key = findIndexKernel(T.doms, domStart, domStart + currentDomIndex, vertex);
          if (key != -1) {
            T.statusList[newWriteOffset + (totalWrite * currentDomIndex) + ind] = 2;

          } else {
            T.statusList[newWriteOffset + (totalWrite * currentDomIndex) + ind] = status;
          }
          if (vertex == domVertex) {
            T.statusList[newWriteOffset + (totalWrite * currentDomIndex) + ind] = 1;
          }
          T.degreeInR[newWriteOffset + (totalWrite * currentDomIndex) + ind] = degR;
          T.degreeInC[newWriteOffset + (totalWrite * currentDomIndex) + ind] = degC;
          //printf("iter %u wrap %u lane %u vertex %u status %u ustar %u level %u writeLoc %u doms %u vertex %u status %u \n", iter, warpId, i, vertex, status, prevUstar, currentDomIndex,
            //newWriteOffset + (totalWrite * currentDomIndex) + ind, domVertex, T.taskList[newWriteOffset + (totalWrite * currentDomIndex) + ind], T.statusList[newWriteOffset + (totalWrite * currentDomIndex) + ind]);

          currentDomIndex++;

        }

        unsigned int mask;
        if (maskGen > 32) {
          mask = (1u << 31) - 1;
          maskGen -= 32;

        } else {
          mask = (1u << maskGen) - 1;

        }

        processed =  __shfl_sync(mask, processed, 0);

      }

      __syncwarp();
      
      if(laneId ==0){
        ui current = 1;
        while(current<=totalDoms){
          T.taskList[newWriteOffset+ (totalWrite*current)-1] = prevUstar;
          T.statusList[newWriteOffset+ (totalWrite*current)-1] = 1;
          T.size[warpId * pSize + totalTasks+current-1] = T.size[warpId * pSize + iter]+2;
          T.taskOffset[warpId* pSize + totalTasks +current] = T.taskOffset[ warpId* pSize + totalTasks ] +(current*totalWrite);
          T.taskOffset[(warpId+1) * pSize - 1]++;
          current++;


        }
        T.ustar[warpId * pSize + iter]=-1;


      }
      //__syncwarp();




    }

  }
}