__device__ ui minn(ui a, ui b) {
  return (a < b) ? a : b;
}

__global__ void increment_if_less(int * a, int b) {
  int old_val, new_val;

  do {
    old_val = atomicCAS(a, 0, 0); // Load the current value of a atomically
    if (old_val >= b) {
      return; // If a is not less than b, exit
    }
    new_val = old_val + 1; // Calculate the new value
  } while (atomicCAS(a, old_val, new_val) != old_val); // Attempt to update a if it hasn't changed
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

__device__ void warpSelectionSort(double *keys, ui *values, ui start, ui end, ui laneId) {
    int n = end - start + 1;
    //int laneId = threadIdx.x & 31;  // equivalent to threadIdx.x % 32
    
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
__global__ void IntialReductionRules(deviceGraphPointers G, deviceInterPointers P, ui size, ui upperBoundSize, ui lowerBoundDegree, ui pSize) {

  // Intial Reduction rule based on core value and distance.
  extern __shared__ ui shared_memory1[];

  // Store the counter
  ui * local_counter = shared_memory1;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui start = minn(warpId * pSize, size);
  ui end = minn((warpId + 1) * pSize, size);

  ui writeOffset = start;
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
      P.initialTaskList[loc + writeOffset] = vertex;
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
    int vertex = P.initialTaskList[start + i];
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

__global__ void ProcessTask(deviceGraphPointers G, deviceTaskPointers T, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax, ui *result) { 

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

  // minimum degree initialized to zero.
  ui currentMinDegree;

  // Connection score used to calculate the ustar.
  double score;
  int ustar;

  double temp;
  ui temp2;
  ui temp3;
  ui degreeBasedUpperBound;

  int otherId;

  int resultIndex,resInd;
  ui currentSize;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  if (laneId == 0) {
    sharedUBDegree[threadIdx.x / warpSize] = UINT_MAX;
    sharedScore[threadIdx.x / warpSize] = 0;
    sharedUstar[threadIdx.x / warpSize] = -1;
    sharedDegree[threadIdx.x / warpSize] = UINT_MAX;

  }
  __syncwarp();

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();

    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;

    ui maskGen = total;
    if (T.ustar[warpId * pSize + iter] != INT_MAX) {
            for (ui i = laneId; i < total; i += warpSize)

      {
        ui ind = startIndex + start + i;
        ui vertex = T.taskList[ind];
        ui status = T.statusList[ind];
        ui degR = T.degreeInR[ind];
        ui degC = T.degreeInC[ind];
        ui hSize = T.size[startIndex + iter];
        ui startNeighbor = G.offset[vertex];
        ui endNeighbor = G.offset[vertex + 1];
        ui ubD = upperBoundSize -1 ;
        ui kl = * G.lowerBoundDegree;

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
        ui cond;

        if (maskGen > warpSize) {
          mask = 0xFFFFFFFF;
          maskGen -= warpSize;
          cond = warpSize;
          //printf("here");

        } else {
          mask = (1u << maskGen) - 1;
          cond = maskGen;
          //printf("2nd here");

        }
        __syncwarp(mask);



        if (status == 0) {
          if ((minn((degR + degC), (degC + upperBoundSize - hSize - 1)) <= * G.lowerBoundDegree) || (ubD < G.distance[vertex])) {
            T.statusList[ind] = 2;
            T.degreeInR[ind] = 0;
            T.degreeInC[ind] = 0;
            for (int j = startNeighbor; j < endNeighbor; j++) {

              resultIndex = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[j]);
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
        if ((status == 1) && ((degC + degR) == ( * G.lowerBoundDegree + 1))) {
           // printf("R3 iter %u wrap %u lane %u index %u vertex %u \n", iter,warpId,laneId,i,vertex);

          for (int j = startNeighbor; j < endNeighbor; j++) {
            resultIndex = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[j]);
            if (resultIndex != -1) {

              if ((atomicCAS( &T.statusList[resultIndex], 0, 1) == 0) && (atomicAdd(&T.size[startIndex + iter], 0) <upperBoundSize)){

                atomicAdd( &T.size[startIndex + iter], 1);
                neig = T.taskList[resultIndex];

                for (int k = G.offset[neig]; k < G.offset[neig + 1]; k++) {
                  resInd = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[k]);
                  if (resInd != -1) {
                    atomicAdd( &T.degreeInC[resInd], 1);
                    if (T.degreeInR[resInd] != 0) {
                      atomicSub( &T.degreeInR[resInd], 1);
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
      maskGen = total;
      for (ui i = laneId; i < total; i += warpSize) {
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

                }else{
                  score +=1;
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
          //printf("here");

        } else {
          mask = (1u << maskGen) - 1;
          cond = maskGen;
          //printf("2nd here");

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


        if (status == 1) {
          currentMinDegree = T.degreeInC[ind];
        }
       // if(warpId==1939)
        //printf("iter %u warpId %u laneId %u i %u vertex %u status %u degree %u mask %u mask gen %u cond %u \n",iter,warpId,laneId,i,vertex,status,currentMinDegree,mask,maskGen,cond);

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          temp2 = __shfl_down_sync(mask, currentMinDegree, offset);

          if (laneId + offset < cond) {
            currentMinDegree = minn(currentMinDegree, temp2);
          }
        }

        currentMinDegree = __shfl_sync(mask, currentMinDegree, 0);
        ui oneside;
        if (status == 1) {
          oneside = T.degreeInC[ind] + upperBoundSize - T.size[startIndex + iter] - 1;
          degreeBasedUpperBound = minn(oneside, T.degreeInR[ind] + T.degreeInC[ind]);
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          temp3 = __shfl_down_sync(mask, degreeBasedUpperBound, offset);

          if (laneId + offset < cond) {
            degreeBasedUpperBound = min(degreeBasedUpperBound, temp3);
          }
        }
        degreeBasedUpperBound = __shfl_sync(mask, degreeBasedUpperBound, 0);

        if (i % warpSize == 0) {
          if (currentMinDegree < sharedDegree[threadIdx.x / warpSize]) {
            sharedDegree[threadIdx.x / warpSize] = currentMinDegree;
          }
          if (score > sharedScore[threadIdx.x / warpSize]) {

            sharedScore[threadIdx.x / warpSize] = score;
            sharedUstar[threadIdx.x / warpSize] = ustar;
          }
          if (degreeBasedUpperBound < sharedUBDegree[threadIdx.x / warpSize]) {
            sharedUBDegree[threadIdx.x / warpSize] = degreeBasedUpperBound;
          }

        }

      }

      __syncwarp();

      if (laneId == 0) {
        currentSize = T.size[startIndex + iter];
        if ((lowerBoundSize <= currentSize) && (currentSize <= upperBoundSize)) {
          if (sharedDegree[threadIdx.x / warpSize] != UINT_MAX) {
            ui old = atomicMax(G.lowerBoundDegree, sharedDegree[threadIdx.x / warpSize]);
            
            if(old<sharedDegree[threadIdx.x / warpSize]){
              result[0] = iter;
            result[1] = warpId;
            printf("kl %u iter %u warp %u new kl %u \n",old, iter, warpId,sharedDegree[threadIdx.x / warpSize]);

            }
          }
        }
        int writeOffset = warpId * pSize;
        if ((sharedScore[threadIdx.x / warpSize] > 0) && (sharedUBDegree[threadIdx.x / warpSize] > * G.lowerBoundDegree) && (sharedUBDegree[threadIdx.x / warpSize] != UINT_MAX) && (total >= lowerBoundSize)) {
          T.ustar[writeOffset + iter] = sharedUstar[threadIdx.x / warpSize];

        } else {
          T.ustar[writeOffset + iter] = INT_MAX;

        }
        sharedUBDegree[threadIdx.x / warpSize] = UINT_MAX;
        sharedScore[threadIdx.x / warpSize] = 0;
        sharedUstar[threadIdx.x / warpSize] = -1;
        sharedDegree[threadIdx.x / warpSize] = UINT_MAX;

      }
    }

  }
}

__global__ void FindDoms(deviceGraphPointers G, deviceTaskPointers T, ui pSize, ui dmax, ui level, ui limitDoms) {
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
    ui writeOffset = startIndex + start;

    if ((T.ustar[warpId * pSize + iter] != INT_MAX) && (T.ustar[warpId * pSize + iter] != -1)) {
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

        if ((status == 0) && (vertex != ustar) && (degC != 0)) {
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

                }
              }

            }
            //printf("iter %u wrap %u lane %u vertex %u status %u neighbor %u  index %d score %f \n",iter,warpId,i,vertex,status,neighbor,resultIndex,score);

          }
          score += (double) T.degreeInR[ind] / dmax;

        }
        //printf("iter %u wrap %u lane %u vertex %u status %u ustar %u doms %u  \n",iter,warpId,i,vertex,status,ustar, is_doms);

        if (is_doms) {

          ui loc = atomicAdd( & sharedCounter[threadIdx.x / 32], 1);
          T.doms[writeOffset + loc] = vertex;
          T.cons[writeOffset + loc] = score;
          //printf("iter %u wrap %u lane %u vertex %u status %u loc %u score %f loc %u \n",iter,warpId,laneId,T.doms[writeOffset + loc],status,loc,T.cons[writeOffset + loc],writeOffset + loc);

        }

      }
    }
    __syncwarp();

    if ((sharedCounter[threadIdx.x / 32] > 1) && (T.ustar[warpId * pSize + iter] != INT_MAX) && (T.ustar[warpId * pSize + iter] != -1)) {

      warpSelectionSort(T.cons, T.doms, startIndex + start, startIndex + start + sharedCounter[threadIdx.x / 32],laneId);

    }
    __syncwarp();
    if ((laneId == 0) && (T.ustar[warpId * pSize + iter] != INT_MAX) && (T.ustar[warpId * pSize + iter] != -1)) {
      T.doms[startIndex + end - 1] = (sharedCounter[threadIdx.x / 32] > limitDoms) ? limitDoms : sharedCounter[threadIdx.x / 32];
      sharedCounter[threadIdx.x / 32] = 0;

    }

  }

}

__global__ void ExpandNew(deviceGraphPointers G, deviceTaskPointers T, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax, ui jump, ui level) {
  extern __shared__ char sharedMemory[];
  size_t sizeOffset = 0;

  ui * sharedCounter = (ui * )(sharedMemory + sizeOffset);
  sizeOffset += WARPS_EACH_BLK * sizeof(ui);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / warpSize;
  int laneId = idx % warpSize;

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

    if ((T.ustar[warpId * pSize + iter] != INT_MAX) && (T.ustar[warpId * pSize + iter] != -1)) {
      ui bufferNum = warpId + jump;
      if (bufferNum > TOTAL_WARPS) {
        bufferNum = bufferNum % TOTAL_WARPS;
      }
      ui totalTasksWrite = T.taskOffset[bufferNum * pSize - 1];
      ui writeOffset = ((bufferNum - 1) * pSize) + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite];
      ui ustar = T.taskList[T.ustar[warpId * pSize + iter]];

      if ((writeOffset + total) <= (bufferNum * pSize - 1)) {
        for (ui i = laneId; i < total; i += 32) {

          ui ind = startIndex + start + i;
          ui vertex = T.taskList[ind];
          ui status = T.statusList[ind];
          ui startNeighbor = G.offset[vertex];
          ui endNeighbor = G.offset[vertex + 1];

          if (T.size[warpId * pSize + iter] < upperBoundSize) {
            ui degInR;
            ui degInC;

            if ((vertex != ustar) && (status != 2)) {

              ui loc = atomicAdd( & sharedCounter[threadIdx.x / 32], 1);
              T.taskList[writeOffset + loc] = vertex;
              T.statusList[writeOffset + loc] = status;
              degInR = T.degreeInR[ind];
              degInC = T.degreeInC[ind];

              for (ui k = G.offset[vertex]; k < G.offset[vertex + 1]; k++) {
                if (G.neighbors[k] == ustar) {

                  T.degreeInC[ind]++;

                  if (degInR != 0) {
                    degInR--;
                  }

                }
              }

              T.degreeInR[writeOffset + loc] = degInR;
              T.degreeInC[writeOffset + loc] = degInC;
              T.degreeInR[ind] = degInR;
              //if(warpId==1630)
              //printf("iter %u wrap id %u laneid %u vertex %u status %u write loc %u new vertex %u new status %u \n",iter,warpId,i,vertex,status,writeOffset + loc,T.taskList[writeOffset + loc], T.statusList[writeOffset + loc]);

              if (T.doms[startIndex + end - 1] != 0) {
                int key = findIndexKernel(T.doms, startIndex + start, startIndex + start + T.doms[startIndex + end - 1], vertex);

                if (key != -1) {
                  T.statusList[ind] = 2;
                  T.statusList[writeOffset + loc] = 2;
                  T.degreeInC[ind] = 0;
                  T.degreeInR[ind] = 0;
                  T.degreeInC[writeOffset + loc] = 0;
                  T.degreeInR[writeOffset + loc] = 0;
                  for (int j = startNeighbor; j < endNeighbor; j++) {

                    resultIndex = findIndexKernel(T.taskList, startIndex + start, startIndex + end, G.neighbors[j]);
                    if (resultIndex != -1) {
                      if (T.degreeInR[resultIndex] != 0) {
                        atomicSub( & T.degreeInR[resultIndex], 1);
                        if (T.degreeInR[resultIndex] == 4294967295) {
                          T.degreeInR[resultIndex] = 0;

                        }
                      }
                    }
                    resultIndex = findIndexKernel(T.taskList, writeOffset, writeOffset + loc, G.neighbors[j]);
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

              //TODO: degre update

            }

          }
        }
        __syncwarp();

        if (laneId == 0) {
          if ((T.size[warpId * pSize + iter] < upperBoundSize) && (T.ustar[warpId * pSize + iter] != -1)) {
            *(T.flag) = 0;
            T.statusList[T.ustar[warpId * pSize + iter]] = 1;

            T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite + 1] = T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite] + sharedCounter[threadIdx.x / warpSize];
            T.taskOffset[bufferNum * pSize - 1]++;
            T.size[(bufferNum - 1) * pSize + totalTasksWrite] = T.size[warpId * pSize + iter];
            T.size[warpId * pSize + iter] += 1;
          }
          sharedCounter[threadIdx.x / 32] = 0;

          /*if (T.doms[startIndex + end - 1] != 0) {
            T.doms[(bufferNum - 1) * pSize + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite] + 1] = iter;
            T.doms[(bufferNum - 1) * pSize + T.taskOffset[(bufferNum - 1) * pSize + totalTasksWrite]] = warpId;
            T.ustar[(bufferNum - 1) * pSize + totalTasksWrite] = pSize * TOTAL_WARPS;

            //printf("iter %u wrap id %u location for new doms %u loc ustar ustar %d \n",iter,warpId,(bufferNum - 1) * pSize + totalTasksWrite,T.ustar[(bufferNum - 1) * pSize + totalTasksWrite]);
          }*/

        }
        __syncwarp(); 
        if()
      } else {
        if (laneId == 0)
          *
          outOfMemoryFlag = 1;

        return;
      }

    }
  }
}
// here i am 
__global__ void ExpandDoms(deviceGraphPointers G, deviceTaskPointers T, taskBuffer B, ui lowerBoundSize, ui upperBoundSize, ui pSize, ui dmax, ui jump) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui startIndex = warpId * pSize;
  ui endIndex = (warpId + 1) * pSize - 1;
  ui totalTasks = T.taskOffset[endIndex];
  ui overflow, position;

  for (ui iter = 0; iter < totalTasks; iter++) {
    __syncwarp();
    ui start = T.taskOffset[startIndex + iter];
    ui end = T.taskOffset[startIndex + iter + 1];
    ui total = end - start;

    if (T.ustar[warpId * pSize + iter] == pSize * TOTAL_WARPS) {
      ui prevWrap = T.doms[startIndex + start];
      ui prevIter = T.doms[startIndex + start + 1];
      ui domStart = prevWrap * pSize + T.taskOffset[prevWrap * pSize + prevIter];
      ui totalDoms = T.doms[prevWrap * pSize + T.taskOffset[prevWrap * pSize + prevIter + 1] - 1];
      ui prevUstar = T.taskList[T.ustar[prevWrap * pSize + prevIter]];
      ui newWriteOffset = startIndex + T.taskOffset[startIndex + totalTasks];
      ui totalWrite = total + 1;
      ui leftSpace;

      for (ui domIndex = 0; domIndex < totalDoms; domIndex++) {
        __syncwarp();

        leftSpace = (warpId + 1) * pSize - (newWriteOffset + totalWrite * domIndex) - 2;
        overFlow = (leftSpace > totalWrite) ? 0 : 1;
        ui mask, currentNum;

        if (i == 0) {
          currentNum = atomicAdd(B.numTasks, 1);
          position = atomicAdd(B.position, totalWrite);
          B.taskOffset[currentNum + 1] = position + totalWrite;
        }
        __shfl_sync(mask, position, 0);
        __shfl_sync(mask, currentNum, 0);

        for (ui i = laneId; i < total; i += 32) {
          ui srcIndex = startIndex + start + i;
          int key2 = findIndexKernel(T.doms, domStart + domIndex, domStart + totalDoms, T.taskList[srcIndex]);
          ui domVertex = T.doms[domStart + domIndex];

          if (!overFlow) {
            ui dstIndex = newWriteOffset + (totalWrite * domIndex) + i;
            T.taskList[dstIndex] = T.taskList[srcIndex];
            T.statusList[dstIndex] = (key2 != -1) ? 0 : T.statusList[srcIndex];
            if (T.taskList[srcIndex] == domVertex) {
              T.statusList[dstIndex] = 1;
            }
          } else {
            ui dstIndex = position + i;
            B.taskList[dstIndex] = T.taskList[srcIndex];
            B.statusList[dstIndex] = (key2 != -1) ? 0 : T.statusList[srcIndex];
            if (T.taskList[srcIndex] == domVertex) {
              B.statusList[dstIndex] = 1;

            }

          }

          __syncwarp();

          if (laneId == 0) {
            if (!overFlow) {
              T.taskList[newWriteOffset + (totalWrite * domIndex) + total] = prevUstar;
              T.statusList[newWriteOffset + (totalWrite * domIndex) + total] = 1;

              T.size[warpId * pSize + totalTasks + domIndex] = T.size[warpId * pSize + iter] + 2;
              T.taskOffset[warpId * pSize + totalTasks + domIndex + 1] = T.taskOffset[warpId * pSize + totalTasks] + ((domIndex + 1) * totalWrite);
              T.taskOffset[(warpId + 1) * pSize - 1]++;
              //printf("iter %u wrap %u domIndex %u size new %u size old %u index new %u index old %u \n",iter,warpId,domIndex,T.size[warpId * pSize + totalTasks + domIndex], T.size[warpId * pSize + iter],
              //warpId * pSize + totalTasks + domIndex, warpId * pSize + iter);
            } else {
              B.taskList[position + total] = prevUstar;
              B.statusList[position + total] = 1;
              T.size[currentNum] = T.size[warpId * pSize + iter] + 2;
            }

          }
        }

      }
    }
  }
}