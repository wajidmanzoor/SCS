// Reduction rule 1,2,3 added in SBSNEW
// Upperbound technique 1 added.
__device__ int findIndexKernel(ui *arr,ui start,ui end ,ui target)

{
  // Perform a binary search to find the index of the target vertex in the task array.
  // 'start' and 'end' are the indices of the task array defining the search range.
  // 'target' is the vertex we are searching for.

        int resultIndex = -1;
        for (ui index = start; index < end; ++index)
        {
            if (arr[index] == target)
            {
                resultIndex = index;
                break;
            }
        }
        return resultIndex;

}

__global__ void SCSSpeedEff(ui *taskList, ui *taskStatus, ui *taskOffset, ui *neighbors,ui *neighborOffset, ui *degree,ui *distanceFromQID, bool *flag, ui *lowerBoundDegree,ui lowerBoundSize, ui upperBoundSize, ui pSize,ui dmax,int *ustarList,ui *subgraphSize)
{
    extern __shared__ char shared_memory[];
    ui sizeOffset = 0;
    // Stores new tasks
    ui *sharedSize = (ui*)(shared_memory + sizeOffset);
    sizeOffset += WARPS_EACH_BLK * sizeof(ui);

    ui *sharedDegree = (ui*)(shared_memory + sizeOffset);
    sizeOffset += WARPS_EACH_BLK * sizeof(ui);

    int *sharedUstar = (int*)(shared_memory + sizeOffset);
    sizeOffset += WARPS_EACH_BLK * sizeof(int);

    double *sharedScore = (double *)(shared_memory + sizeOffset);
    sizeOffset += WARPS_EACH_BLK * sizeof(double);



    // minimum degree intialized to zero.
    ui currentMinDegree;

    // Connection score used to calculate the ustar.
    double score;
    int ustar;

    double temp;
    ui temp2;
    ui temp3;

    int otherId;

    // Declare to store the index in task
    int resultIndex;

    // will store the index of vertex in task.

    ui current_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;




    ui startIndex =  warpId*pSize;
    ui endIndex  =   (warpId+1)*pSize -1;
    ui totalTasks = taskOffset[endIndex];
    if(laneId==0)
        {
            sharedSize[warpId] = 0;
            sharedScore[warpId] = 0;
            sharedUstar[warpId] = -1;
            sharedDegree[warpId]= UINT_MAX;


        }
        __syncwarp();


     for(ui iter =0; iter<totalTasks;iter++)
     {

        ui start = taskOffset[startIndex+iter];
        ui end = taskOffset[startIndex+iter+1];
        ui total = end - start;
        //printf("iter %u wrap %u total %u \n",iter,warpId,totalTasks);


        for(ui i = laneId; i < total ;i+=32)
        {
            int index = startIndex+start+i;
            ui ind = startIndex+start+i;
            ui vertex = taskList[ind];
            ui status = taskStatus[ind];
            ui stat = taskStatus[ind];


            ui startNeighbor = neighborOffset[vertex];
            ui endNeighbor = neighborOffset[vertex+1];


            // intialize.
            score = 0;
            currentMinDegree = 0;

            ustar = -1;
            temp3 = UINT_MAX;
            //printf("block %u iter %u Wrap %u Lane id %u index %u vertex %u status %u size %u score %f s %u e %u \n",blockIdx.x,iter,warpId,i,ind,vertex,status,subgraphSize[startIndex+iter],score,start,end);

            if(status==0)
            {


                // Iterate through neighbors of v.
                for(int j = startNeighbor; j < endNeighbor; j++)
                {
                    // Get index of neighbor of V in task

                    resultIndex = findIndexKernel(taskList,startIndex+start,startIndex+end,neighbors[j]);
                        //printf(" vertex %u index %u result index %d status %u ver %u neighbor %u s %u e %u \n",vertex,ind,resultIndex, taskStatus[resultIndex], taskList[resultIndex],neighbors[j],start,end);

                    if(resultIndex!=-1)
                    {

                        // If neighbor in C increament the connection score.
                        if(taskStatus[resultIndex]==1)
                        {
                            score += (double) 1 / degree[neighbors[j]];
                            //printf(" score %f ",score);


                            //degreeInC++;
                        }

                        /*if((taskStatus[resultIndex]==1) || (taskStatus[resultIndex]==0))
                        {
                            degreeInR++;
                        }*/


                    }
                }
                //printf("\n");
            }

            if(score>0)
            {
              // increament connection score, if vertex can be connected to C
              score += (double) degree[vertex]/dmax;
            }


            for (int offset = WARPSIZE/2 ; offset > 0; offset /= 2)
            {
              temp = __shfl_down_sync(0xFFFFFFFF, score , offset);
              otherId = __shfl_down_sync(0xFFFFFFFF, index, offset);

              if(temp> score && temp > 0)
              {
                score = temp;

                index = otherId;
              }

            }



            // Broadcast the index of vertex and max connection score to all threads in wrap.
            ustar = __shfl_sync(0xFFFFFFFF, index,0);
            score = __shfl_sync(0xFFFFFFFF, score,0);


            // Calculate min degree of each subgraph

            // In vertex in C.
            if (status==1)
            {


              // itterate through neighbors of v
              for(int j = startNeighbor; j < endNeighbor; j++)
              {
                // Get index in task
                resultIndex = findIndexKernel(taskList,startIndex+start,startIndex+end,neighbors[j]);

                if(resultIndex!=-1){
                // If neighbor in C, increament count by 1.
                  if (taskStatus[resultIndex]==1)
                  {

                    currentMinDegree ++;
                    //degreeInC++;
                  }

                  /*if((taskStatus[resultIndex]==1) || (taskStatus[resultIndex]==0)){
                    degreeInR++;
                  }*/
                }
              }
            }


            // Using shuffle down get minimun degree

            for (int offset = WARPSIZE/2 ; offset > 0; offset /= 2)
            {
                temp2 = __shfl_down_sync(0xFFFFFFFF,currentMinDegree  , offset);

                if(temp2 >0 && temp2 < temp3)
                {
                  currentMinDegree = temp2;
                  temp3 = temp2;
                }
            }

            // Boardcast the minimum degree to all thread of the warp
            currentMinDegree = __shfl_sync(0xFFFFFFFF, temp3,0);


            // Calculate the size of the subgraph (C) by summing the status values,
            // where status is 1 if the vertex is in C.
            for (int offset = WARPSIZE/2 ; offset > 0; offset /= 2)
            {
              stat += __shfl_down_sync(0xFFFFFFFF, stat, offset);
            }

            // Boardcast the current size to each thread inside a warp.
            current_size = __shfl_sync(0xFFFFFFFF, stat, 0);


            if(i%32==0)
            {
              sharedSize[warpId] += current_size;
              if (currentMinDegree<sharedDegree[warpId]){
                sharedDegree[warpId] = currentMinDegree;
              }
              if(score>sharedScore[warpId]){

                sharedScore[warpId] = score;
                sharedUstar[warpId] = ustar;
              }



            }

        }

        if(laneId==0){

            if( (lowerBoundSize <= sharedSize[warpId]) && (sharedSize[warpId] <= upperBoundSize))
            {
                if(sharedDegree[warpId]!=UINT_MAX){

                // attomic compare the current min degree and current max min degree (k lower)
                atomicMax(lowerBoundDegree,sharedDegree[warpId]);

                }
            }
            int writeOffset = warpId*pSize;

            if( sharedScore[warpId]>0){
            ustarList[writeOffset+iter] = sharedUstar[warpId];
            }
            subgraphSize[writeOffset+iter] = sharedSize[warpId];
            //printf("iter %u wrap %u  scire %f ustar %d size %u\n ",iter,warpId,sharedScore[warpId],ustarList[writeOffset+iter],subgraphSize[writeOffset+iter]);
            //printf("lower bound degree %u  sharedSize %u sharedUstar %d \n",sharedDegree[warpId],sharedSize[warpId],sharedUstar[warpId]);
            sharedSize[warpId] = 0;
            sharedScore[warpId] = 0;
            sharedUstar[warpId] = -1;
            sharedDegree[warpId]= UINT_MAX;


        }


    }
}


__global__ void Expand(ui *taskList, ui *taskStatus, ui *taskOffset, int *ustarList,ui *subgraphSize,ui *neighbors,ui *neighborOffset, ui *degree,ui *distanceFromQID, bool *flag, ui *lowerBoundDegree,ui lowerBoundSize, ui upperBoundSize, ui pSize,ui dmax){

    extern __shared__ char sharedMemory[];
    size_t sizeOffset = 0;
    // Stores new tasks
    ui *sharedCounter = (ui*)(sharedMemory + sizeOffset);
    sizeOffset += WARPS_EACH_BLK * sizeof(ui);



    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;




    ui startIndex =  warpId*pSize;
    ui endIndex  =   (warpId+1)*pSize -1;
    ui totalTasks = taskOffset[endIndex];

    if(laneId==0){
            sharedCounter[warpId] = 0;
        }
     __syncwarp();

     for(ui iter =0; iter<totalTasks;iter++)
     {

        ui start = taskOffset[startIndex+iter];
        ui end = taskOffset[startIndex+iter+1];
        ui total = end - start;

         for(ui i = laneId; i < total ;i+=32)
        {

            //printf("inside expand ui wrap id %u lane %u totalver %u \n",warpId,i,total);

            ui ind = startIndex+start+i;
            ui vertex = taskList[ind];
            ui status = taskStatus[ind];

             if((subgraphSize[warpId*pSize+iter] < upperBoundSize) && (ustarList[warpId*pSize+iter]!=-1)){
                ui bufferNum = warpId+2;
                if(bufferNum>TOTAL_WARPS){
                    bufferNum=1;
                }
                //printf("Wrap id %u lane id %u actual %u buffer num % u total % u \n",warpId,i,warpId+2,bufferNum,TOTAL_WARPS);
                ui totalTasksWrite = taskOffset[bufferNum*pSize -1];
                ui writeOffset =  ((bufferNum-1)*pSize) + taskOffset[(bufferNum-1)*pSize+totalTasksWrite];


                if((vertex!=taskList[ustarList[warpId*pSize+iter]]) && (status!=2)){
                    ui loc = atomicAdd(&sharedCounter[warpId],1);
                    taskList[writeOffset+loc] = vertex;
                    taskStatus[writeOffset+loc] = status;
                    //printf("iter % u wrap %d lane %u vertex %u status %u index %u \n",iter,warpId,i,taskList[writeOffset+loc],taskStatus[writeOffset+loc],writeOffset+loc);
                }

             }
            // __syncwarp();


        }
        if (laneId==0){
            if((subgraphSize[warpId*pSize+iter] < upperBoundSize) && (ustarList[warpId*pSize+iter]!=-1) ){
            *flag=0;
            ui bufferNum = warpId+2;
            if(bufferNum>TOTAL_WARPS){
                bufferNum=1;
            }
            ui totalTasksWrite = taskOffset[bufferNum*pSize -1];
            taskStatus[taskList[ustarList[warpId*pSize+iter]]]=1;
            taskOffset[(bufferNum-1)*pSize+totalTasksWrite+1] = taskOffset[(bufferNum-1)*pSize+totalTasksWrite]+ sharedCounter[warpId];
            taskOffset[bufferNum*pSize -1]++;
            //printf("iter % u wrap %d offset %u offsetloc %u num tasks %u loc %u \n",iter,warpId,taskOffset[(bufferNum-1)*pSize+totalTasksWrite+1],(bufferNum-1)*pSize+totalTasksWrite+1,taskOffset[bufferNum*pSize -1],bufferNum*pSize -1);
            }

            sharedCounter[warpId]=0;
        }



     }


}

__global__ void IntialReductionRules(ui *neighborsOffset,ui *neighbors,ui *degree,ui *distanceFromQID,ui *coreValues , ui *taskList, ui * taskStatus, ui *numEntries, ui *globalCounter,ui queryVertex , ui size  , ui upperBoundSize, ui lowerBoundDegree,ui pSize)
{

  // Intial Reduction rule based on core value and distance.
  extern __shared__ ui shared_memory1[];

  // Store the counter
  ui *local_counter = shared_memory1;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui start = warpId*pSize;
  ui end = (warpId+1)*pSize;
  ui writeOffset = warpId*pSize;
  if(end>size){
    end = size;
  }
  ui total = end -start;
  if(laneId==0){
    local_counter[warpId]=0;
  //printf("wrapId %d start %u end %u \n",warpId,start,end);
  }
  __syncwarp();

  for(ui i=laneId;i<total;i+=32){

    ui vertex = start+i;
    if( (coreValues[vertex] > lowerBoundDegree) && (distanceFromQID[vertex] < (upperBoundSize-1)) ){
      ui loc = atomicAdd(&local_counter[warpId],1);
      taskList[loc+writeOffset]=vertex;
      taskStatus[loc+writeOffset] = (vertex == queryVertex) ? 1 : 0;


    }


    }
      __syncwarp();
    if(laneId==0){
      numEntries[warpId] =local_counter[warpId];
      atomicAdd(globalCounter,local_counter[warpId]);
  //printf("wrapId %d local c %u counter %u \n",warpId,local_counter[warpId],*globalCounter);

    }

}
__global__ void CompressTask(ui *taskList, ui * taskStatus, ui *numEntries, ui *outputTasks, ui *outputStatus,ui pSize){

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui start = warpId*pSize;
  int temp = warpId-1;
  ui total = numEntries[warpId];
  ui writeOffset = 0;

  for(ui i=laneId;i<total;i+=32){

      while(temp>=0){
        writeOffset += numEntries[temp];
        temp--;
      }

    outputTasks[writeOffset+i] = taskList[start+i];
    outputStatus[writeOffset+i] = taskStatus[start+i];

  }

}
