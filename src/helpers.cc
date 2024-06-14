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

__global__ void SCSSpeedEff(ui *taskList, ui *taskStatus, ui *taskOffset, ui *neighbors,ui *neighborOffset, ui *degree,ui *distanceFromQID, bool *flag, ui *lowerBoundDegree,ui lowerBoundSize, ui upperBoundSize, ui pSize,ui dmax)
{

  // This kernel applies a reduction rule to prune the set R.
  // It compares and updates the minimum degree.
  // Calculated ustar.
  // returns the ustar for each task in the output array.


    extern __shared__ ui shared_memory[];

    // Stores new tasks
    ui *localCounter = shared_memory;


    // minimum degree intialized to zero.
    ui currentMinDegree;

    // Connection score used to calculate the ustar.
    double score;

    int ustar = -1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;


    // Upper bound distance from query vertex.
    ui ubD;
    ui degreeBasedUpperBound;

    ui startIndex =  warpId*pSize;
    ui endIndex  =   (warpId+1)*pSize -1;
    ui totalTasks = taskOffset[endIndex];


    ui degreeInC =0;
    ui degreeInR =0;

    ui temp2;
    ui temp3 = UINT_MAX;
    int temp4;
    double temp;
    int otherId;

    //ui iter = 0;
    /*if(laneId==0){
      iter[warpId]=0;
    }*/


   for(ui iter =0; iter<totalTasks;iter++)
     {

        ui start = taskOffset[startIndex+iter];
        ui end = taskOffset[startIndex+iter+1];
        ui total = end - start;

        degreeBasedUpperBound = UINT_MAX;

        for(ui i = laneId; i < total ;i+=32)
        {   if(i==0){
              localCounter[warpId] = 0;
              //printf("Wrap %u Task : Start %u end %u iter %u \n",warpId,start,end,iter);
            }
            __syncwarp();



            int index = startIndex+start+i;
            ui ind = startIndex+start+i;
            ui vertex = taskList[ind];
            ui status = taskStatus[ind];
            //ui stat = taskStatus[ind];

          //printf("Wrap %u Lane id %u vertex %u status %u \n",warpId,i,vertex,status);

            ui startNeighbor = neighborOffset[vertex];
            ui endNeighbor = neighborOffset[vertex+1];


            // intialize connection score to zero.
            score = 0;
            currentMinDegree = 0;

          if(taskStatus[ind]==0)
          {
            // will store the index of vertex in task.
            int resIndex;

            // Iterate through neighbors of v.
            for(int j = startNeighbor; j < endNeighbor; j++)
            {
              // Get index of neighbor of V in task
              resIndex = findIndexKernel(taskList,start,end,neighbors[j]);
              if(resIndex!=-1){
              // If neighbor in C increament the connection score.
              if(taskStatus[resIndex]==1)
              {
                  score += (double) 1 / degree[neighbors[j]];
                  degreeInC++;
              }

              if((taskStatus[resIndex]==1) || (taskStatus[resIndex]==0)){
                degreeInR++;
              }


            }
            }
          }

            if(score>0){
              // increament connection score, if vertex can be connected to C
              score += (double) degree[vertex]/dmax;
            }

           
            for (int offset = WARPSIZE ; offset > 0; offset /= 2)
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
            if (taskStatus[ind]==1)
            {
              // Declare to store the index in task
              int resultIndex;

              // itterate through neighbors of v
              for(int j = startNeighbor; j < endNeighbor; j++)
              {
                // Get index in task
                resultIndex = findIndexKernel(taskList,start,end,neighbors[j]);

                if(resultIndex!=-1){
                // If neighbor in C, increament count by 1.
                  if (taskStatus[resultIndex]==1)
                  {

                    currentMinDegree ++;
                    degreeInC++;
                  }

                  if((taskStatus[resultIndex]==1) || (taskStatus[resultIndex]==0)){
                    degreeInR++;
                  }
                }
              }
            }


            // Using shuffle down get minimun degree
          
            for (int offset = WARPSIZE ; offset > 0; offset /= 2)
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
            for (int offset = WARPSIZE/2 ; offset > 0; offset /= 2) {
              status += __shfl_down_sync(0xFFFFFFFF, status, offset);
            }

            // Boardcast the current size to each thread inside a warp.
            ui current_size = __shfl_sync(0xFFFFFFFF, status, 0);

            //Compare max min degree and update ustar
            // First thread of the warp.
            if (i == 0)
           {
              if( (lowerBoundSize <= current_size) && (current_size <= upperBoundSize))
              {
                if(currentMinDegree!=UINT_MAX){

                // attomic compare the current min degree and current max min degree (k lower)
                atomicMax(lowerBoundDegree,currentMinDegree);

                }
              }

              ubD = 0;
              if(*lowerBoundDegree<=1) ubD = upperBoundSize-1;
              else{
                  for(ui d = 1; d <= upperBoundSize; d++){
                  if(d == 1 || d == 2){
                      if(*lowerBoundDegree + d > upperBoundSize){
                        ubD = d - 1;
                        break;
                    }
                  }
                  else{
                      ui min_n = *lowerBoundDegree + d + 1 + (d/3) * (*lowerBoundDegree - 2);
                      if(upperBoundSize < min_n){
                          ubD = d - 1;
                          break;
                      }
                  }
                }
              }


           }
           // Boardcast upper bound distance to each thread inside a warp
              ui uperboundDistance =  __shfl_sync(0xFFFFFFFF, ubD,0);

              if(taskStatus[ind] == 0){
                // If distance of vertex is more than the upper bound. change status to 2.
                if(uperboundDistance < distanceFromQID[vertex]){
                  taskStatus[ind]=2;
                }


                if(miv(degreeInC + upperBoundSize - current_size -1, degreeInR) <= *lowerBoundDegree){
                  taskStatus[ind]=2;

                }
              }

              // upper bound technique
              if(taskStatus[ind]==1)
                {

                degreeBasedUpperBound = miv(degreeInC + upperBoundSize - current_size -1, degreeInR);
                }


              
              for (int offset = WARPSIZE ; offset > 0; offset /= 2)
              {
                  temp4 =__shfl_down_sync(0xFFFFFFFF,degreeBasedUpperBound , offset);

                  if ( (temp4 <  degreeBasedUpperBound) && (temp4>0)){
                    degreeBasedUpperBound = temp4;
                  }

              }
              degreeBasedUpperBound = __shfl_sync(0xFFFFFFFF,degreeBasedUpperBound,0);


              __syncwarp();
              
              if((current_size < upperBoundSize) && (score>0)&& (degreeBasedUpperBound >*lowerBoundDegree) )

              {
                ui bufferNum = warpId+2;
                if(bufferNum>TOTAL_WARPS){
                  bufferNum=1;
                }
                //printf("Wrap id %u lane id %u actual %u buffer num % u total % u \n",warpId,i,warpId+2,bufferNum,TOTAL_WARPS);
                ui totalTasksWrite = taskOffset[bufferNum*pSize -1];
                ui writeOffset =  ((bufferNum-1)*pSize) + taskOffset[(bufferNum-1)*pSize+totalTasksWrite];


                if((taskList[ind]!=taskList[ustar]) && (taskStatus[ind]!=2)){
                  ui loc = atomicAdd(&localCounter[warpId],1);
                  taskList[writeOffset+loc] = taskList[ind];
                  taskStatus[writeOffset+loc] = taskStatus[ind];
                  //printf("wrap %d lane %u vertex %u status %u index %u \n",warpId,i,taskList[writeOffset+loc],taskStatus[writeOffset+loc],writeOffset+loc);
                }

               __syncwarp();
               if(i==0){
                *flag=0;
                //printf("wrapId %u index %d usar %u score %f \n",warpId,ustar,taskList[ustar],score);
              taskStatus[ustar]=1;
               taskOffset[(bufferNum-1)*pSize+totalTasksWrite+1] = taskOffset[(bufferNum-1)*pSize+totalTasksWrite]+ localCounter[warpId];
               taskOffset[bufferNum*pSize -1]++;
               }
               __syncwarp();


              }


          }

    }
}




__global__ void IntialReductionRules(ui *neighborsOffset,ui *neighbors,ui *degree,ui *distanceFromQID,ui *coreValues , ui *taskList, ui * taskStatus, ui *numEntries, ui *globalCounter,ui queryVertex , ui size  , ui upperBoundSize, ui lowerBoundDegree,ui pSize)
{

  // Intial Reduction rule based on core value and distance.
  extern __shared__ ui shared_memory[];

  // Store the counter
  ui *local_counter = shared_memory;

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
    if( coreValues[vertex] > lowerBoundDegree ){
      ui loc = atomicAdd(&local_counter[warpId],1);
      taskList[loc+writeOffset]=vertex;
      taskStatus[loc+writeOffset] = (vertex == queryVertex) ? 1 : 0;


    }
   

    }
      __syncwarp();
    if(laneId==0){
      numEntries[warpId] =local_counter[warpId];
      atomicAdd(globalCounter,local_counter[warpId]);
  //printf("wrapId %d counter %u \n",warpId,*globalCounter);

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


     
