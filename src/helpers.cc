__device__ ui minn(ui a, ui b) {
    return (a < b) ? a : b;
}

__device__ int findIndexKernel(ui *arr,ui start,ui end ,ui target)

{
  // Perform a linear search to find the index of the target vertex in the task array.
  // 'start' and 'end' are the indices of the task array defining the search range.
  // 'target' is the vertex we are searching for.

        int resultIndex = -1;
        for (ui index = start; index < end; index++)
        {
            if (arr[index] == target)
            {
                resultIndex = index;
                break;
            }
        }
        return resultIndex;

}
__global__ void IntialReductionRules(deviceGraphPointers G,deviceInterPointers P,ui size, ui upperBoundSize,ui lowerBoundDegree, ui pSize)
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
  if(start>size){
    start = size;
  }
  ui total = end -start;

  if(laneId==0){
    local_counter[threadIdx.x/32]=0;
  //printf("wrapId %d start %u end %u  \n",warpId,start,end);

  }
  __syncwarp();

  for(ui i=laneId;i<total;i+=32){

    ui vertex = start+i;
    if( (G.core[vertex] > lowerBoundDegree) && (G.distance[vertex] < (upperBoundSize-1)) ){
      ui loc = atomicAdd(&local_counter[threadIdx.x/32],1);
      P.intialTaskList[loc+writeOffset]=vertex;
    //printf("block id %u wrapId %u lane id %u vertex %u degree %u \n",blockIdx.x,warpId,i,vertex,G.degree[vertex]);

    }else{
      G.degree[vertex]=0;
    }


    }

      __syncwarp();
    if(laneId==0){
      P.entries[warpId] =local_counter[threadIdx.x/32];

      atomicAdd(P.globalCounter,local_counter[threadIdx.x/32]);
      //printf("block %d wrapId %d entries %u g %u \n",blockIdx.x,warpId,P.entries[warpId],*(P.globalCounter));


    }

}

__global__ void CompressTask(deviceGraphPointers G,deviceInterPointers P, deviceTaskPointers T, ui pSize, ui queryVertex){

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = idx / 32;
  int laneId = idx % 32;

  ui start = warpId*pSize;
  int temp = warpId-1;
  ui total = P.entries[warpId];
  ui writeOffset = 0;
  if(idx==0){
    T.size[0]=1;
  }

  for(ui i=laneId;i<total;i+=32){

      while(temp>=0){
        writeOffset += P.entries[temp];
        temp--;
      }
    int vertex = P.intialTaskList[start+i];
    T.taskList[writeOffset+i] = vertex;
    T.statusList[writeOffset+i] = (vertex == queryVertex ) ? 1 :0;
    ui nStart = G.offset[vertex];
    ui nEnd = G.offset[vertex+1];
    ui degInR = 0;
    ui degInc =0;
    for(ui k = nStart; k < nEnd; k++){
      if((G.neighbors[k]!=queryVertex) && (G.degree[G.neighbors[k]]!=0)){
        degInR++;
      }
      if(G.neighbors[k]==queryVertex){
        degInc++;

      }

    }


    T.degreeInR[writeOffset+i] = degInR;
    T.degreeInC[writeOffset+i] = degInc;
    //printf("warp id %u lane id %u vertex %u status %u degree %u dR %u dC %u \n",warpId,i,T.taskList[writeOffset+i],T.statusList[writeOffset+i],G.degree[vertex], T.degreeInR[writeOffset+i],T.degreeInC[writeOffset+i]);


  }

}

__global__ void ProcessTask(deviceGraphPointers G,deviceTaskPointers T,ui lowerBoundSize, ui upperBoundSize, ui pSize,ui dmax,ui *H,ui level)
{
    extern __shared__ char shared_memory[];
    ui sizeOffset = 0;
    // Stores new tasks
    ui *sharedUBDegree = (ui*)(shared_memory + sizeOffset);
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

    int otherId;

    int resultIndex;
    ui currentSize;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;




    ui startIndex =  warpId*pSize;
    ui endIndex  =   (warpId+1)*pSize -1;
    ui totalTasks = T.taskOffset[endIndex];
    if(laneId==0)
        {
            sharedUBDegree[threadIdx.x/32] = UINT_MAX;
            sharedScore[threadIdx.x/32] = 0;
            sharedUstar[threadIdx.x/32] = -1;
            sharedDegree[threadIdx.x/32]= UINT_MAX;


        }
        __syncwarp();


     for(ui iter =0; iter<totalTasks;iter++)
     {
        __syncwarp();
        

        ui start = T.taskOffset[startIndex+iter];
        ui end = T.taskOffset[startIndex+iter+1];
        ui total = end - start;
        // printf("iter %u wrap %u total %u \n",iter,warpId,totalTasks);

        
        ui maskGen = total;
        for(ui i = laneId; i < total ;i+=32)
        {
            int index = startIndex+start+i;
            ui ind = startIndex+start+i;
            ui vertex = T.taskList[ind];
            ui status = T.statusList[ind];
            //ui degR = T.degreeInR[ind];
            //ui degC = T.degreeInC[ind];
            //ui hSize = T.size[startIndex+iter];
        //printf("iter %u wrap %u lane id %u ind %u vertex %u status %u \n",iter,warpId,i,ind,vertex,status);



            ui startNeighbor = G.offset[vertex];
            ui endNeighbor = G.offset[vertex+1];

            score = 0;
            currentMinDegree = UINT_MAX;

            ustar = -1;



            if(status==0)
            {
                for(int j = startNeighbor; j < endNeighbor; j++)
                {
                    resultIndex = findIndexKernel(T.taskList,startIndex+start,startIndex+end,G.neighbors[j]);
                    if(resultIndex!=-1)
                    {
                        if(T.statusList[resultIndex]==1){
                          if(T.degreeInC[resultIndex]!=0)
                          {
                            score += (double) 1 / T.degreeInC[resultIndex];

                          }else{
                              score+=1;
                          }
                       // printf("Iter %u warp %u lane id % u neighbor %u found %u status %u degof neighbor %u score %f \n ",iter,warpId,i,G.neighbors[j],T.taskList[resultIndex],T.statusList[resultIndex],T.degreeInC[resultIndex],score);

                        }
                    }
                }
            }
            if(score>0)
            {
              score += (double) T.degreeInR[vertex]/dmax;
            }
             unsigned int mask;
            ui cond;
            if(maskGen>32){
               mask = (1u << 31) - 1;
              maskGen-=32;
              cond = 32;

            }
            else{
               mask = (1u << maskGen) - 1;
               cond = maskGen;

            }
            // __syncwarp();

            for (int offset = WARPSIZE/2 ; offset > 0; offset /= 2)
            {
              temp = __shfl_down_sync(mask, score , offset);
              otherId = __shfl_down_sync(mask, index, offset);

              if(temp> score && temp > 0 && (laneId+offset < cond))
              {
                score = temp;
                index = otherId;
              }
            }
            ustar = __shfl_sync(mask, index,0);
            score = __shfl_sync(mask, score,0);
        //printf("iter %u wrap %u lane id %u ind %u vertex %u status %u ustar %d score %f \n",iter,warpId,i,ind,vertex,status,ustar,score);



            if (status==1)
            {
              currentMinDegree = T.degreeInC[ind];
            }

              //atomicMin(&sharedDegree[threadIdx.x/32],currentMinDegree);
            // __syncwarp();
            // if((iter==8)&&(warpId==732) &&(level==15)&&(status==1))
            // printf("before iter %u wrap %u lane id %u ind %u vertex %u status %u  curr degree %u \n",iter,warpId,i,ind,vertex,status,currentMinDegree);


            for (int offset = 16 ; offset > 0; offset /= 2)
            {
                temp2 = __shfl_down_sync(mask,currentMinDegree  , offset);
                //if((iter==8)&&(warpId==732) &&(level==15)&&(status==1))
                //printf("before iter %u wrap %u lane id %u offset %u ind %u vertex %u status %u  curr degree %u \n",iter,warpId,i,offset,ind,vertex,status,temp2);


                if(laneId+offset <cond)
                {
                  currentMinDegree = min(currentMinDegree,temp2);
                }
            }
            // __syncwarp();
            // if((iter==8)&&(warpId==732) &&(level==15)&&(status==1)){
            //   printf("after iter %u wrap %u lane id %u ind %u vertex %u status %u  curr degree %u mask %u cond %u \n",iter,warpId,i,ind,vertex,status,currentMinDegree,mask,cond);
            // }





            currentMinDegree = __shfl_sync(mask, currentMinDegree, 0);

            //  __syncwarp();
            if(i%32==0)
            {
              if (currentMinDegree<sharedDegree[threadIdx.x/32]){
                sharedDegree[threadIdx.x/32] = currentMinDegree;
              }
              if(score>sharedScore[threadIdx.x/32]){

                sharedScore[threadIdx.x/32] = score;
                sharedUstar[threadIdx.x/32] = ustar;
              }



            }

        }
        __syncwarp();

        if(laneId==0){
            currentSize = T.size[startIndex+iter];
            if( (lowerBoundSize <= currentSize) && (currentSize <= upperBoundSize))
            {
                if(sharedDegree[threadIdx.x/32]!=UINT_MAX){
                ui oldvalue = atomicMax(G.lowerBoundDegree,sharedDegree[threadIdx.x/32]);
                if (oldvalue<sharedDegree[threadIdx.x/32]){
                //   printf("iter %u wrap %u shared min %u min %u old % u\n",iter,warpId,sharedDegree[threadIdx.x/32],*G.lowerBoundDegree,oldvalue);

                int y =0;
                for(ui x = 0; x < total ;x+=1){
                  if (T.statusList[startIndex+start+x]==1){
                    H[y] = T.taskList[startIndex+start+x];
                    y++;
                    // printf(" iter %u wrap %u vertex %u status %u degIc %u \n",iter,warpId,T.taskList[startIndex+start+x],T.statusList[startIndex+start+x],T.degreeInC[startIndex+start+x]);
                  }

                }
                H[100] = y;

                }
                }
            }
            int writeOffset = warpId*pSize;

            if( sharedScore[threadIdx.x/32]>0){
            T.ustar[writeOffset+iter] = sharedUstar[threadIdx.x/32];
            }else{
              T.ustar[writeOffset+iter] =INT_MAX;
            }
            sharedUBDegree[threadIdx.x/32] = UINT_MAX;
            sharedScore[threadIdx.x/32] = 0;
            sharedUstar[threadIdx.x/32] = -1;
            sharedDegree[threadIdx.x/32]= UINT_MAX;


        }

     __syncwarp();   
    
    }
}


__global__ void Expand(deviceGraphPointers G,deviceTaskPointers T,ui lowerBoundSize, ui upperBoundSize, ui pSize,ui dmax,ui jump){

    extern __shared__ char sharedMemory[];
    size_t sizeOffset = 0;

    ui *sharedCounter = (ui*)(sharedMemory + sizeOffset);
    sizeOffset += WARPS_EACH_BLK * sizeof(ui);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;

    ui startIndex =  warpId*pSize;
    ui endIndex  =   (warpId+1)*pSize -1;
    ui totalTasks = T.taskOffset[endIndex];

    if(laneId==0){
            sharedCounter[threadIdx.x/32] = 0;
        }
     __syncwarp();

     for(ui iter =0; iter<totalTasks;iter++)
     {

        ui start = T.taskOffset[startIndex+iter];
        ui end = T.taskOffset[startIndex+iter+1];
        ui total = end - start;
        if((T.ustar[warpId*pSize+iter]!= INT_MAX) && (T.ustar[warpId*pSize+iter]!=-1)){
         for(ui i = laneId; i < total ;i+=32)
        {

            //printf("inside expand ui wrap id %u lane %u totalver %u \n",warpId,i,total);

            ui ind = startIndex+start+i;
            ui vertex = T.taskList[ind];
            ui status = T.statusList[ind];

             if((T.size[warpId*pSize+iter] < upperBoundSize) && (T.ustar[warpId*pSize+iter]!=-1)){
                ui bufferNum = warpId+jump;
                if(bufferNum>TOTAL_WARPS){
                    bufferNum= bufferNum%TOTAL_WARPS;
                }
                ui totalTasksWrite = T.taskOffset[bufferNum*pSize -1];
                ui writeOffset =  ((bufferNum-1)*pSize) + T.taskOffset[(bufferNum-1)*pSize+totalTasksWrite];

                //printf("Wrap id %u lane id %u vertex %u status %u \n",warpId,i,T.taskList[ind], T.statusList[ind]);
                ui ustar = T.taskList[T.ustar[warpId*pSize+iter]];
                ui degInR;
                ui degInC;
                if((vertex!=ustar) && (status!=2)){
                    ui loc = atomicAdd(&sharedCounter[threadIdx.x/32],1);
                    T.taskList[writeOffset+loc] = vertex;
                    T.statusList[writeOffset+loc] = status;
                    degInR = T.degreeInR[ind];
                    degInC = T.degreeInC[ind];

                    for (ui k = G.offset[vertex]; k < G.offset[vertex+1]; k++){
                      if(G.neighbors[k]==ustar){


                        T.degreeInC[ind]++;

                        if(degInR!=0){
                        degInR --;
                        T.degreeInR[ind]--;
                        }

                      }
                    }

                    T.degreeInR[writeOffset+loc]=degInR;
                    T.degreeInC[writeOffset+loc]=degInC;



                    //printf("iter % u wrap %d lane %u vertex %u status %u index %u \n",iter,warpId,i,T.taskList[writeOffset+loc],T.statusList[writeOffset+loc],writeOffset+loc);
                }

             }


        }
        if (laneId==0){
            if((T.size[warpId*pSize+iter] < upperBoundSize) && (T.ustar[warpId*pSize+iter]!=-1) ){
            *(T.flag)=0;
            ui bufferNum = warpId+jump;
            if(bufferNum>TOTAL_WARPS){
                bufferNum=bufferNum%TOTAL_WARPS;
            }
            ui totalTasksWrite = T.taskOffset[bufferNum*pSize -1];
            //printf("ustar change loc %u \n",T.ustar[warpId*pSize+iter]);
            T.statusList[T.ustar[warpId*pSize+iter]]=1;

            T.taskOffset[(bufferNum-1)*pSize+totalTasksWrite+1] = T.taskOffset[(bufferNum-1)*pSize+totalTasksWrite]+ sharedCounter[threadIdx.x/32];
            T.taskOffset[bufferNum*pSize -1]++;
            T.size[(bufferNum-1)*pSize+totalTasksWrite] = T.size[warpId*pSize+iter];
            T.size[warpId*pSize+iter] +=1;
            //printf("iter % u wrap %d ustar %u offset %u offsetloc %u num tasks %u loc %u \n",iter,warpId,T.taskList[T.ustar[warpId*pSize+iter]],T.taskOffset[(bufferNum-1)*pSize+totalTasksWrite+1],(bufferNum-1)*pSize+totalTasksWrite+1,T.taskOffset[bufferNum*pSize -1],bufferNum*pSize -1);
            }

            sharedCounter[threadIdx.x/32]=0;
        }
        }

      __syncwarp();

     }


}

__global__ void reduce(deviceGraphPointers G,deviceTaskPointers T,ui pSize, ui upperBoundSize){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    int resultIndex;




    ui startIndex =  warpId*pSize;
    ui endIndex  =   (warpId+1)*pSize -1;
    ui totalTasks = T.taskOffset[endIndex];
    for(ui iter =0; iter<totalTasks;iter++)
     {

        ui start = T.taskOffset[startIndex+iter];
        ui end = T.taskOffset[startIndex+iter+1];
        ui total = end - start;
        //printf("iter %u wrap %u total %u \n",iter,warpId,totalTasks);

        for(ui i = laneId; i < total ;i+=32)

        {
          ui ind = startIndex+start+i;
            ui vertex = T.taskList[ind];
            ui status = T.statusList[ind];
            ui degR = T.degreeInR[ind];
            ui degC = T.degreeInC[ind];
            ui hSize = T.size[startIndex+iter];
            ui startNeighbor = G.offset[vertex];
            ui endNeighbor = G.offset[vertex+1];

          if(status==0){
              if(minn((degR+degC), (degC+upperBoundSize-hSize-1)) <= *G.lowerBoundDegree){
                //printf("removed %u size %u \n",vertex,hSize);
                T.statusList[ind]=2;
                T.degreeInR[ind] = 0;
                T.degreeInC[ind] =0;
                for(int j = startNeighbor; j < endNeighbor; j++){
                  resultIndex = findIndexKernel(T.taskList,startIndex+start,startIndex+end,G.neighbors[j]);
                  if(resultIndex!=-1){
                    if(T.degreeInR[resultIndex]!=0){
                      atomicSub(&T.degreeInR[resultIndex], 1);
                    }
                  }
                }


              }
            }
            ui neig;
            if((status==1)&&((degC+degR)==(*G.lowerBoundDegree+1))){
              for(int j = startNeighbor; j < endNeighbor; j++){
                resultIndex = findIndexKernel(T.taskList,startIndex+start,startIndex+end,G.neighbors[j]);
                if(resultIndex!=-1){
                  if(T.statusList[resultIndex]==0){

                    T.statusList[resultIndex] =1;
                    atomicAdd(&T.size[startIndex+iter],1);
                    neig = T.taskList[resultIndex];
                    for(int k = G.offset[neig]; k < G.offset[neig+1]; k++){
                      resultIndex = findIndexKernel(T.taskList,startIndex+start,startIndex+end,G.neighbors[k]);
                      if(resultIndex!=-1){
                        //if(T.statusList[resultIndex]==1){
                          atomicAdd(&T.degreeInC[resultIndex], 1);
                        //}
                        if(T.degreeInR[resultIndex]!=0){
                          atomicSub(&T.degreeInR[resultIndex], 1);

                        }
                      }


                    }


                  }
                }

              }

            }


        }
     }


}
