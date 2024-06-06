
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

__inline__ __device__ void warpMin2(ui val, ui &minVal, ui &secondMinVal)
{
  // Calculates the second minimum degree within a warp.
  // This is used to determine the minimum degree of a subgraph.
  // Each thread in a warp checks the degree of a vertex, and threads that don't check any vertex have their degree initialized to 0.
  // Therefore, we calculate the second minimum to account for these initialized values.


    unsigned mask = 0xffffffff;

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int other = __shfl_down_sync(mask, val, offset);

        if (other < minVal)
        {
            secondMinVal = minVal;
            minVal = other;
        }
        else if (other < secondMinVal && other > minVal)
        {
            secondMinVal = other;
        }
    }
}

__global__ void SBS(ui *tasksOffset,ui *taskList,ui *taskStatus,ui *neighbors,ui *neighborOffset, ui *lowerBoundDegree,ui TaskSize,ui lowerBoundSize, ui upperBoundSize, int numTasks,int *ustarList, ui *degree,ui dmax,ui *distanceFromQID)
{
    // This kernel applies a reduction rule to prune the set R.
  // It compares and updates the minimum degree.
  // Calculated ustar. 
  // returns the ustar for each task in the output array.

    // minimum degree intialized to zero.
    ui count = 0;

    // Connection score used to calculate the ustar. 
    double score;


    ui minVal = UINT_MAX;
    ui secondMinVal = UINT_MAX;

    // intialize max connection score to zero. 
    double maxScore = 0;

    // intilaize ustar to -1, indicating no ustar was found. 
    int ustar = -1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int wrapId = idx / 32;
    int laneId = idx % 32;

    // Upper bound for distance from query vertex. 
    ui ubD;


    for(ui i = wrapId; i < numTasks; i+=TOTAL_WARPS)
    {

      // For each task get start index , end index  and total verticies. 
      ui start = tasksOffset[i];
      ui end = tasksOffset[i+1];
      ui total = end-start;

      // If lane id is less than total, lane id will check the vetex
      for(ui j = laneId; j < total ;j+=32)
      {

      printf(" wrap id %u start %u end %u \n",i,start,end);


        int id = j;
        int index = start+id;

        // Vertex 
        ui v = taskList[start+id];

        // vertex status 
        ui stat = taskStatus[start+id];

        // offset for neighbors of vertex v 
        ui start_n = neighborOffset[v];
        ui end_n = neighborOffset[v+1];

        // intialize connection score to zero. 
        score = 0; 


        // Calculate ustar 
        // if vertex in R. 
        if(taskStatus[start+id]==0)
        { 
          // increament connection score 
          score += (double) degree[v]/dmax;

          // will store the index of vertex in task. 
          int resIndex; 

          // Iterate through neighbors of v. 
          for(int k = start_n; k <end_n;k++)
          {
            // Get index of neighbor of V in task
            resIndex = findIndexKernel(taskList,start,end,neighbors[k]);

            // If neighbor in C increament the connection score. 
            if(taskStatus[resIndex]==1)
            {
                score += (double) 1 / degree[neighbors[k]];
            }

          }
        }


        // Using shuffle down get the maximum connection score and the index of vertex (ustar) with max connection score. 
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            double temp = __shfl_down_sync(0xFFFFFFFF,score , offset);
            int otherId = __shfl_down_sync(0xFFFFFFFF,index,offset);
            if(temp> maxScore)
            {
              maxScore = temp;
              ustar = otherId;
            }
        }
        

        // Broadcast the index of vertex with max connection score to all threads in wrap. 
        ustar = __shfl_sync(0xFFFFFFFF, ustar,0);
        //printf(" id : %d score %f vertex %u ustar %d \n ", start+id, score,v,taskList[ustar]);



        // Calculate min degree of each subgraph

        // In vertex in C. 
        if (taskStatus[start + id]==1)
        {

          // Declare to store the index in task
          int resultIndex;

          // itterate through neighbors of v 
          for(int k = start_n; k <end_n;k++)
          {

            // Get index in task 
            resultIndex = findIndexKernel(taskList,start,end,neighbors[k]);

            // If neighbor in C, increament count by 1. 
            if (taskStatus[resultIndex]==1)
            {
              count ++;
            }
          }
        }

        printf("Thread Id :  %d wrapId: %d laneId :  %d  Value :  %u Count: %d status %d \n", idx, i,j, v,count,stat);

        // Get second minimum to get the minimum degree. 

        // TODO: try to make it more efficient

        warpMin2(count, minVal, secondMinVal);

        // handle a boundary case  
        if(j==0){
          if(secondMinVal==UINT_MAX){
            secondMinVal = minVal;
          }
        }

        // Boardcast the minimum degree to all thread of the warp
        ui boardcastValue = __shfl_sync(0xFFFFFFFF, secondMinVal,0);

      // Calculate the size of the subgraph (C) by summing the status values,
      // where status is 1 if the vertex is in C.

        for (int offset = 16; offset > 0; offset /= 2) {
        stat += __shfl_down_sync(0xFFFFFFFF, stat, offset);
        }

        // Boardcast the current size to each thread inside a warp.
        ui current_size = __shfl_sync(0xFFFFFFFF, stat,0);

        // First thread of the warp. 
        if(j==0)
        
        {

          // Check if size is between in size limits. 

          if(lowerBoundSize <= current_size && current_size <= upperBoundSize)
          {

            // attomic compare the current min degree and current max min degree (k lower)
            atomicMax(lowerBoundDegree,boardcastValue);
            printf(" Lower bound degree %u boardcastValue %u \n",*lowerBoundDegree,boardcastValue);

          }

          // if size is less than upper bound, only then we will continue to add vertecies to C. 

          if(current_size<upperBoundSize){
          // ustar of the warp will be addded to global memory. 
          ustarList[i] = (int) taskList[ustar];
          //printf(" warp id %u value %u \n",i,ustarList[i] );
          }


        }

        // First thread of the wrap will caculate the upper bound distance. 
        // Needs to be calculated each time as the curent Max min degree changes. 
        if (j==0){
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



        // If distance of vertex is more than the upper bound. change status to 2. 
        if(uperboundDistance < distanceFromQID[j]){
          taskStatus[start+id]=2;
        }
      }

    }

  }

__global__ void IntialReductionRules(ui *neighborsOffset,ui *neighbors,ui *degree,ui *distanceFromQID, ui qid, ui vertexSize , ui *taskList, ui * taskStatus, ui upperBoundSize, ui *globalCounter)
{

  // Intial Reduction rule based on distance. 

  extern __shared__ ui shared_memory[];

  // Store the task 
  ui *shared_task = shared_memory;

  // Store the task status
  ui *shared_status = &shared_memory[blockDim.x];
  

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  ui thread_id = threadIdx.x;
   __shared__ ui local_counter;

  // Intialize local counter 
  if (thread_id == 0)
  {
        local_counter = 0;
    }
  __syncthreads();

// Each thread will handle one vertex 
  for (ui i = idx; i < vertexSize; i += TOTAL_THREAD) {

    // If distance < upperbound add to shared memory
      if (distanceFromQID[i] < upperBoundSize - 1) {
                ui loc = atomicAdd(&local_counter, 1);
                shared_task[loc] = i;

                // If query vertex set status to 1 else to zero. 
                if(i==qid)
                {
                  shared_status[loc] =1;
                }
                else
                {
                  shared_status[loc] =0;
                }
            }
        }


    __syncthreads();


    // Copy from shared memory to Global memory. 
    for (ui i = thread_id; i < local_counter; i += blockDim.x) {
        taskList[*globalCounter + i] = shared_task[i];
        taskStatus[*globalCounter + i] = shared_status[i];
        //printf("id %u globalcounter %u value %u taskStatus %u \n",i,*globalCounter,taskList[*globalCounter + i],taskStatus[*globalCounter + i]);

    }
     __syncthreads();

     // Increament global memory. 
    if (thread_id == 0) {

       atomicAdd(globalCounter, local_counter);

    }
     __syncthreads();


}

__global__ void Branching(int *ustarList,ui *inputTaskList,ui *inputTaskOffset,ui *inputTaskStatus, ui *globalCounter, int inputTaskListSize, ui *outputTaskOffset,ui *outputTaskList,ui *outputTaskStatus, ui prevLevelNodeNum,ui currentLevelNodeNum)
{

  // Create two new tasks based on the old task and ustar 

  extern __shared__ ui shared_memory[];

  // Stores new tasks 
  ui *shared_task = shared_memory;
  // Stores status 
  ui *shared_status = &shared_memory[blockDim.x];


// Local counter for this block. 
  __shared__ ui local_counter;


  ui blockId = blockIdx.x;
//  intialize local counter 
  if(threadIdx.x==0){
    local_counter=0;

  }
   __syncthreads();
  
// Process to create  first new task (C + ustar, R - ustar) 

// Each block handles on task and creates two new tasks
  for (ui i = blockId; i < prevLevelNodeNum; i+= BLK_NUMS )
  {

    // if ustar was found in last kernel call. 
    if(ustarList[i]!=-1){

    // Get start and end index of task
     ui start = inputTaskOffset[i];
     ui end = inputTaskOffset[i+1];
     ui total = end-start;
     

     // Each thread in block check on vertex of a task
     for(ui j = threadIdx.x ; j < total ;j+=BLK_DIM)
      {
        ui ind = start+j;
        ui v = inputTaskList[ind];

        // If vertex in either C or R. I.E wasn't pruned (status set to 2 in SBS function )in last kernel call. 
        if(inputTaskStatus[ind]==0 || inputTaskStatus[ind]==1){
          printf("inputTaskStatus %u \n",inputTaskStatus[ind]);
          ui loc = atomicAdd(&local_counter, 1);

          // Add to shared memory
          shared_task[loc] = v;
          shared_status[loc]= inputTaskStatus[ind];

          // If vertex is ustar , set status to 1 (add to C and remove from R )
          if(v==ustarList[i]){
            shared_status[loc] = 1;

          }

        }


      }
    }
  }
  __syncthreads();

  // Add end index of first new task to global offset. 
  if(threadIdx.x==0){
        outputTaskOffset[2*blockId+1]=local_counter;

      }
__syncthreads();


// Process to create  first new task (C - ustar, R - ustar) 

// Each block handles on task and creates two new tasks
  
  for (ui i = blockId; i < prevLevelNodeNum; i+= BLK_NUMS )
  {
    // If ustar is found in last kernel call    
     if(ustarList[i]!=-1){

    // Get start and end index of Task 
     ui start = inputTaskOffset[i];
     ui end = inputTaskOffset[i+1];
     ui total = end-start;

     // Each thread handles one vertex of task 
     for(ui j = threadIdx.x ; j < total ;j+=BLK_DIM)
      {
        ui ind = start+j;
        ui v = inputTaskList[ind];

        // If vertex status is 0 or 1 I.e If vertex is In C or R and vertex is not ustar
        if( (inputTaskStatus[ind]==0 || inputTaskStatus[ind]==1) && v!=ustarList[i]){
          ui loc = atomicAdd(&local_counter, 1);

          // Add to shared memory
          shared_task[loc] = v;
          shared_status[loc] = inputTaskStatus[ind];

        }


      }
     }
  }
   

    __syncthreads();

  // Add end index of second new task to global offset. 

  if(threadIdx.x==0){
        outputTaskOffset[2*blockId+2]=local_counter;

      }
     __syncthreads();


      // Copy from shared memory to Global memory
      // Problem : Not working currently, need to fix code 
     
        ui total;
        
        total  = outputTaskOffset[2*blockId+2];
        if(threadIdx.x==0){
          outputTaskOffset[2*blockId+1] += outputTaskOffset[2*blockId];
          outputTaskOffset[2*blockId+2] += outputTaskOffset[2*blockId];
        }

       

      for (ui i = threadIdx.x; i < total; i += blockDim.x) {

        outputTaskList[*globalCounter + i] = shared_task[i];
        outputTaskStatus[*globalCounter + i] = shared_status[i];
       printf("id %u globalcounter %u value %u status %u inputTaskListSize \n",i,*globalCounter,outputTaskList[*globalCounter + i],outputTaskStatus[*globalCounter + i]);
      }
       __syncthreads();
       if (threadIdx.x == 0) {

       atomicAdd(globalCounter, total);
    }
    __syncthreads();

    }
     
