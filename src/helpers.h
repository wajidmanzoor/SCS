#include <limits.h>
#include <cuda_runtime.h>

#define BLK_NUMS 2
#define BLK_DIM 641
#define TOTAL_THREAD (BLK_NUMS*BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM/32)
#define TOTAL_WARPS (BLK_NUMS*WARPS_EACH_BLK)



__device__ int findIndexKernel(ui *arr,ui start,ui end ,ui target)
{

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

    ui count = 0;
    double score;
    ui minVal = UINT_MAX;
    ui secondMinVal = UINT_MAX;
    double maxScore = 0;
    int ustar = -1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int wrapId = idx / 32;
    int laneId = idx % 32;
    ui ubD;
    for(ui i = wrapId; i < numTasks; i+=TOTAL_WARPS)
    {
      ui start = tasksOffset[i];
      ui end = tasksOffset[i+1];
      ui total = end-start;

      for(ui j = laneId; j < total ;j+=32)
      {
      printf(" wrap id %u start %u end %u \n",i,start,end);
        int id = j;
        int index = start+id;
        ui v = taskList[start+id];
        ui stat = taskStatus[start+id];
        ui start_n = neighborOffset[v];
        ui end_n = neighborOffset[v+1];
        score = 0; // add later degree / maxdegree
        if(taskStatus[start+id]==0)
        { 
          score += (double) degree[v]/dmax;
          int resIndex; 
          for(int k = start_n; k <end_n;k++)
          {
            resIndex = findIndexKernel(taskList,start,end,neighbors[k]);
            if(taskStatus[resIndex]==1)
            {
                score += (double) 1 / degree[neighbors[k]];
            }

          }
        }

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
          ustar = __shfl_sync(0xFFFFFFFF, ustar,0);
        //printf(" id : %d score %f vertex %u ustar %d \n ", start+id, score,v,taskList[ustar]);


        if (taskStatus[start + id]==1)
        {
          int resultIndex;
          for(int k = start_n; k <end_n;k++)
          {
            resultIndex = findIndexKernel(taskList,start,end,neighbors[k]);
            if (taskStatus[resultIndex]==1)
            {
              count ++;
            }
          }
        }

        printf("Thread Id :  %d wrapId: %d laneId :  %d  Value :  %u Count: %d status %d \n", idx, i,j, v,count,stat);
        warpMin2(count, minVal, secondMinVal);
        if(j==0){
          if(secondMinVal==UINT_MAX){
            secondMinVal = minVal;
          }
        }
        ui boardcastValue = __shfl_sync(0xFFFFFFFF, secondMinVal,0);

        for (int offset = 16; offset > 0; offset /= 2) {
        stat += __shfl_down_sync(0xFFFFFFFF, stat, offset);
        }
        ui current_size = __shfl_sync(0xFFFFFFFF, stat,0);
        if(j==0)
        {
          if(lowerBoundSize <= current_size && current_size <= upperBoundSize)
          {
            atomicMax(lowerBoundDegree,boardcastValue);
            printf(" Lower bound degree %u boardcastValue %u \n",*lowerBoundDegree,boardcastValue);

          }
          if(current_size<upperBoundSize){
          ustarList[i] = (int) taskList[ustar];
          //printf(" wrap id %u value %u \n",i,ustarList[i] );
          }


        }
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
        ui uperboundDistance =  __shfl_sync(0xFFFFFFFF, ubD,0);


        if(uperboundDistance < distanceFromQID[j]){
          taskStatus[start+id]=2;
        }
      }

    }

  }

__global__ void IntialReductionRules(ui *neighborsOffset,ui *neighbors,ui *degree,ui *distanceFromQID, ui qid, ui vertexSize , ui *taskList, ui * taskStatus, ui upperBoundSize, ui *globalCounter)
{

  extern __shared__ ui shared_memory[];
  ui *shared_task = shared_memory;
  ui *shared_status = &shared_memory[blockDim.x];
  

  ui idx = blockIdx.x * blockDim.x + threadIdx.x;
  ui thread_id = threadIdx.x;
   __shared__ ui local_counter;

  if (thread_id == 0)
  {
        local_counter = 0;
    }
  __syncthreads();


  for (ui i = idx; i < vertexSize; i += TOTAL_THREAD) {
      if (distanceFromQID[i] < upperBoundSize - 1) {
                ui loc = atomicAdd(&local_counter, 1);
                shared_task[loc] = i;
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



    for (ui i = thread_id; i < local_counter; i += blockDim.x) {
        taskList[*globalCounter + i] = shared_task[i];
        taskStatus[*globalCounter + i] = shared_status[i];
        //printf("id %u globalcounter %u value %u taskStatus %u \n",i,*globalCounter,taskList[*globalCounter + i],taskStatus[*globalCounter + i]);

    }
     __syncthreads();
    if (thread_id == 0) {

       atomicAdd(globalCounter, local_counter);

    }
     __syncthreads();


}

__global__ void Branching(int *ustarList,ui *inputTaskList,ui *inputTaskOffset,ui *inputTaskStatus, ui *globalCounter, int inputTaskListSize, ui *outputTaskOffset,ui *outputTaskList,ui *outputTaskStatus, ui prevLevelNodeNum,ui currentLevelNodeNum)
{

  extern __shared__ ui shared_memory[];
  ui *shared_task = shared_memory;
  ui *shared_status = &shared_memory[blockDim.x];

  __shared__ ui local_counter;


  ui blockId = blockIdx.x;

  if(threadIdx.x==0){
    local_counter=0;

  }
   __syncthreads();
  

  for (ui i = blockId; i < prevLevelNodeNum; i+= BLK_NUMS )
  {
    if(ustarList[i]!=-1){
     ui start = inputTaskOffset[i];
     ui end = inputTaskOffset[i+1];
     ui total = end-start;
     
     for(ui j = threadIdx.x ; j < total ;j+=BLK_DIM)
      {
        ui ind = start+j;
        ui v = inputTaskList[ind];
        if(inputTaskStatus[ind]==0 || inputTaskStatus[ind]==1){
          printf("inputTaskStatus %u \n",inputTaskStatus[ind]);
          ui loc = atomicAdd(&local_counter, 1);
          shared_task[loc] = v;
          shared_status[loc]= inputTaskStatus[ind];
          if(v==ustarList[i]){
            shared_status[loc] = 1;

          }

        }


      }
    }
  }
  __syncthreads();

  if(threadIdx.x==0){
        outputTaskOffset[2*blockId+1]=local_counter;

      }
__syncthreads();
  for (ui i = blockId; i < prevLevelNodeNum; i+= BLK_NUMS )
  {
     if(ustarList[i]!=-1){
     ui start = inputTaskOffset[i];
     ui end = inputTaskOffset[i+1];
     ui total = end-start;
     for(ui j = threadIdx.x ; j < total ;j+=BLK_DIM)
      {
        ui ind = start+j;
        ui v = inputTaskList[ind];
        if( (inputTaskStatus[ind]==0 || inputTaskStatus[ind]==1) && v!=ustarList[i]){
          ui loc = atomicAdd(&local_counter, 1);
          shared_task[loc] = v;
          shared_status[loc] = inputTaskStatus[ind];

        }


      }
     }
  }
   

    __syncthreads();

  if(threadIdx.x==0){
        outputTaskOffset[2*blockId+2]=local_counter;

      }
     __syncthreads();

     
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
     
