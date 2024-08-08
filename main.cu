#include <stdio.h>
#include <sys/stat.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <iomanip>
#include <sstream>

#include "./inc/heuristic.h"
#include "./src/gpuMemoryAllocation.cu"
#include "./src/helpers.cc"

struct subtract_functor {
  const ui x;

  subtract_functor(ui _x) : x(_x) {}

  __host__ __device__ ui operator()(const ui& y) const { return y - x; }
};

std::string getCurrentDateTime() {
  // Get current time
  std::time_t now = std::time(nullptr);
  // Convert to tm structure
  std::tm* local_time = std::localtime(&now);

  // Create a stringstream to format the date and time
  std::stringstream ss;
  ss << std::put_time(local_time, "%Y-%m-%d_%H-%M-%S");

  return ss.str();
}

int main(int argc, const char* argv[]) {
  if (argc != 11) {
    cout << "wrong input parameters!" << endl;
    exit(1);
    exit(1);
  }
  // ./SCS ./graph.txt 6 9 2 100000
  N1 = atoi(argv[2]);   // size LB
  N2 = atoi(argv[3]);   // size UB
  QID = atoi(argv[4]);  // Query vertex ID
  ui partitionSize = atoi(argv[5]);
  ui isHeu = atoi(argv[6]);
  ui limitDoms = atoi(argv[7]);
  ui bufferSize = atoi(argv[8]);
  double copyLimit = stod(argv[9]);
  ui readLimit = atoi(argv[10]);

 cout<<"Details"<<endl;

  const char* filepath = argv[1];
  load_graph(filepath);

  Timer timer;
  StartTime = (double)clock() / CLOCKS_PER_SEC;

  ui jump = TOTAL_WARPS;

  core_decomposition_linear_list();

  // Description: upper bound defined
  ku = miv(core[QID], N2 - 1);
  kl = 0;
  ubD = N2 - 1;
  if (isHeu) CSSC_heu();
  cout << "Heuristic Kl " << kl << " Ku " << ku << endl;

  if (kl == ku) {
    cout << "heuristic find the OPT!" << endl;
    cout << "mindeg = " << kl << endl;
    cout << "H.size = " << H.size() << endl;
    cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
    return;
  }
  cal_query_dist();

  deviceGraphPointers deviceGraph;
  memoryAllocationGraph(deviceGraph);

  ui BLK_DIM2 = 1024;
  ui BLK_NUM2 = 32;
  ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;
  ui initialPartitionSize = (n / INTOTAL_WARPS) + 1;
  cout << "here " << INTOTAL_WARPS << initialPartitionSize << endl;

  deviceInterPointers initialTask;
  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);

  ubD = 0;
  if (kl <= 1)
    ubD = N2 - 1;
  else {
    for (ui d = 1; d <= N2; d++) {
      if (d == 1 || d == 2) {
        if (kl + d > N2) {
          ubD = d - 1;
          break;
        }
      } else {
        ui min_n = kl + d + 1 + floor(d / 3) * (kl - 2);
        if (N2 < min_n) {
          ubD = d - 1;
          break;
        }
      }
    }
  }

  size_t sharedMemorySizeinitial = 32 * sizeof(ui);

  initialReductionRules<<<BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial>>>(
      deviceGraph, initialTask, n, ubD, kl, initialPartitionSize);
  cudaDeviceSynchronize();

  ui globalCounter;
  cudaMemcpy(&globalCounter, initialTask.globalCounter, sizeof(ui),
             cudaMemcpyDeviceToHost);
  cout << " Total " << globalCounter << endl;

  ui* taskOffset;

  taskOffset = new ui[partitionSize];
  memset(taskOffset, 0, partitionSize * sizeof(ui));
  taskOffset[1] = globalCounter;
  taskOffset[partitionSize - 1] = 1;

  deviceTaskPointers deviceTask;
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize);




  chkerr(cudaMemcpy(deviceTask.taskOffset, taskOffset,
                    partitionSize * sizeof(ui), cudaMemcpyHostToDevice));

  CompressTask<<<BLK_NUM2, BLK_DIM2>>>(deviceGraph, initialTask, deviceTask,
                                       initialPartitionSize, QID);
  cudaDeviceSynchronize();

  size_t sharedMemorySizeTask = 3 * WARPS_EACH_BLK * sizeof(ui) +
                                WARPS_EACH_BLK * sizeof(int) +
                                WARPS_EACH_BLK * sizeof(double);
  size_t sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);
  bool stopFlag;
  int c = 0;

  cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS * partitionSize * sizeof(int));
  std::string totalTime;
  bool *outOfMemoryFlag, outMemFlag;

  cudaMalloc((void**)&outOfMemoryFlag, sizeof(bool));
  cudaMemset(outOfMemoryFlag, 0, sizeof(bool));
  ui* result;
  cudaMalloc((void**)&result, 2 * sizeof(ui));
  size_t sharedMemrySizeDoms = WARPS_EACH_BLK * sizeof(ui);

  ui tempHost, numReadHost, numTaskHost, startOffset, endOffset;
  numTaskHost = 0;

  freeInterPointer(initialTask);

  deviceBufferPointers deviceBuffer;
  memoryAllocationBuffer(deviceBuffer, bufferSize);

ui *taskOffsetHost;

    taskOffsetHost = new ui[TOTAL_WARPS*partitionSize];
    int zeroP;
  while (1) {
    cudaMemset(deviceTask.flag, 1, sizeof(bool));

    ProcessTask<<<BLK_NUMS, BLK_DIM, sharedMemorySizeTask>>>(
        deviceGraph, deviceTask, N1, N2, partitionSize, dMAX, result);
    cudaDeviceSynchronize();
    jump = jump >> 1;
    if (jump == 1) {
      jump = TOTAL_WARPS >> 1;
    }
    FindDoms<<<BLK_NUMS, BLK_DIM, sharedMemrySizeDoms>>>(
        deviceGraph, deviceTask, partitionSize, dMAX, c, limitDoms);
    cudaDeviceSynchronize();

    Expand<<<BLK_NUMS, BLK_DIM, sharedMemorySizeExpand>>>(
        deviceGraph, deviceTask, deviceBuffer, N1, N2, partitionSize, dMAX,
        jump, outOfMemoryFlag, copyLimit, bufferSize, numTaskHost,readLimit);
    cudaDeviceSynchronize();

    cudaMemcpy(&outMemFlag, outOfMemoryFlag, sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&stopFlag, deviceTask.flag, sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&tempHost, deviceBuffer.temp, sizeof(ui),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&numReadHost, deviceBuffer.numReadTasks, sizeof(ui),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&numTaskHost, deviceBuffer.numTask, sizeof(ui),
               cudaMemcpyDeviceToHost);

    if (outMemFlag) {
      cout << "Buffer out of memory " << endl;
      cout << "Level " << c << " jump " << jump << endl;
      cudaMemcpy(&kl, deviceGraph.lowerBoundDegree, sizeof(ui),
                 cudaMemcpyDeviceToHost);
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
      totalTime = integer_to_string(timer.elapsed()).c_str();
      break;
    }



    if (tempHost > 0)
      cout << endl
           << "Temp " << tempHost << " Read " << numReadHost << " Total Tasks "
           << numTaskHost << endl;

    if ((stopFlag) && (numReadHost == 0) && (numTaskHost == 0)) {
      cudaMemcpy(&kl, deviceGraph.lowerBoundDegree, sizeof(ui),
                 cudaMemcpyDeviceToHost);
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
      totalTime = integer_to_string(timer.elapsed()).c_str();
      break;
    }

    if (numTaskHost == numReadHost) {
      cudaMemset(deviceBuffer.numTask, 0, sizeof(ui));
      cudaMemset(deviceBuffer.numReadTasks, 0, sizeof(ui));
      cudaMemset(deviceBuffer.temp, 0, sizeof(ui));
      cudaMemset(deviceBuffer.writeMutex, 0, sizeof(ui));
      cudaMemset(deviceBuffer.readMutex, 0, sizeof(ui));
      cudaMemset(deviceBuffer.taskOffset, 0, (numReadHost + 1) * sizeof(ui));
    }

    if ((numReadHost < numTaskHost) && (numReadHost > 0)) {
      cudaMemcpy(&startOffset, deviceBuffer.taskOffset + numReadHost,
                 sizeof(ui), cudaMemcpyDeviceToHost);
      cudaMemcpy(&endOffset, deviceBuffer.taskOffset + numTaskHost, sizeof(ui),
                 cudaMemcpyDeviceToHost);

      thrust::transform(
          thrust::device_ptr<ui>(deviceBuffer.taskOffset + numReadHost),
          thrust::device_ptr<ui>(deviceBuffer.taskOffset + numTaskHost + 1),
          thrust::device_ptr<ui>(deviceBuffer.taskOffset),
          subtract_functor(startOffset));

      cudaMemset(deviceBuffer.taskOffset + (numTaskHost - numReadHost + 1), 0,
                 numReadHost * sizeof(ui));

      thrust::copy(thrust::device_ptr<ui>(deviceBuffer.size + numReadHost),
                   thrust::device_ptr<ui>(deviceBuffer.size + numTaskHost),
                   thrust::device_ptr<ui>(deviceBuffer.size));

      thrust::copy(
          thrust::device_ptr<ui>(deviceBuffer.taskList + startOffset),
          thrust::device_ptr<ui>(deviceBuffer.taskList + endOffset),
          thrust::device_ptr<ui>(deviceBuffer.taskList));

      thrust::copy(
          thrust::device_ptr<ui>(deviceBuffer.statusList + startOffset),
          thrust::device_ptr<ui>(deviceBuffer.statusList + endOffset),
          thrust::device_ptr<ui>(deviceBuffer.statusList));

      int justCheck = (int)(numTaskHost - numReadHost);

      cudaMemcpy(deviceBuffer.numTask, &justCheck, sizeof(ui),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(deviceBuffer.temp, &justCheck, sizeof(ui),
                 cudaMemcpyHostToDevice);
      cudaMemset(deviceBuffer.writeMutex, 0, sizeof(ui));
      cudaMemset(deviceBuffer.numReadTasks, 0, sizeof(ui));

      cudaMemset(deviceBuffer.readMutex, 0, sizeof(ui));
    }

    cudaMemcpy(&tempHost, deviceBuffer.temp, sizeof(ui),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&numReadHost, deviceBuffer.numReadTasks, sizeof(ui),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&numTaskHost, deviceBuffer.numTask, sizeof(ui),
               cudaMemcpyDeviceToHost);

    if (tempHost > 0)
      cout << " After Temp " << tempHost << " Read " << numReadHost
           << " raed offset " << numTaskHost << endl;

    c++;

    cudaMemcpy(taskOffsetHost,deviceTask.taskOffset,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(&kl, deviceGraph.lowerBoundDegree, sizeof(ui),
                 cudaMemcpyDeviceToHost);

    zeroP =0;
    for(ui i=0;i<TOTAL_WARPS;i++){
      if(taskOffsetHost[(i+1)*partitionSize-1]==0)
        zeroP ++;
    }

   cout << "Level " << c <<" kl " <<kl<<" Empty Partition " << zeroP << endl;
  }
    cudaDeviceSynchronize();
    freeGraph(deviceGraph);
    freeTaskPointer(deviceTask);
    freeBufferPointer(deviceBuffer);
    cudaDeviceSynchronize();


  return 0;
}



