#include <stdio.h>
#include <sys/stat.h>
#include <iomanip>
#include <sstream>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "./inc/heuristic.h"
#include "./src/gpuMemoryAllocation.cu"
#include "./src/helpers.cc"

struct subtract_functor {
  /**
   * Subtract a scalar value from a device vector. 
   */
  const ui x;

  subtract_functor(ui _x): x(_x) {}

  __host__ __device__ ui operator()(const ui & y) const {
    return y - x;
  }
};

std::string getCurrentDateTime() {
  /**
   * Generates a string representing the local time.
   */
  std::time_t now = std::time(nullptr);
  std::tm * local_time = std::localtime( & now);

  std::stringstream ss;
  ss << std::put_time(local_time, "%Y-%m-%d_%H-%M-%S");

  return ss.str();
}

int main(int argc,
  const char * argv[]) {
  if (argc != 11) {
    cout << "wrong input parameters!" << endl;
    exit(1);
    exit(1);
  }
  // ./SCS ./graph.txt 6 9 2 100000
  N1 = atoi(argv[2]); // Size lower bound
  N2 = atoi(argv[3]); // Size upper bound 
  QID = atoi(argv[4]); // Query vertex ID
  ui partitionSize = atoi(argv[5]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  ui isHeu = atoi(argv[6]); // Indicates whether to run the Huetrictic before the SCS: 1 to run, 0 to skip.
  ui limitDoms = atoi(argv[7]); // Specifies the maximum number of DOMs the algorithm will use for task expansion.
  ui bufferSize = atoi(argv[8]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  double copyLimit = stod(argv[9]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  ui readLimit = atoi(argv[10]); // Maximum number of tasks a warp with an empty partition can read from the buffer.

  cout << "Details" << endl;

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  cout << "File Path = " << filepath << endl;
  cout << "QID = " << QID << endl;

  // Read the graph to CPU
  load_graph(filepath);

  Timer timer;
  StartTime = (double) clock() / CLOCKS_PER_SEC;

  // Indicates the partition offset each warp will use to write new tasks.
  ui jump = TOTAL_WARPS;

  // Calculate Core Values 
  core_decomposition_linear_list();

  // Initialize the upper bound for the maximum minimum degree.
  ku = miv(core[QID], N2 - 1);

  // Initialize the lower bound for the maximum minimum degree.
  kl = 0;

  // Initialize the upper bound for distance from Querry vertex.
  ubD = N2 - 1;

  // Run all three heuristic algorithms and to get the lower bound for the maximum minimum degree.
  if (isHeu) CSSC_heu();
  cout << "Heuristic Kl " << kl << " Ku " << ku << endl;

  // If the lower bound and upper bound of the maximum minimum degree are the same, exit and return the result.
  if (kl == ku) {
    cout << "heuristic find the OPT!" << endl;
    cout << "mindeg = " << kl << endl;
    cout << "H.size = " << H.size() << endl;
    cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
    return;
  }

  // Calculate the distance of all verticies from Query vertex. 
  cal_query_dist();

  // Stores graph-related data on the device.
  deviceGraphPointers deviceGraph;

  // Allocates memory for the graph data on the device.
  memoryAllocationGraph(deviceGraph);

  // Block dimension for the initial reduction rules kernel.
  ui BLK_DIM2 = 1024;
  ui BLK_NUM2 = 32;
  ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;

  // Initialize the number of elements each warp will process.
  ui initialPartitionSize = (n / INTOTAL_WARPS) + 1;

  // Stores vertices after applying initial reduction rules 
  deviceInterPointers initialTask;

  // Allocated memory for intermediate results
  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);

  // Calculate the upper bound for distance from Querry Vertex
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

  // Intialize shared memory for intial reduction rules kernel 
  size_t sharedMemorySizeinitial = 32 * sizeof(ui);

  // Applies intial reduction rules. (vertices with core value less than the lower bound of maximum minimum degree and distance greater than the upper bound from the query vertex are removed).
  initialReductionRules << < BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial >>> (
    deviceGraph, initialTask, n, ubD, kl, initialPartitionSize);
  cudaDeviceSynchronize();

  // Stores the total number of vertex left after applying intial reduction rules. 
  ui globalCounter;
  cudaMemcpy( & globalCounter, initialTask.globalCounter, sizeof(ui),
    cudaMemcpyDeviceToHost);
  cout << " Total " << globalCounter << endl;

  // Store task related data for each wrap 
  deviceTaskPointers deviceTask;
  // Allocated memory for task data
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize);

  // Initialize the task offset for the initial task (C = query vertex, R = remaining vertices after applying reduction rules).
  ui * taskOffset;
  taskOffset = new ui[partitionSize];
  memset(taskOffset, 0, partitionSize * sizeof(ui));
  taskOffset[1] = globalCounter;
  taskOffset[partitionSize - 1] = 1;

  // Copy the task offset data to the device.
  chkerr(cudaMemcpy(deviceTask.taskOffset, taskOffset,
    partitionSize * sizeof(ui), cudaMemcpyHostToDevice));

  // Copies data from the intermediate array to the task array, updates status, degree, and recalculates the total number of neighbor for each vertex after vertex removal.
  CompressTask << < BLK_NUM2, BLK_DIM2 >>> (deviceGraph, initialTask, deviceTask,
    initialPartitionSize, QID);
  cudaDeviceSynchronize();

  // Calculate the inclusive sum of the total neighbor array to determine the neighbor offset.
  thrust::inclusive_scan(thrust::device_ptr < ui > (deviceGraph.newOffset),
    thrust::device_ptr < ui > (deviceGraph.newOffset + n + 1),
    thrust::device_ptr < ui > (deviceGraph.newOffset));

  cudaDeviceSynchronize();

  // Shared Memory for Neighbor Update kernel 
  size_t sharedMemoryUpdateNeigh = WARPS_EACH_BLK * sizeof(ui);

  // Removes vertices from the neighbor list that were eliminated from the graph after applying reduction rules.
  NeighborUpdate << < BLK_NUM2, BLK_DIM2, sharedMemoryUpdateNeigh >>> (deviceGraph, n, INTOTAL_WARPS);
  cudaDeviceSynchronize();

  // Shared memory size for Process Task kernel.
  size_t sharedMemorySizeTask = 3 * WARPS_EACH_BLK * sizeof(ui) +
    WARPS_EACH_BLK * sizeof(int) +
    WARPS_EACH_BLK * sizeof(double) + 2 * WARPS_EACH_BLK * sizeof(ui) + N2 * WARPS_EACH_BLK * sizeof(ui);

  // Shared memory size for expand Task kernel. 
  size_t sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);

  // Flag indicating whether tasks are remaining in the task array: 1 if no tasks are left, 0 if at least one task is left.
  bool stopFlag;

  int c = 0;

  // Intialize ustar array to -1. 
  cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS * partitionSize * sizeof(int));

  std::string totalTime;

  // Flag indicating if the buffer is full.
  bool * outOfMemoryFlag, outMemFlag;

  // Allocate memory for the flag and initialize it to zero.
  cudaMalloc((void ** ) & outOfMemoryFlag, sizeof(bool));
  cudaMemset(outOfMemoryFlag, 0, sizeof(bool));

  // 
  ui * result;
  cudaMalloc((void ** ) & result, 2 * sizeof(ui));

  // Shared memory size for Find dominated set kernel 
  size_t sharedMemrySizeDoms = WARPS_EACH_BLK * sizeof(ui);

  // Stores the total number of tasks written to the buffer, total number read from the buffer, 
  // and the start and end offsets of tasks that were written but not yet read. All values pertain to the current level.
  ui tempHost, numReadHost, numTaskHost, startOffset, endOffset;
  numTaskHost = 0;
  numReadHost = 0;
  tempHost = 0;
  startOffset = 0;
  endOffset = 0;

  // Free the memory that stores the intermediate results. 
  freeInterPointer(initialTask);

  // Stores task data in buffer. 
  deviceBufferPointers deviceBuffer;

  // Allocate memory for buffer. 
  memoryAllocationBuffer(deviceBuffer, bufferSize);

  while (1) {
    // This kernel applies all three reduction rules and computes the minimum degree, ustar, and upper bound for the maximum minimum degree for each task.
    ProcessTask << < BLK_NUMS, BLK_DIM, sharedMemorySizeTask >>> (
      deviceGraph, deviceTask, N1, N2, partitionSize, dMAX, result);
    cudaDeviceSynchronize();

    // Indicates the partition offset each warp will use to write new tasks.
    jump = jump >> 1;
    if (jump == 1) {
      jump = TOTAL_WARPS >> 1;
    }

    // This kernel identifies vertices dominated by ustar and sorts them in decreasing order of their connection score.
    FindDoms << < BLK_NUMS, BLK_DIM, sharedMemrySizeDoms >>> (
      deviceGraph, deviceTask, partitionSize, dMAX, c, limitDoms);
    cudaDeviceSynchronize();

    // This kernel writes new tasks, based on ustar and the dominating set, into the task array or buffer. It reads from the buffer and writes to the task array.
    Expand << < BLK_NUMS, BLK_DIM, sharedMemorySizeExpand >>> (
      deviceGraph, deviceTask, deviceBuffer, N1, N2, partitionSize, dMAX,
      jump, outOfMemoryFlag, copyLimit, bufferSize, numTaskHost, readLimit);
    cudaDeviceSynchronize();

    cudaMemcpy( & outMemFlag, outOfMemoryFlag, sizeof(bool),
      cudaMemcpyDeviceToHost);
    cudaMemcpy( & stopFlag, deviceTask.flag, sizeof(bool),
      cudaMemcpyDeviceToHost);
    cudaMemcpy( & tempHost, deviceBuffer.temp, sizeof(ui),
      cudaMemcpyDeviceToHost);
    cudaMemcpy( & numReadHost, deviceBuffer.numReadTasks, sizeof(ui),
      cudaMemcpyDeviceToHost);
    cudaMemcpy( & numTaskHost, deviceBuffer.numTask, sizeof(ui),
      cudaMemcpyDeviceToHost);

    // If buffer is out of memory, exit and return the result.
    if (outMemFlag) {
      cout << "Buffer out of memory " << endl;
      cout << "Level " << c << endl;
      cudaMemcpy( & kl, deviceGraph.lowerBoundDegree, sizeof(ui),
        cudaMemcpyDeviceToHost);
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
      totalTime = integer_to_string(timer.elapsed()).c_str();
      break;
    }

    // if no tasks are left is task array and buffer, exit and return the result
    if ((stopFlag) && (numReadHost == 0) && (numTaskHost == 0)) {
      cudaMemcpy( & kl, deviceGraph.lowerBoundDegree, sizeof(ui),
        cudaMemcpyDeviceToHost);
      cout << "Level " << c << endl;
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
      totalTime = integer_to_string(timer.elapsed()).c_str();
      break;
    }

    // If the number of tasks written to and read from the buffer at this level is the same, reset the buffer.
    if (numTaskHost == numReadHost) {
      cudaMemset(deviceBuffer.numTask, 0, sizeof(ui));
      cudaMemset(deviceBuffer.numReadTasks, 0, sizeof(ui));
      cudaMemset(deviceBuffer.temp, 0, sizeof(ui));
      cudaMemset(deviceBuffer.writeMutex, 0, sizeof(ui));
      cudaMemset(deviceBuffer.readMutex, 0, sizeof(ui));
      cudaMemset(deviceBuffer.taskOffset, 0, (numReadHost + 1) * sizeof(ui));
    }

    // If the number of tasks written to the buffer exceeds the number read at this level, left shift the tasks that were written but not read to the start of the array.
    if ((numReadHost < numTaskHost) && (numReadHost > 0)) {
      cudaMemcpy( & startOffset, deviceBuffer.taskOffset + numReadHost,
        sizeof(ui), cudaMemcpyDeviceToHost);
      cudaMemcpy( & endOffset, deviceBuffer.taskOffset + numTaskHost, sizeof(ui),
        cudaMemcpyDeviceToHost);

      thrust::transform(
        thrust::device_ptr < ui > (deviceBuffer.taskOffset + numReadHost),
        thrust::device_ptr < ui > (deviceBuffer.taskOffset + numTaskHost + 1),
        thrust::device_ptr < ui > (deviceBuffer.taskOffset),
        subtract_functor(startOffset));

      cudaMemset(deviceBuffer.taskOffset + (numTaskHost - numReadHost + 1), 0,
        numReadHost * sizeof(ui));

      thrust::copy(thrust::device_ptr < ui > (deviceBuffer.size + numReadHost),
        thrust::device_ptr < ui > (deviceBuffer.size + numTaskHost),
        thrust::device_ptr < ui > (deviceBuffer.size));

      thrust::copy(
        thrust::device_ptr < ui > (deviceBuffer.taskList + startOffset),
        thrust::device_ptr < ui > (deviceBuffer.taskList + endOffset),
        thrust::device_ptr < ui > (deviceBuffer.taskList));

      thrust::copy(
        thrust::device_ptr < ui > (deviceBuffer.statusList + startOffset),
        thrust::device_ptr < ui > (deviceBuffer.statusList + endOffset),
        thrust::device_ptr < ui > (deviceBuffer.statusList));

      int justCheck = (int)(numTaskHost - numReadHost);

      cudaMemcpy(deviceBuffer.numTask, & justCheck, sizeof(ui),
        cudaMemcpyHostToDevice);
      cudaMemcpy(deviceBuffer.temp, & justCheck, sizeof(ui),
        cudaMemcpyHostToDevice);
      cudaMemset(deviceBuffer.writeMutex, 0, sizeof(ui));
      cudaMemset(deviceBuffer.numReadTasks, 0, sizeof(ui));

      cudaMemset(deviceBuffer.readMutex, 0, sizeof(ui));
    }

    cudaMemcpy( & kl, deviceGraph.lowerBoundDegree, sizeof(ui),
      cudaMemcpyDeviceToHost);

    // Reset stop flag to 1. 
    cudaMemset(deviceTask.flag, 1, sizeof(bool));

    // Reset domination array to zero
    cudaMemset(deviceTask.doms, 0, TOTAL_WARPS * partitionSize * sizeof(ui));

    c++;

    if (c == 100)
      break;

    cout << "Level " << c << " kl " << kl << endl;
  }

  cudaDeviceSynchronize();

  // Free Memory
  freeGraph(deviceGraph);
  freeTaskPointer(deviceTask);
  freeBufferPointer(deviceBuffer);
  cudaDeviceSynchronize();

  return 0;
}