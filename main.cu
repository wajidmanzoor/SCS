#include <stdio.h>
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "./inc/heuristic.h"
#include "./src/gpuMemoryAllocation.cu"
#include "./src/helpers.cc"
#include "./ipc/buffer.h"
#include "./ipc/msgtool.h"





oid listen_for_messages() {

    msg_queue_server server('g'); 
    long type = 1;
    while (true) {
        if (server.recv_msg(type)) {
            char* msg = server.get_msg();
            auto recieveTime = chrono::high_resolution_clock::now();
            queryInfo query(totalQuerry,msg,recieveTime);
            totalQuerry++;
            message_queue_mutex.lock();
            message_queue.push_back(query);
            message_queue_mutex.unlock();
        }
    }
}

void process_messages() {
    while (true) {
        message_queue_mutex.lock();
        while (!message_queue.empty()) {
            vector<queryInfo> messages = message_queue;
            message_queue.clear();
            message_queue_mutex.unlock();

            for (const auto& message : messages) {
                auto processTime = chrono::high_resolution_clock::now()
                message.queryProcessedTime = processTime;

                ui queryId = message.queryId;
                string queryText = message.queryString;

                istringstream iss(queryText);
                vector<ui> argValues;
                ui number;

                // Split the string and store each value in the vector
                while (iss >> number) {
                    argValues.push_back(number);
                }
                queryData query(argValues[0],argValues[1],argValues[2],argValues[3],argValues[4]);
                queries.push_back(query);
                if (isHeu) 
                kl = CSSC_heu(queryId);
                cout<< "Processing QID : "<<queryId<<" Query Message: "<<queryText<<endl;
                cout << "Heuristic Kl " <<  queries[queryId].kl << " Ku " << queries[queryId].ku << endl;
                if (queries[queryId].kl == queries[queryId].ku) {
                  cout<< "Found Solution for  QID : "<<queryId<<" Query Message: "<<queryText<<endl;
                  cout << "heuristic find the OPT!" << endl;
                  cout << "mindeg = " << queries[queryId].kl << endl;
                  cout << "H.size = " << H.size() << endl;
                  cout << "time = " << integer_to_string(queries[queryId].timer.elapsed()).c_str() << endl;
                  continue;
                }
                  // Calculate the distance of all verticies from Query vertex. 
                cal_query_dist(queries[queryId].QID);
                chkerr(cudaMemcpy(deviceGraph.degree+(queryId*n), degree, n * sizeof(ui), cudaMemcpyHostToDevice));
                chkerr(cudaMemcpy(deviceGraph.distance+(queryId*n), q_dist, n * sizeof(ui), cudaMemcpyHostToDevice));
                chkerr(cudaMemcpy(deviceGraph.lowerBoundDegree+(queryId*n), queries[queryId].kl,sizeof(ui),cudaMemcpyHostToDevice));

                chkerr(cudaMemset(initialTask.globalCounter,0,sizeof(ui)));
                chkerr(cudaMemset(initialTask.entries,0,INTOTAL_WARPS*sizeof(ui)));
                // Calculate the upper bound for distance from Querry Vertex
                if (queries[queryId].kl <= 1)
                  queries[queryId].ubD = queries[queryId].N2 - 1;
                else {
                  for (ui d = 1; d <= queries[queryId].N2; d++) {
                    if (d == 1 || d == 2) {
                      if (queries[queryId].kl + d > queries[queryId].N2) {
                        queries[queryId].ubD = d - 1;
                        break;
                      }
                    } else {
                      ui min_n = queries[queryId].kl + d + 1 + floor(d / 3) * (queries[queryId].kl - 2);
                      if (queries[queryId].N2 < min_n) {
                        queries[queryId].ubD = d - 1;
                        break;
                      }
                    }
                  }
                }

                initialReductionRules << < BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial >>> (deviceGenGraph, deviceGraph, initialTask, n, queries[queryId].ubD, queries[queryId].kl, initialPartitionSize,queryId);
                cudaDeviceSynchronize();
                ui globalCounter;
                cudaMemcpy( & globalCounter, initialTask.globalCounter, sizeof(ui),cudaMemcpyDeviceToHost);
                cout << " Total " << globalCounter << endl;

                CompressTask << < BLK_NUM2, BLK_DIM2 >>> (deviceGenGraph, deviceGraph, initialTask, deviceTask,initialPartitionSize, queries[queryId].QID,queryId,n);
                cudaDeviceSynchronize();

                thrust::inclusive_scan(thrust::device_ptr < ui > (deviceGraph.newOffset + ((n+1)*queryId) ),thrust::device_ptr < ui > (deviceGraph.newOffset + ((n+1)*(queryId+1))),thrust::device_ptr < ui > (deviceGraph.newOffset+((n+1)*queryId)));
                cudaDeviceSynchronize();

                NeighborUpdate << < BLK_NUM2, BLK_DIM2, sharedMemoryUpdateNeigh >>> (deviceGenGraph, deviceGraph, INTOTAL_WARPS,queryId,n,m);
                cudaDeviceSynchronize();



            }
            //this_thread::sleep_for(chrono::milliseconds(100));

            message_queue_mutex.lock();
        }
        message_queue_mutex.unlock();

        
    }
}


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


int main(int argc,
  const char * argv[]) {
  if (argc != 6) {
    cout << "wrong input parameters!" << endl;
    exit(1);
    exit(1);
  }

  cout << "Details" << endl;

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  ui partitionSize = atoi(argv[2]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  ui bufferSize = atoi(argv[3]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  double copyLimit = stod(argv[4]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  ui readLimit = atoi(argv[5]); // Maximum number of tasks a warp with an empty partition can read from the buffer.
  cout << "File Path = " << filepath << endl;
  cout << "QID = " << QID << endl;

  // Read the graph to CPU
  load_graph(filepath);
  // Calculate Core Values 
  core_decomposition_linear_list();

  deviceGraphGenPointers deviceGenGraph;
  memoryAllocationGenGraph(deviceGenGraph);
  totalQuerry = 0;
  q_dist = new ui[n];
  deviceGraphPointer deviceGraph;

  // Allocates memory for the graph data on the device.
  memeoryAllocationGraph(deviceGraph,100);

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

  // Store task related data for each wrap 
  deviceTaskPointers deviceTask;
  // Allocated memory for task data
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize, 100);

  // Intialize shared memory for intial reduction rules kernel 
  size_t sharedMemorySizeinitial = INTOTAL_WARPS * sizeof(ui);
  size_t sharedMemoryUpdateNeigh = WARPS_EACH_BLK * sizeof(ui);

  // Stores task data in buffer. 
  deviceBufferPointers deviceBuffer;

  // Allocate memory for buffer. 
  memoryAllocationBuffer(deviceBuffer, bufferSize);

  size_t sharedMemorySizeTask = 3 * WARPS_EACH_BLK * sizeof(ui) +
    WARPS_EACH_BLK * sizeof(int) +
    WARPS_EACH_BLK * sizeof(double) + 2 * WARPS_EACH_BLK * sizeof(ui) + N2 * WARPS_EACH_BLK * sizeof(ui);

  // Shared memory size for expand Task kernel. 
  size_t sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);
  
  // Flag indicating whether tasks are remaining in the task array: 1 if no tasks are left, 0 if at least one task is left.
  bool *stopFlag;
  stopFlag = new bool[100];

  memset(stopFlag,true,100*sizeof(bool));

  int c = 0;

  // Intialize ustar array to -1. 
  cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS * partitionSize * sizeof(int));

  // Flag indicating if the buffer is full.
  bool * outOfMemoryFlag, outMemFlag;

  // Allocate memory for the flag and initialize it to zero.
  cudaMalloc((void ** ) & outOfMemoryFlag, sizeof(bool));
  cudaMemset(outOfMemoryFlag, 0, sizeof(bool));

  // Shared memory size for Find dominated set kernel 
  size_t sharedMemrySizeDoms = WARPS_EACH_BLK * sizeof(ui);

  ui tempHost, numReadHost, numTaskHost, startOffset, endOffset;
  numTaskHost = 0;
  numReadHost = 0;
  tempHost = 0;
  startOffset = 0;
  endOffset = 0;

  thread listener(listen_for_messages);
  thread processor(process_messages);

  StartTime = (double) clock() / CLOCKS_PER_SEC;

  // Indicates the partition offset each warp will use to write new tasks.
  ui jump = TOTAL_WARPS;

  // Shared memory size for Process Task kernel.

  std::string totalTime;

  

  /*while (1) {
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

  cudaDeviceSynchronize();*/

  // Free Memory
  freeGraph(deviceGraph);
  freeTaskPointer(deviceTask);
  freeBufferPointer(deviceBuffer);
  cudaDeviceSynchronize();

  return 0;
}