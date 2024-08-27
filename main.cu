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
#include "./ipc/msgtool.h"

struct subtract_functor {
  const ui x;

  subtract_functor(ui _x): x(_x) {}

  __host__ __device__ ui operator()(const ui & y) const {
    return y - x;
  }
};


void listenForMessages() {

  msg_queue_server server('g');
  long type = 1;
  while (true) {
    if (server.recv_msg(type)) {
      char * msg = server.get_msg();
      queryInfo query(totalQuerry, msg);
      totalQuerry++;
      messageQueueMutex.lock();
      messageQueue.push_back(query);
      messageQueueMutex.unlock();
    }
  }
}

void processMessages() {
  while (true) {
    messageQueueMutex.lock();
    while (!messageQueue.empty()) {
      vector < queryInfo > messages = messageQueue;
      messageQueue.clear();
      messageQueueMutex.unlock();

      for (const auto & message: messages) {

        ui queryId = message.queryId;
        string queryText = message.queryString;

        istringstream iss(queryText);
        vector < ui > argValues;
        ui number, countArgs;
        countArgs = 0;

        while (iss >> number) {
          argValues.push_back(number);
        }
        if (countArgs != 5) {
          cout << "Client Size: wrong input parameters! " << message << endl;
          continue;
        }
        queryData query(argValues[0], argValues[1], argValues[2], argValues[3], argValues[4]);
        queries.push_back(query);
        if (queries[queryId].isHeu)
          CSSC_heu(queryId);
        cout << "Processing : " << "QueryId = " << queryId << ", " << queries[queryId] << endl;
        if (queries[queryId].kl == queries[queryId].ku) {
          cout << "heuristic find the OPT!" << endl;
          cout << "Found Solution : " << "QueryId = " << queryId << ", " << queries[queryId] << endl;
          continue;
        }

        cal_query_dist(queries[queryId].QID);
        chkerr(cudaMemcpy(deviceGraph.degree + (queryId * n), degree, n * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.distance + (queryId * n), q_dist, n * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.lowerBoundDegree + queryId, & (queries[queryId].kl), sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.lowerBoundSize + queryId, & (queries[queryId].N1), sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.upperBoundSize + queryId, & (queries[queryId].N2), sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.limitDoms + queryId, & (queries[queryId].limitDoms), sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMemset(initialTask.globalCounter, 0, sizeof(ui)));
        chkerr(cudaMemset(initialTask.entries, 0, INTOTAL_WARPS * sizeof(ui)));

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
        maxN2 = mav(maxN2,queries[queryId].N2);

        queries[queryId].receiveTimer.restart();

        initialReductionRules << < BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial >>> (deviceGenGraph, deviceGraph, initialTask, n, queries[queryId].ubD, queries[queryId].kl, initialPartitionSize, queryId);
        cudaDeviceSynchronize();

        ui globalCounter;
        chkerr(cudaMemcpy( & globalCounter, initialTask.globalCounter, sizeof(ui), cudaMemcpyDeviceToHost));

        CompressTask << < BLK_NUM2, BLK_DIM2 >>> (deviceGenGraph, deviceGraph, initialTask, deviceTask, initialPartitionSize, queries[queryId].QID, queryId, n,jump);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * queryId)), thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * (queryId + 1))), thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * queryId)));
        cudaDeviceSynchronize();

        numQueriesProcessing++;

        NeighborUpdate << < BLK_NUM2, BLK_DIM2, sharedMemoryUpdateNeigh >>> (deviceGenGraph, deviceGraph, INTOTAL_WARPS, queryId, n, m);
        cudaDeviceSynchronize();


      }

      messageQueueMutex.lock();
    }
    messageQueueMutex.unlock();

    if (numQueriesProcessing != 0) {

      chkerr(cudaMemset(deviceGraph.flag, 0, limitQueries * sizeof(bool)));
      sharedMemorySizeTask = 3 * WARPS_EACH_BLK * sizeof(ui) +
       WARPS_EACH_BLK * sizeof(int) +
       WARPS_EACH_BLK * sizeof(double) + 2 * WARPS_EACH_BLK * sizeof(ui) + maxN2 * WARPS_EACH_BLK * sizeof(ui);

      ProcessTask << < BLK_NUMS, BLK_DIM, sharedMemorySizeTask >>> (
        deviceGenGraph, deviceGraph, deviceTask, partitionSize, maxN2, n, m,dMAX);
      cudaDeviceSynchronize();

      // Indicates the partition offset each warp will use to write new tasks.
      jump = jump >> 1;
      if (jump == 1) {
        jump = TOTAL_WARPS >> 1;
      }

      // This kernel identifies vertices dominated by ustar and sorts them in decreasing order of their connection score.
      FindDoms << < BLK_NUMS, BLK_DIM, sharedMemorySizeDoms >>> (
        deviceGenGraph, deviceGraph, deviceTask, partitionSize, n, m,dMAX);
      cudaDeviceSynchronize();

      // This kernel writes new tasks, based on ustar and the dominating set, into the task array or buffer. It reads from the buffer and writes to the task array.
      Expand << < BLK_NUMS, BLK_DIM, sharedMemorySizeExpand >>> (
        deviceGenGraph, deviceGraph, deviceTask, deviceBuffer, partitionSize,
        jump, outOfMemoryFlag, copyLimit, bufferSize, numTaskHost, readLimit, n, m, dMAX);
      cudaDeviceSynchronize();

      chkerr(cudaMemcpy( & outMemFlag, outOfMemoryFlag, sizeof(bool),
        cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy( & tempHost, deviceBuffer.temp, sizeof(ui),
        cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy( & numReadHost, deviceBuffer.numReadTasks, sizeof(ui),
        cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy( & numTaskHost, deviceBuffer.numTask, sizeof(ui),
        cudaMemcpyDeviceToHost));

      chkerr(cudaMemcpy(queryStopFlag, deviceGraph.flag, limitQueries * sizeof(bool), cudaMemcpyDeviceToHost));

      for (ui i = 0; i < totalQuerry; i++) {
        if (i < queries.size()) {
          if ((queryStopFlag[i]) && (!queries[i].solFlag)) {
            chkerr(cudaMemcpy( & (queries[i].numRead), deviceGraph.numRead + i, sizeof(ui), cudaMemcpyDeviceToHost));
            chkerr(cudaMemcpy( & (queries[i].numWrite), deviceGraph.numWrite + i, sizeof(ui), cudaMemcpyDeviceToHost));
            if ((queries[i].numRead == queries[i].numWrite)) {
              chkerr(cudaMemcpy( & (queries[i].kl), deviceGraph.lowerBoundDegree + i, sizeof(ui), cudaMemcpyDeviceToHost));
              cout << "Found Solution : " << "QueryId = " << i << ", " << queries[i] << endl;
              numQueriesProcessing--;
            }
          }
        }

      }

      if (numTaskHost == numReadHost) {
        chkerr(cudaMemset(deviceBuffer.numTask, 0, sizeof(ui)));
        chkerr(cudaMemset(deviceBuffer.numReadTasks, 0, sizeof(ui)));
        chkerr(cudaMemset(deviceBuffer.temp, 0, sizeof(ui)));
        chkerr(cudaMemset(deviceBuffer.writeMutex, 0, sizeof(ui)));
        chkerr(cudaMemset(deviceBuffer.readMutex, 0, sizeof(ui)));
        chkerr(cudaMemset(deviceBuffer.taskOffset, 0, (numReadHost + 1) * sizeof(ui)));
      }

      // If the number of tasks written to the buffer exceeds the number read at this level, left shift the tasks that were written but not read to the start of the array.
      if ((numReadHost < numTaskHost) && (numReadHost > 0)) {
        chkerr(cudaMemcpy( & startOffset, deviceBuffer.taskOffset + numReadHost,
          sizeof(ui), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy( & endOffset, deviceBuffer.taskOffset + numTaskHost, sizeof(ui),
          cudaMemcpyDeviceToHost));

        thrust::transform(
          thrust::device_ptr < ui > (deviceBuffer.taskOffset + numReadHost),
          thrust::device_ptr < ui > (deviceBuffer.taskOffset + numTaskHost + 1),
          thrust::device_ptr < ui > (deviceBuffer.taskOffset),
          subtract_functor(startOffset));

        chkerr(cudaMemset(deviceBuffer.taskOffset + (numTaskHost - numReadHost + 1), 0,
          numReadHost * sizeof(ui)));

        thrust::copy(thrust::device_ptr < ui > (deviceBuffer.size + numReadHost),
          thrust::device_ptr < ui > (deviceBuffer.size + numTaskHost),
          thrust::device_ptr < ui > (deviceBuffer.size));

        thrust::copy(thrust::device_ptr < ui > (deviceBuffer.queryIndicator + numReadHost),
          thrust::device_ptr < ui > (deviceBuffer.queryIndicator + numTaskHost),
          thrust::device_ptr < ui > (deviceBuffer.queryIndicator));

        thrust::copy(
          thrust::device_ptr < ui > (deviceBuffer.taskList + startOffset),
          thrust::device_ptr < ui > (deviceBuffer.taskList + endOffset),
          thrust::device_ptr < ui > (deviceBuffer.taskList));

        thrust::copy(
          thrust::device_ptr < ui > (deviceBuffer.statusList + startOffset),
          thrust::device_ptr < ui > (deviceBuffer.statusList + endOffset),
          thrust::device_ptr < ui > (deviceBuffer.statusList));

        int justCheck = (int)(numTaskHost - numReadHost);

        chkerr(cudaMemcpy(deviceBuffer.numTask, & justCheck, sizeof(ui),
          cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceBuffer.temp, & justCheck, sizeof(ui),
          cudaMemcpyHostToDevice));
        chkerr(cudaMemset(deviceBuffer.writeMutex, 0, sizeof(ui)));
        chkerr(cudaMemset(deviceBuffer.numReadTasks, 0, sizeof(ui)));

        chkerr(cudaMemset(deviceBuffer.readMutex, 0, sizeof(ui)));
      }
      chkerr(cudaMemset(deviceTask.doms, 0, TOTAL_WARPS * partitionSize * sizeof(ui)));

    }

  }
}

int main(int argc, const char * argv[]) {
  if (argc != 6) {
    cout << "Server wrong input parameters!" << endl;
    exit(1);
  }

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  partitionSize = atoi(argv[2]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  bufferSize = atoi(argv[3]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  copyLimit = stod(argv[4]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  readLimit = atoi(argv[5]); // Maximum number of tasks a warp with an empty partition can read from the buffer.
  load_graph(filepath);
  core_decomposition_linear_list();
  limitQueries = 100;
  
  memoryAllocationGenGraph(deviceGenGraph);
  memeoryAllocationGraph(deviceGraph, limitQueries);

  totalQuerry = 0;
  q_dist = new ui[n];


  jump = TOTAL_WARPS;
  maxN2 = 0;

  initialPartitionSize = (n / INTOTAL_WARPS) + 1;
  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize, limitQueries);
  memoryAllocationBuffer(deviceBuffer, bufferSize);


  sharedMemorySizeinitial = INTOTAL_WARPS * sizeof(ui);
  sharedMemoryUpdateNeigh = WARPS_EACH_BLK * sizeof(ui);

  queryStopFlag = new bool[limitQueries];

  chkerr(cudaMemset(queryStopFlag, true, limitQueries * sizeof(bool)));

  chkerr(cudaMalloc((void ** ) & outOfMemoryFlag, sizeof(bool)));
  chkerr(cudaMemset(outOfMemoryFlag, 0, sizeof(bool)));

  sharedMemorySizeDoms = WARPS_EACH_BLK * sizeof(ui);
  sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);

  numTaskHost = 0;
  numReadHost = 0;
  tempHost = 0;
  startOffset = 0;
  endOffset = 0;

  thread listener(listenForMessages);
  thread processor(processMessages);
  listener.join();
  processor.join();
  cudaDeviceSynchronize();
  freeGenGraph(deviceGenGraph);
  freeGraph(deviceGraph);
  freeInterPointer(initialTask);
  freeTaskPointer(deviceTask);
  freeBufferPointer(deviceBuffer);

  return 0;
}