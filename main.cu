#include "./inc/heuristic.h"
#include "./src/gpuMemoryAllocation.cc"
#include "./src/helpers.cc"



#define CUDA_CHECK_ERROR(kernelName) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA Error in kernel %s, file %s at line %d: %s\n", kernelName, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct subtract_functor {
  const ui x;

  subtract_functor(ui _x): x(_x) {}

  __host__ __device__ ui operator()(const ui & y) const {
    return y - x;
  }
};

bool isServerExit(const string & str) {
  string trimmedStr = str;
  trimmedStr.erase(trimmedStr.find_last_not_of(" \n\r\t") + 1);
  trimmedStr.erase(0, trimmedStr.find_first_not_of(" \n\r\t"));

  transform(trimmedStr.begin(), trimmedStr.end(), trimmedStr.begin(), ::tolower);

  return trimmedStr == "server_exit";
}

inline void preprocessQuery(string msg) {
  istringstream iss(msg);
  vector < ui > argValues;
  ui number, countArgs;
  countArgs = 0;
  while (iss >> number) {
    argValues.push_back(number);
    countArgs++;
  }
  int ind = -1;
  for (ui x = 0; x < limitQueries; x++) {
    if (queries[x].solFlag != 0) {
      ind = x;
      break;
    }
  }
  queries[ind].updateQueryData(argValues[0], argValues[1], argValues[2], argValues[3], argValues[4], totalQuerry, ind);
  totalQuerry++;
  if (queries[ind].isHeu)
    CSSC_heu(ind);
  cout <<"rank "<<worldRank<< " Processing : " << queries[ind] << endl;
  if (queries[ind].kl == queries[ind].ku) {
    cout <<"rank "<<worldRank<< " heuristic find the OPT!" << endl;
    cout <<"rank "<<worldRank<< " Found Solution : " << queries[ind] << endl;
    queries[ind].solFlag = 1;
    
  }else{
    cal_query_dist(queries[ind].QID);
    chkerr(cudaMemcpy(deviceGraph.degree + (ind * n), degree, n * sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(deviceGraph.distance + (ind * n), q_dist, n * sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(deviceGraph.lowerBoundDegree + ind, & (queries[ind].kl), sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(deviceGraph.lowerBoundSize + ind, & (queries[ind].N1), sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(deviceGraph.upperBoundSize + ind, & (queries[ind].N2), sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(deviceGraph.limitDoms + ind, & (queries[ind].limitDoms), sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMemset(initialTask.globalCounter, 0, sizeof(ui)));
    chkerr(cudaMemset(initialTask.entries, 0, INTOTAL_WARPS * sizeof(ui)));
    chkerr(cudaMemset(deviceGraph.newOffset + ((n + 1) * ind), 0, (n + 1) * sizeof(ui)));

    if (queries[ind].kl <= 1)
      queries[ind].ubD = queries[ind].N2 - 1;
    else {
      for (ui d = 1; d <= queries[ind].N2; d++) {
        if (d == 1 || d == 2) {
          if (queries[ind].kl + d > queries[ind].N2) {
            queries[ind].ubD = d - 1;
            break;
          }
        } else {
          ui min_n = queries[ind].kl + d + 1 + floor(d / 3) * (queries[ind].kl - 2);
          if (queries[ind].N2 < min_n) {
            queries[ind].ubD = d - 1;
            break;
          }
        }
      }
    }
    maxN2 = mav(maxN2, queries[ind].N2);

    queries[ind].receiveTimer.restart();

    initialReductionRules << < BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial >>> (deviceGenGraph, deviceGraph, initialTask, n, queries[ind].ubD, initialPartitionSize, ind);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Intial Reduction ");

    ui globalCounter;
    chkerr(cudaMemcpy( & globalCounter, initialTask.globalCounter, sizeof(ui), cudaMemcpyDeviceToHost));

    CompressTask << < BLK_NUM2, BLK_DIM2 >>> (deviceGenGraph, deviceGraph, initialTask, deviceTask, initialPartitionSize, queries[ind].QID, ind, n, partitionSize, TOTAL_WARPS, factor);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Compress ");

    thrust::inclusive_scan(thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * ind)), thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * (ind + 1))), thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * ind)));
    cudaDeviceSynchronize();

    numQueriesProcessing++;

    NeighborUpdate << < BLK_NUMS, BLK_DIM, sharedMemoryUpdateNeigh >>> (deviceGenGraph, deviceGraph, TOTAL_WARPS, ind, n, m);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Neighbor  ");
  }
  
}

inline void processQueries() {
  chkerr(cudaMemset(deviceGraph.flag, 0, limitQueries * sizeof(ui)));

  sharedMemorySizeTask = 2 * WARPS_EACH_BLK * sizeof(ui) + WARPS_EACH_BLK * sizeof(int) + WARPS_EACH_BLK * sizeof(double) + 2 * WARPS_EACH_BLK * sizeof(ui) + maxN2 * WARPS_EACH_BLK * sizeof(ui);

  ProcessTask << < BLK_NUMS, BLK_DIM, sharedMemorySizeTask >>> (
    deviceGenGraph, deviceGraph, deviceTask, partitionSize, factor, maxN2, n, m, dMAX, limitQueries);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Process Task");

  // Indicates the partition offset each warp will use to write new tasks.
  int jump = 0;
  jump = jump >> 1;
  if (jump == 1) {
    jump = TOTAL_WARPS >> 1;
  }

  // This kernel identifies vertices dominated by ustar and sorts them in decreasing order of their connection score.
  FindDoms << < BLK_NUMS, BLK_DIM, sharedMemorySizeDoms >>> (
    deviceGenGraph, deviceGraph, deviceTask, partitionSize, factor, n, m, dMAX);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Find Doms");

  // This kernel writes new tasks, based on ustar and the dominating set, into the task array or buffer. It reads from the buffer and writes to the task array.
  //ui flag = 1 ;
  //chkerr(cudaMemcpy( &flag, &deviceBuffer.outOfMemoryFlag, sizeof(ui),cudaMemcpyDeviceToHost));

  Expand << < BLK_NUMS, BLK_DIM, sharedMemorySizeExpand >>> (
    deviceGenGraph, deviceGraph, deviceTask, deviceBuffer, partitionSize, factor,
    jump, copyLimit, bufferSize, numTaskHost - numReadHost, readLimit, n, m, dMAX);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Expand ");
  RemoveCompletedTasks << < BLK_NUMS, BLK_DIM >>> (deviceGraph, deviceTask, partitionSize, factor);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("Remove Completed ");

  chkerr(cudaMemcpy( & (outMemFlag), deviceBuffer.outOfMemoryFlag, sizeof(ui), cudaMemcpyDeviceToHost));
  if (outMemFlag)
    cout <<"rank "<<worldRank<< " Warning !!!!! buffer out of memory answers will be approx." << outMemFlag << endl;

  chkerr(cudaMemcpy( & tempHost, deviceBuffer.temp, sizeof(ui),
    cudaMemcpyDeviceToHost));
  chkerr(cudaMemcpy( & numReadHost, deviceBuffer.numReadTasks, sizeof(ui),
    cudaMemcpyDeviceToHost));
  chkerr(cudaMemcpy( & numTaskHost, deviceBuffer.numTask, sizeof(ui),
    cudaMemcpyDeviceToHost));

  chkerr(cudaMemcpy(queryStopFlag, deviceGraph.flag, limitQueries * sizeof(ui), cudaMemcpyDeviceToHost));

  for (ui i = 0; i < limitQueries; i++) {
    if ((queryStopFlag[i] == 0) && (queries[i].solFlag == 0)) {
      chkerr(cudaMemcpy( & (queries[i].numRead), deviceGraph.numRead + i, sizeof(ui), cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy( & (queries[i].numWrite), deviceGraph.numWrite + i, sizeof(ui), cudaMemcpyDeviceToHost));
      if ((queries[i].numRead == queries[i].numWrite)) {
        chkerr(cudaMemcpy( & (queries[i].kl), deviceGraph.lowerBoundDegree + i, sizeof(ui), cudaMemcpyDeviceToHost));
        cout <<"rank "<<worldRank<<"Found Solution : " << queries[i] << endl;
        //Send result Data to Rank 0 system 
        queries[i].solFlag = 1;
        numQueriesProcessing--;
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
    //cout << "Num read " << numReadHost << " num Task " << numTaskHost << endl;
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

  thrust::device_ptr < ui > d_input_ptr(deviceTask.numTasks);
  thrust::device_ptr < ui > d_sortedIndex_ptr(deviceTask.sortedIndex);
  thrust::device_ptr < ui > d_mapping_ptr(deviceTask.mapping);

  thrust::device_vector < ui > d_temp_input(d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);

  thrust::sequence(thrust::device, d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);

  thrust::sort_by_key(thrust::device,
    d_temp_input.begin(), d_temp_input.end(),
    d_sortedIndex_ptr,
    thrust::less < ui > ());

  thrust::scatter(thrust::device,
    thrust::make_counting_iterator < ui > (0),
    thrust::make_counting_iterator < ui > (TOTAL_WARPS),
    d_sortedIndex_ptr,
    d_mapping_ptr);

  thrust::transform(thrust::device, d_mapping_ptr, d_mapping_ptr + TOTAL_WARPS, d_mapping_ptr, subtract_from(TOTAL_WARPS - 1));

}


void listenForMessages() {

  msg_queue_server server('g');
  long type = 1;
  //cout<<"rank "<<worldRank<<" listen here "<<endl;

  while (true) {
    if (server.recv_msg(type)) {
      string msg = server.get_msg();
      queryInfo query(totalQuerry, msg);
      totalQuerry++;
      messageQueueMutex.lock();
      messageQueue.push_back(query);
      messageQueueMutex.unlock();
      if (isServerExit(msg))
        break;
    }
  }
}

void processMessageMasterServer() {
  bool stopListening = false;

  vector <MPI_Request> requests(worldSize);
  vector <MPI_Status> status(worldSize);
  
  vector < SystemInfo > systems(worldSize);
  vector < int > nQP(worldSize);

  vector <MPI_Request> endRequests(worldSize);
  vector <MPI_Status> endStatus(worldSize);
  vector <int> endFlag(worldSize);
  vector <SystemStatus> systemStatus(worldSize);
  for(int i = 0; i <worldSize;i++){
    endFlag[i] = 0;
    systemStatus[i] = IDLE;
  }

  systems[0] = {0, 0, 0};
  nQP[0] = 0;
  for (int i = 1; i < worldSize; i++) {
    systems[i] = { i, 0, 1 };
    nQP[i] = 0;
  }
  //cout<<"rank "<<worldRank<<" process here "<<endl;
  int x = 0;
  while (true) {
    if (!stopListening) {
      messageQueueMutex.lock();
      while ((!messageQueue.empty()) && (leastQuery < limitQueries)) {
        
        queryInfo message = messageQueue.front();
        cout<<"rank "<<worldRank<<" read "<<x<<" msg : "<<message.queryString<<endl;
        x++;
        messageQueue.erase(messageQueue.begin());
        messageQueueMutex.unlock();
        ui queryId = message.queryId;
        string msg = message.queryString;
        if (isServerExit(msg)) {
          stopListening = true;
          //send server quit msg to every system 
          for (int i = 1; i < worldSize; i++) {
            MessageType msgType = TERMINATE;
            MPI_Send( & msgType, 1, MPI_INT, i, TAG_MTYPE, MPI_COMM_WORLD);
          }
        } else {

          systems[0].numQueriesProcessing = numQueriesProcessing;

          for (int i = 1; i < worldSize; i++) {
           
            if (systems[i].flag) {

              MPI_Irecv( & nQP[i], 1, MPI_INT, i, TAG_NQP, MPI_COMM_WORLD, & requests[i]);
            }

            MPI_Test( &requests[i], & systems[i].flag, & status[i]);
            if (systems[i].flag) {
              systems[i].numQueriesProcessing = nQP[i];

            }

          }

          cout<<"Num processing ";
          for(ui i=0;i<worldSize;i++){
            cout<<systems[i].numQueriesProcessing<<" ";
          }
          cout<<endl;

          auto leastLoadedSystem = *std::min_element(systems.begin(), systems.end(),
            [](const SystemInfo & a,
              const SystemInfo & b) {
              return a.numQueriesProcessing < b.numQueriesProcessing;
            });
          leastQuery = leastLoadedSystem.numQueriesProcessing + 1;
          if (leastLoadedSystem.rank == 0) {
            cout<<"self 0 got msg "<<msg<<endl;
            preprocessQuery(msg);

          } else {

            if(systemStatus[leastLoadedSystem.rank] == IDLE){
              systemStatus[leastLoadedSystem.rank] = PROCESSING;
              endFlag[leastLoadedSystem.rank] = 1;

            }
            cout<<"Rank 0 sending to rank "<<leastLoadedSystem.rank<<" msg "<<msg<<endl;
            MessageType msgType = PROCESS_MESSAGE;
            MPI_Send( & msgType, 1, MPI_INT, leastLoadedSystem.rank, TAG_MTYPE, MPI_COMM_WORLD);

            MPI_Send(msg.c_str(), msg.length(), MPI_CHAR, leastLoadedSystem.rank, TAG_MSG, MPI_COMM_WORLD);

            // Get confirmation 

          }

        }

        messageQueueMutex.lock();
      }
      messageQueueMutex.unlock();
    }

    if (numQueriesProcessing != 0) {
      processQueries();
    }
    
    for(int i =1 ; i < worldSize ; i ++){
      if(systemStatus[i]==PROCESSING){
        if (endFlag[i]) {

              MPI_Irecv( &systemStatus[i], 1, MPI_INT, i,TAG_TERMINATE, MPI_COMM_WORLD, & endRequests[i]);
          }

        MPI_Test( &endRequests[i], &endFlag[i], & endStatus[i]);


      }
       
    }

    if ((numQueriesProcessing == 0) && (stopListening)){
      bool allTerminatedOrIdle = std::all_of(systemStatus.begin(), systemStatus.end(), [](SystemStatus status) { return status == SystemStatus::TERMINATED || status == SystemStatus::IDLE; });
      if (allTerminatedOrIdle)
        break;
      
    }



  }

}

void processMessageOtherServer() {
  int flag = true;
  MPI_Request request;
  MPI_Status status;
  bool stopListening = false;
  MessageType msgType;
  while (true) {
    if (!stopListening) {
      MPI_Send( &numQueriesProcessing, 1, MPI_INT, 0, TAG_NQP, MPI_COMM_WORLD);

      if (flag) {
        MPI_Irecv( & msgType, 1, MPI_INT, 0, TAG_MTYPE, MPI_COMM_WORLD, & request);
      }

      MPI_Test( & request, & flag, & status);
      if (flag) {
        if (msgType == TERMINATE) {
          stopListening = true;

        } else {
          char buffer[1024];
          MPI_Recv(buffer, 1024, MPI_CHAR, 0, TAG_MSG, MPI_COMM_WORLD, & status);

          int count;
          MPI_Get_count(&status, MPI_CHAR, &count);
          buffer[count] = '\0'
          string msg(buffer, count);
          cout<<"Rank "<<worldRank<<" recieved from  rank 0  msg "<<msg<<endl;

          preprocessQuery(msg);
        }
      }
    }

    if (numQueriesProcessing != 0) {
      processQueries();

    }

    if ((numQueriesProcessing == 0) && (stopListening))
    {
      SystemStatus ss = TERMINATED;
      MPI_Send( &ss, 1, MPI_INT, 0 , TAG_TERMINATE, MPI_COMM_WORLD);
      break;

    }

  }
}


int main(int argc,const char * argv[]) {
  if (argc != 8) {
    cerr << "Server wrong input parameters!" << endl;
    exit(1);
  }

  char** new_argv = new char*[argc];
  for (int i = 0; i < argc; i++) {
      new_argv[i] = const_cast<char*>(argv[i]);
  }

  int mpi_init_result = MPI_Init( & argc, & new_argv);
  if (mpi_init_result != MPI_SUCCESS) {
    cerr << "Error initializing MPI." << endl;
    return 1;
  }

  MPI_Comm_size(MPI_COMM_WORLD, & worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, & worldRank);
  
  cout<<"rank "<<worldRank<<" Size "<<worldSize<<endl;

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  partitionSize = atoi(argv[2]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  bufferSize = atoi(argv[3]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  copyLimit = stod(argv[4]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  readLimit = atoi(argv[5]); // Maximum number of tasks a warp with an empty partition can read from the buffer.
  limitQueries = atoi(argv[6]);
  factor = atoi(argv[7]);

  queries = new queryData[limitQueries];

  for (ui i = 0; i < limitQueries; i++) {
    queryData query;
    queries[i] = query;
  }

  load_graph(filepath);
  core_decomposition_linear_list();

  memoryAllocationGenGraph(deviceGenGraph);
  memoryAllocationGraph(deviceGraph, limitQueries);

  totalQuerry = 0;
  q_dist = new ui[n];
  outMemFlag = 0;

  maxN2 = 0;

  initialPartitionSize = (n / INTOTAL_WARPS) + 1;

  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize, limitQueries, factor);
  memoryAllocationBuffer(deviceBuffer, bufferSize, limitQueries, factor);

  sharedMemorySizeinitial = INTOTAL_WARPS * sizeof(ui);
  sharedMemoryUpdateNeigh = WARPS_EACH_BLK * sizeof(ui);

  queryStopFlag = new ui[limitQueries];

  memset(queryStopFlag, 0, limitQueries * sizeof(ui));

  sharedMemorySizeDoms = WARPS_EACH_BLK * sizeof(ui);
  sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);

  numTaskHost = 0;
  numReadHost = 0;
  tempHost = 0;
  startOffset = 0;
  endOffset = 0;
  numQueriesProcessing = 0;
  
  if (worldRank == 0) {
    leastQuery = 0;
    thread listener(listenForMessages);
    thread processor(processMessageMasterServer);
    listener.join();
    processor.join();
    MPI_Finalize();
  } else {
    processMessageOtherServer();
    MPI_Finalize();
  }

  cudaDeviceSynchronize();
  cout << "End" << endl;
  freeGenGraph(deviceGenGraph);
  freeGraph(deviceGraph);
  freeInterPointer(initialTask);
  freeTaskPointer(deviceTask);
  freeBufferPointer(deviceBuffer);

  return 0;
}