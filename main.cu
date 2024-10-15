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

int c =0;
bool serverQuit = false;

bool fileExists(const string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Function to write or append data to the file
void writeOrAppend(const string& filename, const string& data) {
    ofstream file;
    
    // Check if the file exists
    if (fileExists(filename)) {
        // Open the file in append mode if it exists
        file.open(filename, ios::app);
    } else {
        // Open the file in write mode if it doesn't exist
        file.open(filename);
    }
    
    if (file.is_open()) {
        file << data << endl;
        file.close();
    } else {
        cerr << "Unable to open the file." << endl;
    }
}
struct subtract_functor {
  const ui x;

  subtract_functor(ui _x): x(_x) {}

  __host__ __device__ ui operator()(const ui & y) const {
    return y - x;
  }
};

bool isServerExit(const std::string& str) {
    std::string trimmedStr = str;
    trimmedStr.erase(trimmedStr.find_last_not_of(" \n\r\t") + 1);
    trimmedStr.erase(0, trimmedStr.find_first_not_of(" \n\r\t"));

    std::transform(trimmedStr.begin(), trimmedStr.end(), trimmedStr.begin(), ::tolower);

    return trimmedStr == "server_exit";
}



void listenForMessages() {

  msg_queue_server server('g');
  long type = 1;
  while (true) {
    if (server.recv_msg(type)) {
      string msg = server.get_msg();
      queryInfo query(totalQuerry, msg);
      totalQuerry++;
      messageQueueMutex.lock();
      messageQueue.push_back(query);
      messageQueueMutex.unlock();
      if(isServerExit(msg))
      break;
    }
  }
}

void processMessages() {
  while (true) {
    messageQueueMutex.lock();
    while ( (!messageQueue.empty()) && (numQueriesProcessing < limitQueries)) {
      queryInfo message = messageQueue.front();
      messageQueue.erase(messageQueue.begin());
      messageQueueMutex.unlock();

        ui queryId = message.queryId;
        string queryText = message.queryString;

        istringstream iss(queryText);
        vector<ui> argValues;
        ui number, countArgs;
        countArgs = 0;
        if(isServerExit(queryText)){
          serverQuit = true;
          break;
        }

        while (iss >> number) {
          argValues.push_back(number);
          countArgs++;
        }

        if (countArgs != 5) {
          cout << "Client wrong input parameters! " << message << endl;
          continue;
        }
        int ind  = -1;
        for(ui x =0; x < limitQueries;x++){
          if(queries[x].solFlag!=0){
            ind = x;
            break;
          }
        }

        queries[ind].updateQueryData(argValues[0], argValues[1], argValues[2], argValues[3], argValues[4],queryId,ind);
        if (queries[ind].isHeu)
          CSSC_heu(ind);
        if (queries[ind].kl == queries[ind].ku) {
          stringstream ss;
          ss <<queries[ind].N1<< "|" << queries[ind].N2 << "|"<< queries[ind].QID << "|"<< integer_to_string(queries[ind].receiveTimer.elapsed()).c_str() << "|"<< queries[ind].kl << "|"<<"0"<< "|"<<"1";
          writeOrAppend(fileName,ss.str());
          cout << "heuristic find the OPT!" << endl;
          cout << "Found Solution : " << queries[ind] << endl;
          queries[ind].solFlag = 1;
          continue;
        }

        cal_query_dist(queries[ind].QID);
        chkerr(cudaMemcpy(deviceGraph.degree + (ind * n), degree, n * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.distance + (ind * n), q_dist, n * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.lowerBoundDegree + ind, &(queries[ind].kl), sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.lowerBoundSize + ind, &(queries[ind].N1), sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.upperBoundSize + ind, & (queries[ind].N2), sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(deviceGraph.limitDoms + ind, &(queries[ind].limitDoms), sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMemset(initialTask.globalCounter, 0, sizeof(ui)));
        chkerr(cudaMemset(initialTask.entries, 0, INTOTAL_WARPS * sizeof(ui)));
        chkerr(cudaMemset(deviceGraph.newOffset + ((n + 1) * ind),0,(n+1)*sizeof(ui)));


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
        maxN2 = mav(maxN2,queries[ind].N2);

        queries[ind].receiveTimer.restart();

        initialReductionRules << < BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial >>> (deviceGenGraph, deviceGraph, initialTask, n, queries[ind].ubD, initialPartitionSize, ind);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("Intial Reduction ");

        ui globalCounter;
        chkerr(cudaMemcpy( &globalCounter, initialTask.globalCounter, sizeof(ui), cudaMemcpyDeviceToHost));

        ui writeWarp, ntasks, space;
        chkerr(cudaMemcpy( &writeWarp, deviceTask.sortedIndex, sizeof(ui), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy( &ntasks, deviceTask.numTasks + writeWarp, sizeof(ui), cudaMemcpyDeviceToHost));
        ui offsetPsize = partitionSize/factor;
        chkerr(cudaMemcpy( &space, deviceTask.taskOffset + (writeWarp*offsetPsize + ntasks) , sizeof(ui), cudaMemcpyDeviceToHost));
        if(globalCounter>=(partitionSize-space)){
          cout << "Intial Task > partition Size " << message << endl;
          continue;

        }


        CompressTask << < BLK_NUM2, BLK_DIM2 >>> (deviceGenGraph, deviceGraph, initialTask, deviceTask, initialPartitionSize, queries[ind].QID, ind, n,partitionSize,TOTAL_WARPS,factor);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("Compress ");

        thrust::inclusive_scan(thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * ind)), thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * (ind + 1))), thrust::device_ptr < ui > (deviceGraph.newOffset + ((n + 1) * ind)));
        cudaDeviceSynchronize();

        numQueriesProcessing++;

        NeighborUpdate << < BLK_NUMS, BLK_DIM , sharedMemoryUpdateNeigh >>> (deviceGenGraph, deviceGraph,deviceTask, TOTAL_WARPS, ind, n, m,partitionSize,factor);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("Neighbor  ");
        thrust::device_ptr<ui> d_sortedIndex_ptr(deviceTask.sortedIndex);
        thrust::device_ptr<ui> d_mapping_ptr(deviceTask.mapping);

        thrust::device_vector<ui> d_temp_input(d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);

        thrust::sequence(thrust::device, d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);

        thrust::sort_by_key(thrust::device,
                            d_temp_input.begin(), d_temp_input.end(),
                            d_sortedIndex_ptr,
                            thrust::less<ui>());

        thrust::scatter(thrust::device,
                        thrust::make_counting_iterator<ui>(0),
                        thrust::make_counting_iterator<ui>(TOTAL_WARPS),
                        d_sortedIndex_ptr,
                        d_mapping_ptr);


        thrust::transform(thrust::device, d_mapping_ptr, d_mapping_ptr + TOTAL_WARPS, d_mapping_ptr, subtract_from(TOTAL_WARPS-1));
      messageQueueMutex.lock();
    }
    messageQueueMutex.unlock();

    if (numQueriesProcessing != 0) {
      chkerr(cudaMemset(deviceGraph.flag,0, limitQueries * sizeof(ui)));
      chkerr(cudaMemset(deviceTask.doms, 0, TOTAL_WARPS * partitionSize * sizeof(ui)));
      
      sharedMemorySizeTask = 2 * WARPS_EACH_BLK * sizeof(ui) + WARPS_EACH_BLK * sizeof(int) + WARPS_EACH_BLK * sizeof(double) + 2 * WARPS_EACH_BLK * sizeof(ui) + maxN2 * WARPS_EACH_BLK * sizeof(ui);

      ProcessTask << < BLK_NUMS, BLK_DIM, sharedMemorySizeTask >>> (
        deviceGenGraph, deviceGraph, deviceTask, partitionSize, factor,maxN2, n, m,dMAX,limitQueries);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERROR("Process Task");

      // This kernel identifies vertices dominated by ustar and sorts them in decreasing order of their connection score.
      FindDoms << < BLK_NUMS, BLK_DIM, sharedMemorySizeDoms >>> (
        deviceGenGraph, deviceGraph, deviceTask, partitionSize,factor, n, m,dMAX,limitQueries);
      cudaDeviceSynchronize();
     CUDA_CHECK_ERROR("Find Doms");



      // This kernel writes new tasks, based on ustar and the dominating set, into the task array or buffer. It reads from the buffer and writes to the task array.
      //ui flag = 1 ;
      //chkerr(cudaMemcpy( &flag, &deviceBuffer.outOfMemoryFlag, sizeof(ui),cudaMemcpyDeviceToHost));
      Expand << < BLK_NUMS, BLK_DIM, sharedMemorySizeExpand >>> (
        deviceGenGraph, deviceGraph, deviceTask, deviceBuffer, partitionSize,factor, copyLimit, bufferSize, numTaskHost-numReadHost, readLimit, n, m, dMAX,limitQueries);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERROR("Expand ");
      RemoveCompletedTasks<<<BLK_NUMS, BLK_DIM>>>( deviceGraph,deviceTask, partitionSize,factor);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERROR("Remove Completed ");


      chkerr(cudaMemcpy(&(outMemFlag), deviceBuffer.outOfMemoryFlag, sizeof(ui),cudaMemcpyDeviceToHost));
     
      
      chkerr(cudaMemcpy( &tempHost, deviceBuffer.temp, sizeof(ui),
        cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy( &numReadHost, deviceBuffer.numReadTasks, sizeof(ui),
        cudaMemcpyDeviceToHost));
      chkerr(cudaMemcpy( &numTaskHost, deviceBuffer.numTask, sizeof(ui),
        cudaMemcpyDeviceToHost));


      chkerr(cudaMemcpy(queryStopFlag, deviceGraph.flag, limitQueries * sizeof(ui), cudaMemcpyDeviceToHost));

      for (ui i = 0; i < limitQueries; i++) {
          if ((queryStopFlag[i]==0) && (queries[i].solFlag==0)) {
            chkerr(cudaMemcpy( & (queries[i].numRead), deviceGraph.numRead + i, sizeof(ui), cudaMemcpyDeviceToHost));
            chkerr(cudaMemcpy( & (queries[i].numWrite), deviceGraph.numWrite + i, sizeof(ui), cudaMemcpyDeviceToHost));
            if ((queries[i].numRead == queries[i].numWrite)) {
              chkerr(cudaMemcpy( & (queries[i].kl), deviceGraph.lowerBoundDegree + i, sizeof(ui), cudaMemcpyDeviceToHost));
              
              stringstream ss;
              ss <<queries[i].N1<< "|" << queries[i].N2 << "|"<< queries[i].QID << "|"<< integer_to_string(queries[i].receiveTimer.elapsed()).c_str() << "|"<< queries[i].kl << "|"<<"0"<< "|"<<"0";
              writeOrAppend(fileName,ss.str());
              cout << "Found Solution : " << queries[i] << endl;
              queries[i].solFlag = 1;
              numQueriesProcessing--;
            }
          }


      }
       if(outMemFlag){
        for (ui i = 0; i < limitQueries; i++) {
          if ( queries[i].solFlag==0) {
          chkerr(cudaMemcpy( & (queries[i].kl), deviceGraph.lowerBoundDegree + i, sizeof(ui), cudaMemcpyDeviceToHost));
          
          stringstream ss;
          ss <<queries[i].N1<< "|" << queries[i].N2 << "|"<< queries[i].QID << "|"<< integer_to_string(queries[i].receiveTimer.elapsed()).c_str() << "|"<< queries[i].kl << "|"<<"2"<< "|"<<"0";
          writeOrAppend(fileName,ss.str());
          cout <<"Buffer out of memory !"<<endl;
          cout << "Found Solution : " << queries[i] << endl;
          queries[i].solFlag = 1;
          numQueriesProcessing--;
            
          }
       }
       break;
        
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
      c++;
      thrust::device_ptr<ui> d_sortedIndex_ptr(deviceTask.sortedIndex);
      thrust::device_ptr<ui> d_mapping_ptr(deviceTask.mapping);

      thrust::device_vector<ui> d_temp_input(d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);

      thrust::sequence(thrust::device, d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);

      thrust::sort_by_key(thrust::device,
                          d_temp_input.begin(), d_temp_input.end(),
                          d_sortedIndex_ptr,
                          thrust::less<ui>());

      thrust::scatter(thrust::device,
                      thrust::make_counting_iterator<ui>(0),
                      thrust::make_counting_iterator<ui>(TOTAL_WARPS),
                      d_sortedIndex_ptr,
                      d_mapping_ptr);


      thrust::transform(thrust::device, d_mapping_ptr, d_mapping_ptr + TOTAL_WARPS, d_mapping_ptr, subtract_from(TOTAL_WARPS-1));

    }

    if(c==200){
        for (ui i = 0; i < limitQueries; i++) {
          if ( queries[i].solFlag==0) {
          chkerr(cudaMemcpy( & (queries[i].kl), deviceGraph.lowerBoundDegree + i, sizeof(ui), cudaMemcpyDeviceToHost));
          
          stringstream ss;
          ss <<queries[i].N1<< "|" << queries[i].N2 << "|"<< queries[i].QID << "|"<< integer_to_string(queries[i].receiveTimer.elapsed()).c_str() << "|"<< queries[i].kl << "|"<<"1"<< "|"<<"0";
          writeOrAppend(fileName,ss.str());
          cout <<"Levels > 200 !"<<endl;
          cout << "Found Solution : " << queries[i] << endl;
          queries[i].solFlag = 1;
          numQueriesProcessing--;
            
          }
       }
       break;
        
    }

    if(serverQuit && (numQueriesProcessing==0))
    break;

  }
}

int main(int argc, const char * argv[]) {
  if (argc != 8) {
    cout << "Server wrong input parameters!" << endl;
    exit(1);
  }

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  partitionSize = atoi(argv[2]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  bufferSize = atoi(argv[3]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  copyLimit = stod(argv[4]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  readLimit = atoi(argv[5]); // Maximum number of tasks a warp with an empty partition can read from the buffer.
  limitQueries = atoi(argv[6]);
  factor = atoi(argv[7]); 
  
  graphPath = argv[1];
  size_t pos = graphPath.find_last_of("/\\");
  fileName = (pos != string::npos) ? graphPath.substr(pos + 1) : graphPath;

  fileName = "./results/maxdeg/" + fileName;

  if (!fileExists(fileName)) {
      string header = "N1|N2|QID|Time|Degree|Overtime|Heu";
      ofstream file;
      file.open(fileName, ios::app);
          if (file.is_open()) {
              file << header << endl;
              file.close();
          }
  }

  queries = new queryData[limitQueries];
  for(ui i =0; i < limitQueries;i ++ ){
     queryData query;
    queries[i] = query;
  }

  load_graph(filepath);




  core_decomposition_linear_list();

  memoryAllocationGenGraph(deviceGenGraph);
  memeoryAllocationGraph(deviceGraph, limitQueries);

  totalQuerry = 0;
  q_dist = new ui[n];
  outMemFlag = 0;

  maxN2 = 0;
  if (n <= WARPSIZE) {
        BLK_DIM2 = WARPSIZE;
        BLK_NUM2 = 1;
  } else if (n <= BLK_NUMS) {
      BLK_DIM2 = std::ceil(static_cast<float>(n) / WARPSIZE) * WARPSIZE;
      BLK_NUM2 = 1;
  } else {
      BLK_DIM2 = BLK_DIM;
      BLK_NUM2 = std::min(BLK_NUMS, static_cast<int>(std::ceil(static_cast<float>(n) / BLK_DIM2)));
  }

  INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / WARPSIZE;

  initialPartitionSize = static_cast<ui>(std::ceil(static_cast<float>(n) / INTOTAL_WARPS));
  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize, limitQueries,factor);
  memoryAllocationBuffer(deviceBuffer, bufferSize,limitQueries,factor);


  sharedMemorySizeinitial = INTOTAL_WARPS * sizeof(ui);
  sharedMemoryUpdateNeigh = WARPS_EACH_BLK * sizeof(ui);

  queryStopFlag = new ui[limitQueries];

  memset(queryStopFlag,0, limitQueries * sizeof(ui));

  sharedMemorySizeDoms = WARPS_EACH_BLK * sizeof(ui);
  sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);

  numTaskHost = 0;
  numReadHost = 0;
  tempHost = 0;
  startOffset = 0;
  endOffset = 0;

  numQueriesProcessing = 0;
  thread listener(listenForMessages);
	thread processor(processMessages);
  listener.join();
  processor.join();
  cudaDeviceSynchronize();
  cout<<"End"<<endl;
  freeGenGraph(deviceGenGraph);
  freeGraph(deviceGraph);
  freeInterPointer(initialTask);
  freeTaskPointer(deviceTask);
  freeBufferPointer(deviceBuffer);
  cudaDeviceSynchronize();
  cudaDeviceReset();


  return 0;
}
