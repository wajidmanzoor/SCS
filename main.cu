#include "./inc/heuristic.h"
#include "./src/gpuMemoryAllocation.cc"
#include "./src/helpers.cc"
#include "./src/mpi.cpp"


#define CUDA_CHECK_ERROR(kernelName) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA Error in kernel %s, file %s at line %d: %s\n", kernelName, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

bool isServerExit(const string& str) {
    string trimmedStr = str;
    trimmedStr.erase(trimmedStr.find_last_not_of(" \n\r\t") + 1);
    trimmedStr.erase(0, trimmedStr.find_first_not_of(" \n\r\t"));

    transform(trimmedStr.begin(), trimmedStr.end(), trimmedStr.begin(), ::tolower);

    return trimmedStr == "server_exit";
}

void distributeGlobalParams(GlobalParams &params, int rank) {

    MPI_Datatype mpi_global_params;
    int blocklengths[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, 
                            MPI_UNSIGNED, MPI_DOUBLE, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint offsets[9];

    offsets[0] = offsetof(GlobalParams, n);
    offsets[1] = offsetof(GlobalParams, m);
    offsets[2] = offsetof(GlobalParams, dMAX);
    offsets[3] = offsetof(GlobalParams, partitionSize);
    offsets[4] = offsetof(GlobalParams, bufferSize);
    offsets[5] = offsetof(GlobalParams, copyLimit);
    offsets[6] = offsetof(GlobalParams, readLimit);
    offsets[7] = offsetof(GlobalParams, limitQueries);
    offsets[8] = offsetof(GlobalParams, factor);

    MPI_Type_create_struct(9, blocklengths, offsets, types, &mpi_global_params);
    MPI_Type_commit(&mpi_global_params);

    int ready = 1;
    MPI_Allreduce(MPI_IN_PLACE, &ready, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!ready) {
      cerr << "Error: Not all processes are ready to receive Params data." << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;

    }
       
    MPI_Bcast(&params, 1, mpi_global_params, 0, MPI_COMM_WORLD);

    int received = 1;
    MPI_Allreduce(MPI_IN_PLACE, &received, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!received){
        cerr << "Error: Not all processes received the Params data." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    // Free the MPI datatype
    MPI_Type_free(&mpi_global_params);
}

void distributeQueryDataArray(queryData* queries, int numQueries, int rank) {
    MPI_Datatype mpi_query_data;
    int blocklengths[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_C_BOOL,
                            MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED,
                            MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED,
                            MPI_UNSIGNED};
    MPI_Aint offsets[13];

    offsets[0] = offsetof(queryData, N1);
    offsets[1] = offsetof(queryData, N2);
    offsets[2] = offsetof(queryData, QID);
    offsets[3] = offsetof(queryData, isHeu);
    offsets[4] = offsetof(queryData, limitDoms);
    offsets[5] = offsetof(queryData, kl);
    offsets[6] = offsetof(queryData, ku);
    offsets[7] = offsetof(queryData, ubD);
    offsets[8] = offsetof(queryData, solFlag);
    offsets[9] = offsetof(queryData, numRead);
    offsets[10] = offsetof(queryData, numWrite);
    offsets[11] = offsetof(queryData, querryId);
    offsets[12] = offsetof(queryData, ind);

    MPI_Type_create_struct(13, blocklengths, offsets, types, &mpi_query_data);
    MPI_Type_commit(&mpi_query_data);

    int ready = 1;
    MPI_Allreduce(MPI_IN_PLACE, &ready, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!ready) {
        cerr << "Error: Not all processes are ready to receive queryData array." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // First, broadcast the number of queries
    MPI_Bcast(&numQueries, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Then, broadcast the queryData array
    MPI_Bcast(queries, numQueries, mpi_query_data, 0, MPI_COMM_WORLD);

    int received = 1;
    MPI_Allreduce(MPI_IN_PLACE, &received, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!received) {
        cerr << "Error: Not all processes received the queryData array." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Free the MPI datatype
    MPI_Type_free(&mpi_query_data);
}

int main(int argc, const char * argv[]) {
  if (argc != 8) {
    cerr << "Server wrong input parameters!" << endl;
    exit(1);
  }
  
  int mpi_init_result = MPI_Init(&argc, &argv);
  if (mpi_init_result != MPI_SUCCESS) {
      cerr << "Error initializing MPI." << endl;
      return 1;
  }

  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  partitionSize = atoi(argv[2]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  bufferSize = atoi(argv[3]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  copyLimit = stod(argv[4]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  readLimit = atoi(argv[5]); // Maximum number of tasks a warp with an empty partition can read from the buffer.
  limitQueries = atoi(argv[6]);
  factor = atoi(argv[7]); 

  queryData *queries;
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

  initialPartitionSize = (n / INTOTAL_WARPS) + 1;
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

  return 0;
}
