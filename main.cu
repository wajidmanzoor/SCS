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

bool isServerExit(const std::string& str) {
    std::string trimmedStr = str;
    trimmedStr.erase(trimmedStr.find_last_not_of(" \n\r\t") + 1);
    trimmedStr.erase(0, trimmedStr.find_first_not_of(" \n\r\t"));

    std::transform(trimmedStr.begin(), trimmedStr.end(), trimmedStr.begin(), ::tolower);

    return trimmedStr == "server_exit";
}

void distributeGlobalParams(GlobalParams &params, int rank) {
    if (rank == 0) {
        // Rank 0 already has the correct values, no need to modify
    }

    // Create an MPI datatype for our struct
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

    // Broadcast the struct from rank 0 to all other ranks
    MPI_Bcast(&params, 1, mpi_global_params, 0, MPI_COMM_WORLD);

    // Free the MPI datatype
    MPI_Type_free(&mpi_global_params);
}

void initializeMPIandGPU(int argc, char* argv[], deviceGraphGenPointers &genGraph, deviceGraphPointers &graph, GlobalParams &params) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the GPU device for this process
    cudaSetDevice(rank % CUDA_VISIBLE_DEVICES);

    // Distribute global parameters
    distributeGlobalParams(params, rank);

    // Allocate memory and distribute data for genGraph
    memoryAllocationGenGraph(genGraph, rank);

    // Allocate memory for graph
    memeoryAllocationGraph(graph, params.limitQueries, rank);

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
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

  queries = new queryData[limitQueries];
  for(ui i =0; i < limitQueries;i ++ ){
     queryData query;
    queries[i] = query;
  }

  load_graph(filepath);
  core_decomposition_linear_list();


  deviceGraphGenPointers deviceGenGraph;
  deviceGraphPointers deviceGraph;
  GlobalParams params;

    // Initialize params for rank 0
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) == 0) {
      params.n = n;
      params.m = m;
      params.dMAX = dMAX;
      params.partitionSize = partitionSize;
      params.bufferSize = bufferSize;
      params.copyLimit = copyLimit;
      params.readLimit = readLimit;
      params.limitQueries = limitQueries;
      params.factor = factor;
  }

  initializeMPIandGPU(argc, argv, deviceGenGraph, deviceGraph, params);

  memoryAllocationGenGraph(deviceGenGraph);
  memeoryAllocationGraph(deviceGraph, limitQueries);

  deviceGraphGenPointers genGraph;
  deviceGraphPointers graph;
  initializeMPIandGPU(argc, argv, genGraph, graph, limitQueries);
  
  totalQuerry = 0;
  q_dist = new ui[n];
  outMemFlag = 0;

  jump = TOTAL_WARPS;
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
