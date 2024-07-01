#include <stdio.h>
#include "./inc/heuristic.h"
#include "./src/helpers.cc"
#include "./src/gpuMemoryAllocation.cu"

int main(int argc,
  const char * argv[]) {

  if (argc != 6) {
    cout << "wrong input parameters!" << endl;
    exit(1);
    exit(1);

  }
  // ./SCS ./graph.txt 6 9 2 100000
  N1 = atoi(argv[2]); //size LB
  N2 = atoi(argv[3]); //size UB
  QID = atoi(argv[4]); //Query vertex ID
  ui paritionSize = atoi(argv[5]);

  const char * filepath = argv[1];
  load_graph(filepath);

  Timer timer;
  StartTime = (double) clock() / CLOCKS_PER_SEC;

  ui jump = TOTAL_WARPS;

  core_decomposition_linear_list();

  // Description: upper bound defined
  ku = miv(core[QID], N2 - 1);
  kl = 0;
  ubD = N2 - 1;
  CSSC_heu();
  cout<< "Heuristic Kl "<< kl << " Ku "<<ku<<endl;

  if(kl==ku){
        cout<<"heuristic find the OPT!"<<endl;
        cout<<"mindeg = "<<kl<<endl;
        cout<<"H.size = "<<H.size()<<endl;
        cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;
        return;
    }
  cal_query_dist();

  deviceGraphPointers deviceGraph;
  memoryAllocationGraph(deviceGraph);

  ui BLK_DIM2 = 1024;
  ui BLK_NUM2 = 1;
  ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;
  ui intialParitionSize = (n / INTOTAL_WARPS) + 1;
  cout << "here " << INTOTAL_WARPS << intialParitionSize << endl;

  deviceInterPointers intialTask;
  memoryAllocationIntialTask(intialTask, INTOTAL_WARPS, intialParitionSize);

  size_t sharedMemrySizeIntial = 32 * sizeof(ui);

  IntialReductionRules << < BLK_NUM2, BLK_DIM2, sharedMemrySizeIntial >>> (deviceGraph, intialTask, n, N2, kl, intialParitionSize);
  cudaDeviceSynchronize();

  ui globalCounter;
  cudaMemcpy( & globalCounter, intialTask.globalCounter, sizeof(ui), cudaMemcpyDeviceToHost);
  cout << " Total " << globalCounter << endl;

  ui * taskOffset;

  taskOffset = new ui[paritionSize];
  memset(taskOffset, 0, paritionSize * sizeof(ui));
  taskOffset[1] = globalCounter;
  taskOffset[paritionSize - 1] = 1;

  deviceTaskPointers deviceTask;
  memoryAllocationTask(deviceTask, TOTAL_WARPS, paritionSize);

  chkerr(cudaMemcpy(deviceTask.taskOffset, taskOffset, paritionSize * sizeof(ui), cudaMemcpyHostToDevice));

  CompressTask << < BLK_NUM2, BLK_DIM2 >>> (deviceGraph, intialTask, deviceTask, intialParitionSize, QID);
  cudaDeviceSynchronize();

  size_t sharedMemrySizeTask = 3 * WARPS_EACH_BLK * sizeof(ui) + WARPS_EACH_BLK * sizeof(int) + WARPS_EACH_BLK * sizeof(double);
  size_t sharedMemrySizeExpand = WARPS_EACH_BLK * sizeof(ui);
  bool stopFlag;
  int c = 0;

  ui * H;
  cudaMalloc((void ** ) & H, (globalCounter + 1) * sizeof(ui));
  cudaMemset(H, 0, sizeof(ui));
  cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS * paritionSize * sizeof(int));

  while (1) {

    cudaMemset(deviceTask.flag, 1, sizeof(bool));

    cudaMemcpy( & stopFlag, deviceTask.flag, sizeof(bool), cudaMemcpyDeviceToHost);

    ProcessTask << < BLK_NUMS, BLK_DIM, sharedMemrySizeTask >>> (deviceGraph, deviceTask, N1, N2, paritionSize, dMAX, H, c);
    cudaDeviceSynchronize();
    jump = jump >> 1;
    Expand << < BLK_NUMS, BLK_DIM, sharedMemrySizeExpand >>> (deviceGraph, deviceTask, N1, N2, paritionSize, dMAX, jump);

    cudaDeviceSynchronize();
    reduce << < BLK_NUMS, BLK_DIM, sharedMemrySizeTask >>> (deviceGraph, deviceTask, paritionSize, N2);
    cudaDeviceSynchronize();
    cudaMemcpy( & stopFlag, deviceTask.flag, sizeof(bool), cudaMemcpyDeviceToHost);

    if (stopFlag) {
      cudaMemcpy( & kl, deviceGraph.lowerBoundDegree, sizeof(ui), cudaMemcpyDeviceToHost);
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;

      break;
    }
    if (jump == 1) {
      jump = TOTAL_WARPS;
    }
    c++;
  }

  freeInterPointer(intialTask);
  freeGraph(deviceGraph);
  freeTaskPointer(deviceTask);

  return 0;

}