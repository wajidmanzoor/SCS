#include <stdio.h>
#include "./inc/heuristic.h"
#include "./src/helpers.cc"
#include "./src/gpuMemoryAllocation.cu"
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>



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

int main(int argc,
  const char * argv[]) {

  if (argc != 9) {
    cout << "wrong input parameters!" << endl;
    exit(1);
    exit(1);

  }
  // ./SCS ./graph.txt 6 9 2 100000
  N1 = atoi(argv[2]); //size LB
  N2 = atoi(argv[3]); //size UB
  QID = atoi(argv[4]); //Query vertex ID
  ui partitionSize = atoi(argv[5]);
  ui isHeu = atoi(argv[6]);
  ui limitDoms = atoi(argv[7]);
  ui bufferSize = atoi(argv[8]);


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
  if(isHeu)
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
  ui BLK_NUM2 = 32;
  ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;
  ui initialPartitionSize = (n / INTOTAL_WARPS) + 1;
  cout << "here " << INTOTAL_WARPS << initialPartitionSize << endl;

  deviceInterPointers initialTask;
  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);



  ubD = 0;
  if(kl<=1) ubD = N2-1;
  else{
      for(ui d = 1; d <= N2; d++){
          if(d == 1 || d == 2){
              if(kl + d > N2){
                  ubD = d - 1;
                  break;
              }
          }
          else{
              ui min_n = kl + d + 1 + floor(d/3) * (kl - 2);
              if(N2 < min_n){
                  ubD = d - 1;
                  break;
              }
          }
      }
  }

  size_t sharedMemorySizeinitial = 32 * sizeof(ui);

  initialReductionRules <<< BLK_NUM2, BLK_DIM2, sharedMemorySizeinitial >>> (deviceGraph, initialTask, n, ubD, kl, initialPartitionSize);
  cudaDeviceSynchronize();

  ui globalCounter;
  cudaMemcpy( &globalCounter, initialTask.globalCounter, sizeof(ui), cudaMemcpyDeviceToHost);
  cout << " Total " << globalCounter << endl;

  ui * taskOffset;

  taskOffset = new ui[partitionSize];
  memset(taskOffset, 0, partitionSize * sizeof(ui));
  taskOffset[1] = globalCounter;
  taskOffset[partitionSize - 1] = 1;

  deviceTaskPointers deviceTask;
  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize);

  deviceBufferPointers deviceBuffer;

  memoryAllocationBuffer(deviceBuffer,bufferSize);

  chkerr(cudaMemcpy(deviceTask.taskOffset, taskOffset, partitionSize * sizeof(ui), cudaMemcpyHostToDevice));

  CompressTask <<< BLK_NUM2, BLK_DIM2 >>> (deviceGraph, initialTask, deviceTask, initialPartitionSize, QID);
  cudaDeviceSynchronize();

  size_t sharedMemorySizeTask = 3 * WARPS_EACH_BLK * sizeof(ui) + WARPS_EACH_BLK * sizeof(int) + WARPS_EACH_BLK * sizeof(double);
  size_t sharedMemorySizeExpand = WARPS_EACH_BLK * sizeof(ui);
  bool stopFlag;
  int c = 0;

  cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS * partitionSize * sizeof(int));
 std::string totalTime;
 bool *outOfMemoryFlag, outMemFlag;

 cudaMalloc((void**)&outOfMemoryFlag,sizeof(bool));
 cudaMemset(outOfMemoryFlag,0,sizeof(bool));
 ui *result;
 cudaMalloc((void**)&result,2*sizeof(ui));
  size_t sharedMemrySizeDoms = WARPS_EACH_BLK * sizeof(ui);

 ui tempHost, numReadHost, numTaskHost, startOffset, endOffset,numOffsetHost;



  while (1) {

    cudaMemset(deviceTask.flag, 1, sizeof(bool));

    ProcessTask << < BLK_NUMS, BLK_DIM, sharedMemorySizeTask >>> (deviceGraph, deviceTask, N1, N2, partitionSize, dMAX,result);
    cudaDeviceSynchronize();
    jump = jump >> 1;
    if (jump == 1) {
      jump = TOTAL_WARPS>>1;
    }
    FindDoms<<<BLK_NUMS, BLK_DIM,sharedMemrySizeDoms>>>(deviceGraph, deviceTask,partitionSize,dMAX,c,limitDoms);
    cudaDeviceSynchronize();
    Expand <<< BLK_NUMS, BLK_DIM, sharedMemorySizeExpand >>> (deviceGraph, deviceTask,deviceBuffer, N1, N2, partitionSize, dMAX, jump,outOfMemoryFlag);
    cudaDeviceSynchronize();

    cudaMemcpy(&outMemFlag,outOfMemoryFlag,sizeof(bool),cudaMemcpyDeviceToHost );


    //LeftShift <<<BLK_NUMS, BLK_DIM>>>(deviceBuffer);
    if(outMemFlag){
      cout <<"partition out of memory "<<endl;
      cout<<"Level "<<c<<" jump "<<jump<<endl;
      cudaMemcpy( & kl, deviceGraph.lowerBoundDegree, sizeof(ui), cudaMemcpyDeviceToHost);
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
      totalTime = integer_to_string(timer.elapsed()).c_str();
      break;

    }


    cudaMemcpy(&stopFlag, deviceTask.flag, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tempHost,deviceBuffer.temp,sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(&numReadHost,deviceBuffer.numReadTasks,sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(&numTaskHost,deviceBuffer.numTask,sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(&numOffsetHost,deviceBuffer.numOffset,sizeof(ui),cudaMemcpyDeviceToHost);


    cout<< "Temp "<< tempHost << " Read "<<numReadHost<<" tasks "<<numTaskHost<<" host offset "<<numOffsetHost<<endl;

     if ((stopFlag) && ( numReadHost == 0 )) {
      cudaMemcpy( & kl, deviceGraph.lowerBoundDegree, sizeof(ui), cudaMemcpyDeviceToHost);
      cout << "Max min degree " << kl << endl;
      cout << "time = " << integer_to_string(timer.elapsed()).c_str() << endl;
      totalTime = integer_to_string(timer.elapsed()).c_str();

      break;
    }

    if(numTaskHost == numReadHost){
      cout<<endl<<" num read inside equal "<<endl;

      cudaMemset(deviceBuffer.numTask,0,sizeof(ui));
      cudaMemset(deviceBuffer.numReadTasks,0,sizeof(ui));
      cudaMemset(deviceBuffer.temp,0,sizeof(ui));
      cudaMemset(deviceBuffer.numOffset,0,sizeof(ui));




    }


    if((numReadHost<numTaskHost)&&(numReadHost>0)){
      cout<<" num read inside "<<endl;
      cout<<"thrust cpy"<<endl;

        cudaMemcpy(&startOffset,deviceBuffer.taskOffset+numReadHost,sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(&endOffset,deviceBuffer.taskOffset + numTaskHost,sizeof(ui),cudaMemcpyDeviceToHost);


      thrust::copy(
        thrust::device_ptr<ui>(deviceBuffer.taskOffset + numReadHost),
        thrust::device_ptr<ui>(deviceBuffer.taskOffset + numTaskHost+1),
        thrust::device_ptr<ui>(deviceBuffer.taskOffset)
      );

      thrust::copy(
        thrust::device_ptr<ui>(deviceBuffer.size + numReadHost),
        thrust::device_ptr<ui>(deviceBuffer.size + numTaskHost),
        thrust::device_ptr<ui>(deviceBuffer.size)
      );

      thrust::copy(
        thrust::device_ptr<ui>(deviceBuffer.taskList + startOffset),
        thrust::device_ptr<ui>(deviceBuffer.taskList + endOffset + 1),
        thrust::device_ptr<ui>(deviceBuffer.taskList)
      );

      thrust::copy(
        thrust::device_ptr<ui>(deviceBuffer.statusList + startOffset),
        thrust::device_ptr<ui>(deviceBuffer.statusList + endOffset + 1),
        thrust::device_ptr<ui>(deviceBuffer.statusList)
      );

      thrust::copy(
        thrust::device_ptr<ui>(deviceBuffer.degreeInC + startOffset),
        thrust::device_ptr<ui>(deviceBuffer.degreeInC + endOffset + 1),
        thrust::device_ptr<ui>(deviceBuffer.degreeInC)
      );
    thrust::copy(
        thrust::device_ptr<ui>(deviceBuffer.degreeInR + startOffset),
        thrust::device_ptr<ui>(deviceBuffer.degreeInR + endOffset + 1),
        thrust::device_ptr<ui>(deviceBuffer.degreeInR)
      );

      cudaMemset(deviceBuffer.numTask,numTaskHost-numReadHost,sizeof(ui));
      cudaMemset(deviceBuffer.numReadTasks,0,sizeof(ui));
      cudaMemset(deviceBuffer.temp,numTaskHost-numReadHost,sizeof(ui));
      cudaMemset(deviceBuffer.numOffset,numTaskHost-numReadHost,sizeof(ui));




    }

    c++;
    if(c==4)
    break;
    cout<<"Level "<<c<<" jump "<<jump<<endl;

  }


ui *offsetHost, *taskHost, *statusHost,*degCHost,*degRHost, *sizeHost, *domsHost;
int *ustarHost;
double *consHost;
offsetHost = new ui[TOTAL_WARPS*partitionSize];
taskHost = new ui[TOTAL_WARPS*partitionSize];
statusHost = new ui[TOTAL_WARPS*partitionSize];
degCHost = new ui[TOTAL_WARPS*partitionSize];
degRHost = new ui[TOTAL_WARPS*partitionSize];
sizeHost = new ui[TOTAL_WARPS*partitionSize];
domsHost = new ui[TOTAL_WARPS*partitionSize];
consHost = new double[TOTAL_WARPS*partitionSize];
ustarHost = new int[TOTAL_WARPS*partitionSize];

cudaMemcpy(offsetHost,deviceTask.taskOffset,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(taskHost,deviceTask.taskList,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(statusHost,deviceTask.statusList,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);

cudaMemcpy(degCHost,deviceTask.degreeInC,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(degRHost,deviceTask.degreeInR,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(sizeHost,deviceTask.size,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);

cudaMemcpy(domsHost,deviceTask.doms,TOTAL_WARPS*partitionSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(consHost,deviceTask.cons,TOTAL_WARPS*partitionSize*sizeof(double),cudaMemcpyDeviceToHost);

cudaMemcpy(ustarHost,deviceTask.ustar,TOTAL_WARPS*partitionSize*sizeof(int),cudaMemcpyDeviceToHost);


ui *bufferHost, *statusBHost, *offsetBHost, *sizeBHost ;





bufferHost = new ui[bufferSize];
statusBHost = new ui[bufferSize];
offsetBHost =  new ui[bufferSize];
sizeBHost = new ui[bufferSize];
cudaMemcpy(bufferHost,deviceBuffer.taskList,bufferSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(statusBHost,deviceBuffer.statusList,bufferSize*sizeof(ui),cudaMemcpyDeviceToHost);

cudaMemcpy(offsetBHost,deviceBuffer.taskOffset,bufferSize*sizeof(ui),cudaMemcpyDeviceToHost);
cudaMemcpy(sizeBHost,deviceBuffer.size,bufferSize*sizeof(ui),cudaMemcpyDeviceToHost);




ui *hresult;
hresult = new ui[2];

cudaMemcpy(hresult,result,2*sizeof(ui),cudaMemcpyDeviceToHost);

ui currentEntry,start,end;
ofstream outFile;
outFile.open(getCurrentDateTime()+".txt");
outFile << filepath << " "<<N1<<" "<<N2<<" "<<QID<<endl;
outFile << "Time "<<totalTime<<" Min Max Degree "<<kl<<endl;
for(ui i=0;i<TOTAL_WARPS;i++){
currentEntry = offsetHost[(i+1)*partitionSize-1];
outFile <<"tasks num "<<currentEntry<<" end " <<offsetHost[i*partitionSize+currentEntry]<<endl;

if(currentEntry!= 0 ){
  cout<<"Current entry "<<currentEntry<<endl;
  start = i*partitionSize ;
  end = (i+1)*partitionSize ;
  cout <<" partition "<<i<<" start "<<start<<" end "<<end<<endl;

   cout<<"off ";
  for(ui j = start;j<end;j++){
    cout<< offsetHost[j]<< " ";
  }
  cout<<endl;

  cout<<"Vert ";
  for(ui j = start;j<end;j++){
    cout<< taskHost[j]<< " ";
  }
  cout<<endl;
  cout <<"Stat ";

  for(ui j = start;j<end;j++){
    cout<< statusHost[j]<< " ";
  }
  cout<<endl;
  cout <<"degC ";

  for(ui j = start;j<end;j++){
    cout<< degCHost[j]<< " ";
  }
  cout<<endl;
  cout <<"degR ";
  for(ui j = start;j<end;j++){
    cout<< degRHost[j]<< " ";
  }
  cout<<endl;
  cout<<"size ";
  for(ui j = start;j<end;j++){
    cout<< sizeHost[j]<< " ";
  }
  cout<<endl;

  cout<<"ustar   ";
for(ui i = 0; i < bufferSize; i++ ){
  cout<<ustarHost[i]<< "  ";

}
cout<<endl;



}


//cout << endl;
}
cudaDeviceSynchronize();

cout<<"Buffer ";
for(ui i = 0; i < bufferSize; i++ ){
  cout<<bufferHost[i]<< "  ";

}
cout<<endl;
cout<<"status ";
for(ui i = 0; i < bufferSize; i++ ){
  cout<<statusBHost[i]<< "  ";

}

cout<<endl;
cout<<"offset ";
for(ui i = 0; i < bufferSize; i++ ){
  cout<<offsetBHost[i]<< "  ";

}

cout<<endl;
cout<<"size   ";
for(ui i = 0; i < bufferSize; i++ ){
  cout<<sizeBHost[i]<< "  ";

}

  freeInterPointer(initialTask);
  freeGraph(deviceGraph);
  freeTaskPointer(deviceTask);

  return 0;

}
