#include <stdio.h>
#include "./src/Graph.h"
#include "./src/helpers.cc"
#include "./src/gpuMemoryAllocation.cu"

int main(int argc, const char * argv[] ) {

    if(argc!=6){
        cout<<"wrong input parameters!"<<endl;exit(1);
        exit(1);

    }
    // ./SCS ./graph.txt 6 9 2 100000
    N1 = atoi(argv[2]); //size LB
    N2 = atoi(argv[3]); //size UB
    QID = atoi(argv[4]); //Query vertex ID
    ui paritionSize = atoi(argv[5]);

    const char* filepath = argv[1];
    load_graph(filepath);

    Timer timer;
    StartTime = (double)clock() / CLOCKS_PER_SEC;


    ui jump = TOTAL_WARPS;

    core_decomposition_linear_list();

    // Description: upper bound defined
    ku = miv(core[QID], N2-1);
    kl = 0;
    ubD = N2-1;
    cal_query_dist();

    deviceGraphPointers deviceGraph;
    memoryAllocationGraph(deviceGraph);

    ui BLK_DIM2 = 128;
    ui BLK_NUM2 = 1;
    ui INTOTAL_WARPS=(BLK_NUM2*BLK_DIM2)/32;
    ui intialParitionSize = (n/INTOTAL_WARPS)+1;
    cout<<"here "<<INTOTAL_WARPS<<intialParitionSize<<endl;

    deviceInterPointers intialTask;
    memoryAllocationIntialTask(intialTask,INTOTAL_WARPS,intialParitionSize);

    size_t sharedMemrySizeIntial =  32* sizeof(ui);

    IntialReductionRules<<<BLK_NUM2,BLK_DIM2,sharedMemrySizeIntial>>>(deviceGraph,intialTask,n ,N2,kl,intialParitionSize);
    cudaDeviceSynchronize();

    ui globalCounter;
    cudaMemcpy(&globalCounter,intialTask.globalCounter,sizeof(ui),cudaMemcpyDeviceToHost);
    cout<<" Total "<<globalCounter<<endl;

    ui *taskOffset;

    taskOffset = new ui[paritionSize];
    memset(taskOffset, 0, paritionSize * sizeof(ui));
    taskOffset[1]= globalCounter;
    taskOffset[paritionSize-1] = 1;

    deviceTaskPointers deviceTask;
    memoryAllocationTask(deviceTask,TOTAL_WARPS,paritionSize);

    chkerr(cudaMemcpy(deviceTask.taskOffset,taskOffset,paritionSize*sizeof(ui),cudaMemcpyHostToDevice));

    CompressTask<<<BLK_NUM2,BLK_DIM2>>>(deviceGraph,intialTask,deviceTask,intialParitionSize,QID);
    cudaDeviceSynchronize();


    size_t sharedMemrySizeTask = 3*WARPS_EACH_BLK * sizeof(ui) + WARPS_EACH_BLK * sizeof(int) + WARPS_EACH_BLK * sizeof(double);
    size_t sharedMemrySizeExpand = WARPS_EACH_BLK * sizeof(ui);
    bool stopFlag;
    int c=0;

    ui *H;
    cudaMalloc((void**)&H,(globalCounter+1)*sizeof(ui));
    cudaMemset(H,0,sizeof(ui));
    cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS*paritionSize*sizeof(int));

    while(1){

        cudaMemset(deviceTask.flag,1,sizeof(bool));

        cudaMemcpy(&stopFlag,deviceTask.flag,sizeof(bool),cudaMemcpyDeviceToHost);

        ProcessTask <<<BLK_NUMS,BLK_DIM,sharedMemrySizeTask>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,H,c);
        cudaDeviceSynchronize();
       //jump = (ui) dis(gen);
        jump = jump>> 1;
        //cout<<"jump "<<jump<<endl;
        Expand <<<BLK_NUMS,BLK_DIM,sharedMemrySizeExpand>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,jump);

        cudaDeviceSynchronize();
        reduce <<<BLK_NUMS,BLK_DIM,sharedMemrySizeTask>>>(deviceGraph,deviceTask,paritionSize,N2);
        cudaDeviceSynchronize();
        cudaMemcpy(&stopFlag,deviceTask.flag,sizeof(bool),cudaMemcpyDeviceToHost);
        //cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
        cout <<"level "<<c<<"min degree "<<kl<<endl;
        if(stopFlag){
          cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
          cout << "Max min degree "<<kl<<endl;
          cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;

          break;
        }
        if(jump==1){
          jump = TOTAL_WARPS;
        }
        c++;
    }

    /*ui *hH;
    hH = new ui[globalCounter+1];
    cudaMemcpy(hH,H,(globalCounter+1)*sizeof(ui),cudaMemcpyDeviceToHost);
    cout << "H ";
    for(ui i =0;i<hH[globalCounter];i++){
      cout<<hH[i]<<" ";
    }
    cout <<endl;
    cout<<endl;*/
    /*ui *task, *status, *size, *off,*dc,*dr;
    int *ustar;
    task = new ui[TOTAL_WARPS*paritionSize];
    status = new ui[TOTAL_WARPS*paritionSize];
    size = new ui[TOTAL_WARPS*paritionSize];
    ustar = new int[TOTAL_WARPS*paritionSize];

    off = new ui[TOTAL_WARPS*paritionSize];
    dc = new ui[TOTAL_WARPS*paritionSize];
    dr = new ui[TOTAL_WARPS*paritionSize];


    cudaMemcpy(task,deviceTask.taskList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(status,deviceTask.statusList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(off,deviceTask.taskOffset,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(size,deviceTask.size,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(ustar,deviceTask.ustar,TOTAL_WARPS*paritionSize*sizeof(int),cudaMemcpyDeviceToHost);

    cudaMemcpy(dc,deviceTask.degreeInC,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(dr,deviceTask.degreeInR,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);


    for(ui i =0;i<TOTAL_WARPS;i++){
      if(off[(i+1)*paritionSize-1]==0){
        cout<<" partion "<<i<<" Num tasks "<<off[(i+1)*paritionSize-1]<<endl;

        }
      for(ui j =0;j<off[(i+1)*paritionSize-1] ;j++){
        if(off[i*paritionSize+j+1]>=100000){
        cout <<"start "<<off[i*paritionSize+j]<<" end "<<off[i*paritionSize+j+1] <<" size "<<size[i*paritionSize+j]<<" ustar "<<ustar[i*paritionSize+j]<<endl;
       }for(ui k = off[i*paritionSize+j]; k <off[i*paritionSize+j+1];k++){
          cout<< task[i*paritionSize+k] << " ";

        }
        cout <<endl;
        for(ui k = off[i*paritionSize+j]; k <off[i*paritionSize+j+1];k++){
          cout<< status[i*paritionSize+k] << " ";

        }
        cout <<endl;
        for(ui k = off[i*paritionSize+j]; k <off[i*paritionSize+j+1];k++){
          cout<< dc[i*paritionSize+k] << " ";

        }
        cout <<endl;
        for(ui k = off[i*paritionSize+j]; k <off[i*paritionSize+j+1];k++){
          cout<< dr[i*paritionSize+k] << " ";

        }
        //cout <<endl;

      }
    }*/

    freeInterPointer(intialTask);
    freeGraph(deviceGraph);
    freeTaskPointer(deviceTask);


    return 0;

}