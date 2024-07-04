#include <stdio.h>
#include "./inc/heuristic.h"
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
    CSSC_heu();
    // Description: If Klower of H is equal to Kupper. return sol and break
    if(kl==ku){
        cout<<"heuristic find the OPT!"<<endl;
        cout<<"mindeg = "<<kl<<endl;
        cout<<"H.size = "<<H.size()<<endl;
        cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;
        return;
    }
    cout<< " Kl "<< kl << " Ku "<<ku<<endl;

    // Description: Calculate Diameter using klower and N2.
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



    cal_query_dist();

    deviceGraphPointers deviceGraph;
    memoryAllocationGraph(deviceGraph);

    ui BLK_DIM2 = 1024;
    ui BLK_NUM2 = 2;
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
    cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS*paritionSize*sizeof(int));

    //ui *off, *hsize, *task_h, *status_h, degc_h, degR_h;
    //int *ustar;
    while(1){

        cudaMemset(deviceTask.flag,1,sizeof(bool));

        ProcessTask <<<BLK_NUMS,BLK_DIM,sharedMemrySizeTask>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX);
        cudaDeviceSynchronize();
        jump = jump>>1;
        ReductionRules << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize, N2);
        cudaDeviceSynchronize();
        Expand <<<BLK_NUMS,BLK_DIM,sharedMemrySizeExpand>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,jump);
        cudaDeviceSynchronize();
        DegreeUpdate << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize);
        cudaDeviceSynchronize();
        cudaMemcpy(&stopFlag,deviceTask.flag,sizeof(bool),cudaMemcpyDeviceToHost);
        if(stopFlag){
          cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
          cout << "Max min degree "<<kl<<endl;
          cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;

          break;
        }
        //cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);

        //cout<<"KL  "<<kl<<endl;


        if(jump==1){
          jump = TOTAL_WARPS;
        }

    }

    	  /*off = new ui[TOTAL_WARPS*paritionSize];
        ustar = new int[TOTAL_WARPS*paritionSize];
        hsize = new ui[TOTAL_WARPS*paritionSize];

        task_h = new ui[TOTAL_WARPS*paritionSize];
        status_h = new ui[TOTAL_WARPS*paritionSize];
        degc_h = new ui[TOTAL_WARPS*paritionSize];
        degR_h = new ui[TOTAL_WARPS*paritionSize];





        cudaMemcpy(off,deviceTask.taskOffset,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
      cudaMemcpy(ustar,deviceTask.ustar,TOTAL_WARPS*paritionSize*sizeof(int),cudaMemcpyDeviceToHost);
      cudaMemcpy(hsize,deviceTask.size,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);

      cudaMemcpy(task_h,deviceTask.taskList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
      cudaMemcpy(status_h,deviceTask.statusList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
      cudaMemcpy(degR_h,deviceTask.degreeInR,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
      cudaMemcpy(degc_h,deviceTask.degreeInC,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);

        for(ui i =0;i<TOTAL_WARPS;i++){
            
           
            if (off[(i+1)*paritionSize-1]>0){
              for(ui j = 0; j<off[(i+1)*paritionSize-1];j++){
                for(ui k = off[i*paritionSize+j];k<off[i*paritionSize+j+1];k++){
                  cout<<task_h[i*paritionSize+j+k]<<" ";
                }
                cout<<endl;
                for(ui k = off[i*paritionSize+j];k<off[i*paritionSize+j+1];k++){
                  cout<<status_h[i*paritionSize+j+k]<<" ";
                }
                cout<<endl;

                for(ui k = off[i*paritionSize+j];k<off[i*paritionSize+j+1];k++){
                  cout<<degR_h[i*paritionSize+j+k]<<" ";
                }
                cout<<endl;

                for(ui k = off[i*paritionSize+j];k<off[i*paritionSize+j+1];k++){
                  cout<<degc_h[i*paritionSize+j+k]<<" ";
                }
                cout<<endl;
              cout<<endl;

              }
            }
            


        }*/



    cudaDeviceSynchronize();

    freeInterPointer(intialTask);
    freeGraph(deviceGraph);
    freeTaskPointer(deviceTask);


    return 0;

}