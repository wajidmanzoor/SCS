#include <stdio.h>
#include "./inc/heuristic.h"
#include "./src/helpers.cc"
#include "./src/gpuMemoryAllocation.cu"



#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}




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
    size_t sharedMemrySizeDoms = WARPS_EACH_BLK * sizeof(ui);


    bool stopFlag;
    cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS*paritionSize*sizeof(int));
    ui *off,*task,*status,*hsize;
    ui *doms;
    int *ustar;
    double *cons;


    off = new ui[TOTAL_WARPS*paritionSize];
    doms = new ui[TOTAL_WARPS*paritionSize];
    task = new ui[TOTAL_WARPS*paritionSize];
    status = new ui[TOTAL_WARPS*paritionSize];
    hsize = new ui[TOTAL_WARPS*paritionSize];
    ustar = new int[TOTAL_WARPS*paritionSize];
    cons = new double[TOTAL_WARPS*paritionSize];
    ui pars[6] = {0,1727,2590,3455,5182,6045};

    int c =0;
    while(1){

        cudaMemset(deviceTask.flag,1,sizeof(bool));

        ProcessTask <<<BLK_NUMS,BLK_DIM,sharedMemrySizeTask>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX);
        cudaDeviceSynchronize();
        jump = jump>>1;
        //cout<<"Total Waraps "<<TOTAL_WARPS<<" jump "<<jump<<endl;
        FindDoms<<<BLK_NUMS, BLK_DIM,sharedMemrySizeDoms>>>(deviceGraph, deviceTask,paritionSize,dMAX);
        cudaDeviceSynchronize();
        ExpandNew <<<BLK_NUMS,BLK_DIM,sharedMemrySizeExpand>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,jump);
        cudaDeviceSynchronize();

        ExpandDoms<<<BLK_NUMS,BLK_DIM>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,jump);
        cudaDeviceSynchronize();

        ReductionRules << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize, N2);
        cudaDeviceSynchronize();
        DegreeUpdate << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize);
        cudaDeviceSynchronize();
        cout<< "***********************************Level********************************** "<<c<<" KL  "<<kl<<endl;

        cudaMemcpy(off,deviceTask.taskOffset,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(doms,deviceTask.doms,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(task,deviceTask.taskList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(status,deviceTask.statusList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(hsize,deviceTask.size,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(ustar,deviceTask.ustar,TOTAL_WARPS*paritionSize*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cons,deviceTask.cons,TOTAL_WARPS*paritionSize*sizeof(double),cudaMemcpyDeviceToHost);





        ui start,total,x;
        for(ui i =0;i<TOTAL_WARPS;i++){
          x = i+1;
          //cout<<x<<" ";
          if(off[x*paritionSize-1]!=0){
          cout<<" :- Partition "<<i<<" Task Num " <<off[x*paritionSize-1]<<" Available End index "<<paritionSize-1 <<" Offset of Last Task " <<i*off[x*paritionSize-1]+off[i*paritionSize+off[x*paritionSize-1]];
          /*cout<<"Offs ";
          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<off[j]<<" ";
          }
          cout<<endl;
          cout<<"Task ";
          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<task[j]<<" ";
          }
          cout<<endl;
          cout<<"Stat ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<status[j]<<" ";
          }
          cout<<endl;
          cout<<"Usta ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<ustar[j]<<" ";
          }
          cout<<endl;

          cout<<"Size ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<hsize[j]<<" ";
          }
          cout<<endl;
          cout<<"Doms ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<doms[j]<<" ";
          }
          cout<<endl;
          cout<<"Cons ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            cout<<cons[j]<<" ";
          }
          cout<<endl;*/
        }
        }
        cout<<endl;
        cudaMemset(deviceTask.doms,0,TOTAL_WARPS*paritionSize*sizeof(ui));


        cudaMemcpy(&stopFlag,deviceTask.flag,sizeof(bool),cudaMemcpyDeviceToHost);
        if(stopFlag){
          cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
          cout << "Max min degree "<<kl<<endl;
          cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;

          break;
        }
        cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);




        if(jump==1){
          jump = TOTAL_WARPS;
        }
        c++;
        //cout<< "Level "<<c<<endl;
        if(c==13)
        break;


    }




    cudaDeviceSynchronize();

    //freeInterPointer(intialTask);
    //freeGraph(deviceGraph);
    //freeTaskPointer(deviceTask);


    return 0;

}