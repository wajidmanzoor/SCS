#include <stdio.h>
#include "./inc/heuristic.h"
#include "./src/helpers.cc"
#include "./src/gpuMemoryAllocation.cu"
#include <sys/stat.h>


#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}





int main(int argc, const char * argv[] ) {

    if(argc!=8){
        cout<<"wrong input parameters!"<<endl;exit(1);
        exit(1);

    }
    // ./SCS ./graph.txt 6 9 2 100000
    N1 = atoi(argv[2]); //size LB
    N2 = atoi(argv[3]); //size UB
    QID = atoi(argv[4]); //Query vertex ID
    ui paritionSize = atoi(argv[5]);
    int isHeu = atoi(argv[6]);
    ui limitDoms = atoi(argv[7]);

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
    if(isHeu){
    CSSC_heu();

    }
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


    ui stopFlag;
    /*cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS*paritionSize*sizeof(int));
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
    */



    int c =0;
    while(1){

        cudaMemset(deviceTask.flag,1,sizeof(ui));

        ProcessTask <<<BLK_NUMS,BLK_DIM,sharedMemrySizeTask>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,c);
        cudaDeviceSynchronize();
        jump = jump>>1;
        if(jump==1){
          jump = TOTAL_WARPS/2;
        }





        FindDoms<<<BLK_NUMS, BLK_DIM,sharedMemrySizeDoms>>>(deviceGraph, deviceTask,paritionSize,dMAX,c,limitDoms);
        cudaDeviceSynchronize();

        ExpandNew<<<BLK_NUMS,BLK_DIM,sharedMemrySizeExpand>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,jump,c);
        cudaDeviceSynchronize();
        //if(c<10)
        //{
        ExpandDoms<<<BLK_NUMS,BLK_DIM>>>(deviceGraph,deviceTask, N1, N2, paritionSize, dMAX,jump);
        cudaDeviceSynchronize();
        //}
         DegreeUpdate << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize);
        cudaDeviceSynchronize();
        ReductionRules << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize, N2);
        cudaDeviceSynchronize();

        DegreeUpdate << < BLK_NUMS, BLK_DIM>>> (deviceGraph, deviceTask, paritionSize);
        cudaDeviceSynchronize();

        cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);

        cout<< "***********************************Level********************************** "<<c<<" KL  "<<kl<<" jump "<<jump<<endl;



        cudaMemcpy(&stopFlag,deviceTask.flag,sizeof(ui),cudaMemcpyDeviceToHost);
        if(stopFlag){
          cudaMemcpy(&kl,deviceGraph.lowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
          cout << "Max min degree "<<kl<<endl;
          cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;

          break;
        }

        /*cudaMemcpy(off,deviceTask.taskOffset,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(doms,deviceTask.doms,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(task,deviceTask.taskList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(status,deviceTask.statusList,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(hsize,deviceTask.size,TOTAL_WARPS*paritionSize*sizeof(ui),cudaMemcpyDeviceToHost);
        cudaMemcpy(ustar,deviceTask.ustar,TOTAL_WARPS*paritionSize*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cons,deviceTask.cons,TOTAL_WARPS*paritionSize*sizeof(double),cudaMemcpyDeviceToHost);

        if((c>9)){

        string filename = "changesV9.txt";
        ofstream outFile;

        if (fileExists(filename)) {
          outFile.open(filename, ios_base::app);
        } else {
            outFile.open(filename);
        }
        if (outFile.is_open()){
          outFile<<"Level "<<c<<" K lower "<<kl<<" jump "<<jump<<endl;
        ui x;
        for(ui i =0;i<TOTAL_WARPS;i++){
          x = i+1;
          //cout<<x<<" ";
          if((off[x*paritionSize-1]!=0)||(i==5085)){
          outFile<<"Partition "<<i<<" Task Num" <<off[x*paritionSize-1]<<endl;
          outFile<<"Offs ";
          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<off[j]<<" ";
          }
          outFile<<endl;
          outFile<<"Task ";
          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<task[j]<<" ";
          }
          outFile<<endl;
          outFile<<"Stat ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<status[j]<<" ";
          }
          outFile<<endl;
          outFile<<"Usta ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<ustar[j]<<" ";
          }
          outFile<<endl;

          outFile<<"Size ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<hsize[j]<<" ";
          }
          outFile<<endl;
          outFile<<"Doms ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<doms[j]<<" ";
          }
          outFile<<endl;
          outFile<<"Cons ";

          for(ui j =i*paritionSize; j<x*paritionSize;j++){
            outFile<<cons[j]<<" ";
          }
          outFile<<endl;
        }
        }}
        }*/
        cudaMemset(deviceTask.doms,0,TOTAL_WARPS*paritionSize*sizeof(ui));






        if(jump==1){
          jump = TOTAL_WARPS;
        }
        c++;
        if(c==50)
          break;


    }




    cudaDeviceSynchronize();

    //freeInterPointer(intialTask);
    //freeGraph(deviceGraph);
    //freeTaskPointer(deviceTask);


    return 0;

}