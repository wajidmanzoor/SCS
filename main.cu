#include <stdio.h>
#include "./src/Graph.h"
#include "./src/helpers.cc"


void cal_query_dist()
{

    // Description : Intialize querry distance array with INF
    q_dist = new ui[n];
    for(ui i =0;i<n;i++)
        q_dist[i] = INF;

    // Description: Queue that stores vertices
    queue<ui> Q;

    // Description : set distance of querry vertex as 0.
    q_dist[QID] = 0;

    // Description: Push querry vertex to Queue.
    Q.push(QID);

    // Description : Itterate till queue is empty
    while (!Q.empty()) {

        // Description : Get first vertex (v) from queue.
        ui v = Q.front();
        Q.pop();

        // Description: Iterate through the neighbors of V
        for(ui i = pstart[v]; i < pstart[v+1]; i++){
            ui w = edges[i];

            // Description : if distance of neighbor is INF, set to dstance of parent + 1.
            // Push neighbor to queue.
            if(q_dist[w] == INF){
                q_dist[w] = q_dist[v] + 1;
                Q.push(w);
            }
        }
    }
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

    ui intialParitionSize = (n/TOTAL_WARPS)+1;
    Timer timer;
    StartTime = (double)clock() / CLOCKS_PER_SEC;

    core_decomposition_linear_list();

    // Description: upper bound defined
    ku = miv(core[QID], N2-1);
    kl = 0;
    ubD = N2-1;
    cal_query_dist();

    ui *deviceOffset,*deviceNeighbors,*deviceDegree, *deviceDistance,*deviceCore;
    ui *deviceLowerBoundDegree;


    cudaMalloc((void**)&deviceCore, n * sizeof(ui));
    cudaMemcpy(deviceCore, core, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&deviceDegree, n * sizeof(ui));
    cudaMemcpy(deviceDegree, degree, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&deviceOffset, (n+1) * sizeof(ui));
    cudaMemcpy(deviceOffset, pstart, (n+1) * sizeof(ui), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&deviceNeighbors, (2*m) * sizeof(ui));
    cudaMemcpy(deviceNeighbors, edges, (2*m) * sizeof(ui), cudaMemcpyHostToDevice);

     cudaMalloc((void**)&deviceDistance, n * sizeof(ui));
    cudaMemcpy(deviceDistance, q_dist, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&deviceLowerBoundDegree, sizeof(ui));
    cudaMemcpy(deviceLowerBoundDegree, &kl,sizeof(ui),cudaMemcpyHostToDevice);

    ui *deviceIntialTaskList, *deviceIntialStatusList, *deviceGlobalCounter,*deviceEntries;
    ui BLK_DIM2 = 1024;
    ui INTOTAL_WARPS=BLK_DIM2/32;
    ui intialParitionSize = (n/INTOTAL_WARPS)+1;
    ui intialSize = intialParitionSize*INTOTAL_WARPS;
    cout<<"Psize "<<intialParitionSize<<" Size "<<intialSize<<endl;

    cudaMalloc((void**)&deviceIntialTaskList, intialSize*sizeof(ui));

    cudaMalloc((void**)&deviceIntialStatusList, intialSize*sizeof(ui));

    cudaMalloc((void**)&deviceGlobalCounter, sizeof(ui));
    cudaMalloc((void**)&deviceEntries,INTOTAL_WARPS* sizeof(ui));

    ui globalCounter = 0;
    cudaMemcpy(deviceGlobalCounter, &globalCounter, sizeof(ui), cudaMemcpyHostToDevice);

    size_t shared_memory_size =  INTOTAL_WARPS* sizeof(ui);

    IntialReductionRules<<<1,BLK_DIM2,shared_memory_size>>>(deviceOffset,deviceNeighbors,deviceDegree,deviceDistance,deviceCore,deviceIntialTaskList,deviceIntialStatusList,deviceEntries, deviceGlobalCounter,QID,n ,N2,kl,intialParitionSize);
    cudaDeviceSynchronize();
    cudaMemcpy(&globalCounter,deviceGlobalCounter,sizeof(ui),cudaMemcpyDeviceToHost);
    cout<<" Total "<<globalCounter<<endl;

    ui *reducedTaskList, *reducedStatusList;

    cudaMalloc((void**)&reducedTaskList, globalCounter*sizeof(ui));
    cudaMalloc((void**)&reducedStatusList, globalCounter*sizeof(ui));



    CompressTask<<<1,BLK_DIM2>>>(deviceIntialTaskList,deviceIntialStatusList,deviceEntries,reducedTaskList, reducedStatusList,intialParitionSize);
    cudaDeviceSynchronize();


    cudaFree(deviceIntialTaskList);
    cudaFree(deviceIntialStatusList);
    cudaFree(deviceEntries);
    cudaFree(deviceGlobalCounter);

    ui *taskOffset;

    taskOffset = new ui[paritionSize];
    memset(taskOffset, 0, paritionSize * sizeof(ui));
    taskOffset[1]= globalCounter;
    taskOffset[paritionSize-1] = 1;

    ui *deviceTaskList,*deviceStatusList, *deviceTaskOffset, *deviceSubgraphSize;
    int *deviceUstar;


    cudaMalloc((void**)&deviceTaskList, TOTAL_WARPS*paritionSize*sizeof(ui));
    cudaMalloc((void**)&deviceStatusList, TOTAL_WARPS*paritionSize*sizeof(ui));

    cudaMemcpy(deviceTaskList, reducedTaskList,globalCounter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(deviceStatusList, reducedStatusList,globalCounter*sizeof(ui),cudaMemcpyDeviceToDevice);

    cudaMalloc((void**)&deviceTaskOffset, TOTAL_WARPS*paritionSize*sizeof(ui));
    cudaMemcpy(deviceTaskOffset,taskOffset,paritionSize*sizeof(ui),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&deviceUstar, TOTAL_WARPS*paritionSize*sizeof(int));

    cudaMalloc((void**)&deviceSubgraphSize, TOTAL_WARPS*paritionSize*sizeof(ui));

    cudaFree(reducedTaskList);
    cudaFree(reducedStatusList);

    bool *deviceStopFlag;
    bool stopFlag;

    cudaMalloc((void**)&deviceStopFlag,sizeof(bool));

    shared_memory_size = 3*WARPS_EACH_BLK * sizeof(ui) + WARPS_EACH_BLK * sizeof(int) + WARPS_EACH_BLK * sizeof(double);
    size_t shared_memory_size1 = WARPS_EACH_BLK * sizeof(ui);



    int c= 0;
    while(1){

        cudaMemset(deviceStopFlag,1,sizeof(bool));
        cudaMemset(deviceUstar, -1, TOTAL_WARPS*paritionSize*sizeof(int));

        cudaMemcpy(&stopFlag,deviceStopFlag,sizeof(bool),cudaMemcpyDeviceToHost);

        SCSSpeedEff <<<BLK_NUMS,BLK_DIM,shared_memory_size>>>(deviceTaskList, deviceStatusList,deviceTaskOffset,deviceNeighbors, deviceOffset, deviceDegree, deviceDistance, deviceStopFlag, deviceLowerBoundDegree, N1, N2, paritionSize, dMAX,deviceUstar,deviceSubgraphSize);
        cudaDeviceSynchronize();

        Expand <<<BLK_NUMS,BLK_DIM,shared_memory_size1>>>(deviceTaskList, deviceStatusList,deviceTaskOffset,deviceUstar,deviceSubgraphSize,deviceNeighbors, deviceOffset, deviceDegree, deviceDistance, deviceStopFlag, deviceLowerBoundDegree, N1, N2, paritionSize, dMAX);

        cudaMemcpy(&stopFlag,deviceStopFlag,sizeof(bool),cudaMemcpyDeviceToHost);
        cudaMemcpy(&kl, deviceLowerBoundDegree,sizeof(ui),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
       // cout<< " Max Min degree  "<<kl<<endl;
        if(stopFlag){
          cout<< " Max Min degree  "<<kl<<endl;
          cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;
          cout<< "C "<<c<<endl;
          break;
        }

        if(c%3==0){
          cout<<"C= "<<c<<endl;
        }

        c++;
      }
    cudaFree(deviceOffset);
    cudaFree(deviceNeighbors);
    cudaFree(deviceDegree);
    cudaFree(deviceDistance);
    cudaFree(deviceCore);

    cudaFree(deviceLowerBoundDegree);
    cudaFree(deviceTaskList);
    cudaFree(deviceStatusList);
    cudaFree(deviceTaskOffset);
    cudaFree(deviceStopFlag);


    return 0;

}