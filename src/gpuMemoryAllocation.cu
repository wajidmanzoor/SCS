#include "../inc/ListLinearHeap.h"

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}


void memoryAllocationGraph(deviceGraphPointers &G){
    chkerr(cudaMalloc((void**)&(G.core), n * sizeof(ui)));
    chkerr(cudaMemcpy(G.core, core, n * sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&(G.degree), n * sizeof(ui)));
    chkerr(cudaMemcpy(G.degree, degree, n * sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&(G.offset), (n+1) * sizeof(ui)));
    chkerr(cudaMemcpy(G.offset, pstart, (n+1) * sizeof(ui), cudaMemcpyHostToDevice));


    chkerr(cudaMalloc((void**)&(G.neighbors), (2*m) * sizeof(ui)));
    chkerr(cudaMemcpy(G.neighbors, edges, (2*m) * sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&(G.distance), n * sizeof(ui)));
    chkerr(cudaMemcpy(G.distance, q_dist, n * sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&(G.lowerBoundDegree), sizeof(ui)));
    chkerr(cudaMemcpy(G.lowerBoundDegree, &kl,sizeof(ui),cudaMemcpyHostToDevice));
}

void memoryAllocationinitialTask(deviceInterPointers &p, ui numWraps,ui psize){
    chkerr(cudaMalloc((void**)&(p.initialTaskList), numWraps*psize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.globalCounter), sizeof(ui)));
    chkerr(cudaMemset(p.globalCounter,0,sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.entries),numWraps* sizeof(ui)));
}

void memoryAllocationTask(deviceTaskPointers &p, ui numWraps, ui pSize){
    chkerr(cudaMalloc((void**)&(p.taskList), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&p.statusList, numWraps*pSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.degreeInR), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.degreeInC), numWraps*pSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.taskOffset), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMemset(p.taskOffset,0, numWraps*pSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.ustar), numWraps*pSize*sizeof(int)));

    chkerr(cudaMalloc((void**)&(p.size), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMemset(p.size,0,numWraps*pSize*sizeof(ui)));

     chkerr(cudaMalloc((void**)&(p.doms), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.cons), numWraps*pSize*sizeof(double)));


    chkerr(cudaMalloc((void**)&(p.flag),sizeof(bool)));


}

void memoryAllocationBuffer(deviceBufferPointers &p,ui numWraps, ui pSize){

    chkerr(cudaMalloc((void**)&(p.taskOffset), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMemset(p.taskOffset,0, numWraps*pSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.taskList), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&p.statusList, numWraps*pSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.degreeInR), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.degreeInC), numWraps*pSize*sizeof(ui)));



    chkerr(cudaMalloc((void**)&(p.size), numWraps*pSize*sizeof(ui)));
    chkerr(cudaMemset(p.size,0,numWraps*pSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.numTask), sizeof(ui)));
    chkerr(cudaMemset(p.numTask,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.temp), sizeof(ui)));
    chkerr(cudaMemset(p.temp,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.numReadTasks), sizeof(ui)));
    chkerr(cudaMemset(p.numReadTasks,0,sizeof(ui)));


}

void freeGraph(deviceGraphPointers &p){
    chkerr(cudaFree(p.core));
    chkerr(cudaFree(p.degree));
    chkerr(cudaFree(p.distance));
    chkerr(cudaFree(p.lowerBoundDegree));
    chkerr(cudaFree(p.neighbors));
    chkerr(cudaFree(p.offset));

}


void freeInterPointer(deviceInterPointers &p){
    chkerr(cudaFree(p.entries));
    chkerr(cudaFree(p.globalCounter));
    chkerr(cudaFree(p.initialTaskList));

}

void freeTaskPointer(deviceTaskPointers &p){
    chkerr(cudaFree(p.size));
    chkerr(cudaFree(p.statusList));
    chkerr(cudaFree(p.taskList));
    chkerr(cudaFree(p.taskOffset));
    chkerr(cudaFree(p.ustar));
    //chkerr(cudaFree(p.flag));


}

