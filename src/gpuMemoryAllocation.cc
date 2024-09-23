#include "../inc/Intialize.h"

#define chkerr(code) { chkerr_impl((code), __FILE__, __LINE__); }

inline void chkerr_impl(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " | File: " << file
                  << " | Line: " << line << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
}
struct subtract_from {
    const ui x;
    subtract_from(ui _x) : x(_x) {}
    __host__ __device__ ui operator()(const ui & y) const {
        return x - y;
    }
};

void memoryAllocationGenGraph(deviceGraphGenPointers &G) {
    int ready_for_allocation = 1;
    MPI_Allreduce(MPI_IN_PLACE, &ready_for_allocation, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!ready_for_allocation) {
        cerr << "Error: Not all processes are ready for memory allocation GenGraph ." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    chkerr(cudaMalloc((void**)&(G.core), n * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.degree), n * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.offset), (n+1) * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.neighbors), (2*m) * sizeof(ui)));

    chkerr(cudaMemcpy(G.core, core, n * sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(G.degree, degree, n * sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(G.offset, pstart, (n+1) * sizeof(ui), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(G.neighbors, edges, (2*m) * sizeof(ui), cudaMemcpyHostToDevice));
}

void memoryAllocationGraph(deviceGraphPointers &G, ui totalQueries) {

    int ready_for_allocation = 1;
    MPI_Allreduce(MPI_IN_PLACE, &ready_for_allocation, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!ready_for_allocation) {
        cerr << "Error: Not all processes are ready for graph memory allocation." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    chkerr(cudaMalloc((void**)&(G.degree), totalQueries * n * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.distance), totalQueries * n * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.lowerBoundDegree), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.lowerBoundSize), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.upperBoundSize), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.limitDoms), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.flag), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.numRead), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.numWrite), totalQueries * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.newNeighbors), totalQueries * (2 *m) * sizeof(ui)));
    chkerr(cudaMalloc((void**)&(G.newOffset), totalQueries * (n + 1) * sizeof(ui)));

    chkerr(cudaMemset(G.lowerBoundDegree, 0, totalQueries * sizeof(ui)));
    chkerr(cudaMemset(G.lowerBoundSize, 0, totalQueries * sizeof(ui)));
    chkerr(cudaMemset(G.upperBoundSize, 0, totalQueries * sizeof(ui)));
    chkerr(cudaMemset(G.limitDoms, 0, totalQueries * sizeof(ui)));
    chkerr(cudaMemset(G.numRead, 0, totalQueries * sizeof(ui)));
    chkerr(cudaMemset(G.numWrite, 0, totalQueries * sizeof(ui)));
    chkerr(cudaMemset(G.newOffset, 0, totalQueries * (n + 1) * sizeof(ui)));

}


void memoryAllocationinitialTask(deviceInterPointers &p, ui numWraps,ui psize){
    chkerr(cudaMalloc((void**)&(p.initialTaskList), numWraps*psize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.globalCounter), sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.entries),numWraps* sizeof(ui)));
}

void memoryAllocationTask(deviceTaskPointers &p, ui numWraps, ui pSize, ui totalQueries, ui factor){
    ui taskSize = numWraps*pSize;
    ui offsetSize = (numWraps)* (pSize/factor);
    ui limitTasks = (pSize/factor) -1;
    ui otherSize = numWraps * limitTasks;
    chkerr(cudaMalloc((void**)&(p.limitTasks), sizeof(ui)));
    chkerr(cudaMemcpy(p.limitTasks, &limitTasks, sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&(p.numTasks), numWraps*sizeof(ui)));
    chkerr(cudaMemset(p.numTasks,0, numWraps*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.sortedIndex), numWraps*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.mapping), numWraps*sizeof(ui)));

    thrust::device_ptr<ui> d_sortedIndex_ptr(deviceTask.sortedIndex);
    thrust::device_ptr<ui> d_mapping_ptr(deviceTask.mapping);

    thrust::sequence(thrust::device, d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS);
    thrust::transform(thrust::device, d_sortedIndex_ptr, d_sortedIndex_ptr + TOTAL_WARPS, d_mapping_ptr, subtract_from(TOTAL_WARPS-1));

    chkerr(cudaMalloc((void**)&(p.taskOffset), (offsetSize)*sizeof(ui)));
    chkerr(cudaMemset(p.taskOffset,0, (offsetSize)*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.queryIndicator), otherSize*sizeof(ui)));
    thrust::device_ptr<ui> dev_ptr1(p.queryIndicator);
    thrust::fill(dev_ptr1, dev_ptr1 + otherSize, totalQueries);

    chkerr(cudaMalloc((void**)&(p.taskList), taskSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&p.statusList, taskSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.degreeInR), taskSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.degreeInC),taskSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.ustar), otherSize*sizeof(int)));
    thrust::device_ptr<int> dev_ptr(p.ustar);
    thrust::fill(dev_ptr, dev_ptr + otherSize , -1);

    chkerr(cudaMalloc((void**)&(p.size), otherSize*sizeof(ui)));
    chkerr(cudaMemset(p.size,0,otherSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.doms), taskSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&(p.cons), taskSize*sizeof(double)));



}

void memoryAllocationBuffer(deviceBufferPointers &p, ui bufferSize, ui totalQueries, ui factor){

    ui offsetSize = bufferSize/factor;
    ui limitTasks = bufferSize/factor -1;
    ui otherSize = limitTasks;

    chkerr(cudaMalloc((void**)&(p.limitTasks), sizeof(ui)));
    chkerr(cudaMemcpy(p.limitTasks, &otherSize, sizeof(ui), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&(p.numTask), sizeof(ui)));
    chkerr(cudaMemset(p.numTask,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.temp), sizeof(ui)));
    chkerr(cudaMemset(p.temp,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.numReadTasks), sizeof(ui)));
    chkerr(cudaMemset(p.numReadTasks,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.taskOffset), offsetSize*sizeof(ui)));
    chkerr(cudaMemset(p.taskOffset,0, offsetSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.queryIndicator), otherSize*sizeof(ui)));
    thrust::device_ptr<ui> dev_ptr(p.queryIndicator);
    thrust::fill(dev_ptr, dev_ptr + otherSize, totalQueries);

    chkerr(cudaMalloc((void**)&(p.taskList), bufferSize*sizeof(ui)));
    chkerr(cudaMalloc((void**)&p.statusList, bufferSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.size), otherSize*sizeof(ui)));
    chkerr(cudaMemset(p.size,0,otherSize*sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.writeMutex), sizeof(ui)));
    chkerr(cudaMemset(p.writeMutex,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.readMutex), sizeof(ui)));
    chkerr(cudaMemset(p.readMutex,0,sizeof(ui)));

    chkerr(cudaMalloc((void**)&(p.outOfMemoryFlag), sizeof(ui)));
    chkerr(cudaMemset(p.outOfMemoryFlag,0,sizeof(ui)));

}

void freeGenGraph(deviceGraphGenPointers &p){
    chkerr(cudaFree(p.offset));
    chkerr(cudaFree(p.neighbors));
    chkerr(cudaFree(p.degree));
    chkerr(cudaFree(p.core));


}

void freeGraph(deviceGraphPointers &p){

    chkerr(cudaFree(p.degree));
    chkerr(cudaFree(p.distance));
    chkerr(cudaFree(p.newNeighbors));
    chkerr(cudaFree(p.newOffset));
    chkerr(cudaFree(p.lowerBoundDegree));
    chkerr(cudaFree(p.lowerBoundSize));
    chkerr(cudaFree(p.upperBoundSize));
    chkerr(cudaFree(p.limitDoms));
    chkerr(cudaFree(p.flag));
    chkerr(cudaFree(p.numRead));
    chkerr(cudaFree(p.numWrite));


}


void freeInterPointer(deviceInterPointers &p){
    chkerr(cudaFree(p.initialTaskList));
    chkerr(cudaFree(p.globalCounter));
    chkerr(cudaFree(p.entries));

}

void freeTaskPointer(deviceTaskPointers &p){
    chkerr(cudaFree(p.taskList));
    chkerr(cudaFree(p.statusList));
    chkerr(cudaFree(p.taskOffset));
    chkerr(cudaFree(p.size));
    chkerr(cudaFree(p.degreeInR));
    chkerr(cudaFree(p.degreeInC));
    chkerr(cudaFree(p.ustar));
    chkerr(cudaFree(p.doms));
    chkerr(cudaFree(p.cons));
    chkerr(cudaFree(p.queryIndicator));


}

void freeBufferPointer(deviceBufferPointers &p){
    chkerr(cudaFree(p.taskOffset));
    chkerr(cudaFree(p.taskList));
    chkerr(cudaFree(p.statusList));
    chkerr(cudaFree(p.size));
    chkerr(cudaFree(p.numTask));
    chkerr(cudaFree(p.temp));
    chkerr(cudaFree(p.numReadTasks));
    chkerr(cudaFree(p.writeMutex));
    chkerr(cudaFree(p.readMutex));
    chkerr(cudaFree(p.queryIndicator));
    chkerr(cudaFree(p.outOfMemoryFlag));


}

