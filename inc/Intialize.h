#pragma once
#include "Utility.h"
#include "Timer.h"
#include <limits.h>
#include <cuda_runtime.h>
#include <string>
#include <mutex>
#include <map>
#include <iomanip>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thread>
#include <thrust/device_vector.h>


#include "../ipc/msgtool.h"


#define BLK_NUMS 64
#define BLK_DIM 1024
#define TOTAL_THREAD (BLK_NUMS*BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM/32)
#define TOTAL_WARPS (BLK_NUMS*WARPS_EACH_BLK)


#define TAG_NQP 1
#define TAG_MTYPE 2
#define TAG_MSG 3
#define TAG_TERMINATE 4
#define TAG_R 5
#define TAG_RESULT 6


ui BLK_DIM2 = 1024;
ui BLK_NUM2 = 4;
ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;



using namespace std;


int worldRank, worldSize; 

vector<ui> H;

ui n; //vertices
ui m; //edges
ui  dMAX;

ui * pstart; //neighbors offset
ui * edges; //neighbors
ui * peel_sequence;
ui * degree;
ui * core;
ui * q_dist;

ui initialPartitionSize;
ui outMemFlag;
ui *queryStopFlag;
ui jump;

ui partitionSize;
ui bufferSize;
double copyLimit;
ui readLimit;
ui limitQueries;
ui factor;

ui tempHost;
ui numReadHost;
ui numTaskHost;
ui startOffset;
ui endOffset;

ui totalQuerry;
ui leastQuery;
ui numQueriesProcessing;
ui maxN2;


size_t sharedMemorySizeinitial;
size_t  sharedMemoryUpdateNeigh;
size_t sharedMemorySizeTask;
size_t sharedMemorySizeDoms;
size_t sharedMemorySizeExpand;


typedef struct  {

    ui *offset;
    ui *neighbors;
    ui *core;
    ui *degree;

}deviceGraphGenPointers;

typedef struct {

    ui *degree;
    ui *distance;
    ui *newNeighbors;
    ui *newOffset;
    ui *lowerBoundDegree;
    ui *lowerBoundSize;
    ui *upperBoundSize;
    ui *limitDoms;
    ui *flag;
    ui *numRead;
    ui *numWrite;

}deviceGraphPointers;

typedef struct  {

     ui *taskList;
     ui *statusList;
     ui *taskOffset;
     ui *size;
     ui *degreeInR;
     ui *degreeInC;
     int *ustar;
     ui *doms;
     double *cons;
     ui *queryIndicator;
     ui *numTasks;
     ui *limitTasks;
     ui *sortedIndex;
     ui *mapping;


}deviceTaskPointers;

typedef struct {

    ui *taskOffset;
    ui *taskList;
    ui *statusList;
    ui  *size;
    ui *numTask;
    ui *temp;
    ui *numReadTasks;
    ui *writeMutex;
    ui *readMutex;
    ui *queryIndicator;
    ui *outOfMemoryFlag;
    ui *limitTasks;

}deviceBufferPointers;

typedef struct  {

     ui *initialTaskList;
     ui *globalCounter;
     ui *entries;

}deviceInterPointers;

struct queryData
{
	 ui N1; //size LB
     ui N2; //size UB
     ui QID;
     ui kl; //min deg
     ui ku; //max min deg
     ui ubD;
     ui isHeu;
     ui limitDoms;
     ui numRead;
     ui numWrite;
     ui solFlag;
     ui querryId;
     ui ind;
     Timer receiveTimer; // Timer to track time from received to processed

	queryData(){
          this->N1 = 0;
          this->N2 = 0;
          this->QID = 0;
          this->isHeu = 0;
          this->limitDoms = 0;
          this->kl = 0;
          this->ku = 0;
          this->ubD = 0;
          this->solFlag = 2;
          this->numRead = 0;
          this->numWrite = 0;
          this->querryId = 0;
          ui ind = UINT_MAX;
          this->receiveTimer.restart();


     }
	void updateQueryData(ui N1, ui N2, ui QID,ui isHeu,ui limitDoms, ui querryId,ui ind){
          this->N1 = N1;
          this->N2 = N2;
          this->QID = QID;
          this->isHeu = isHeu;
          this->limitDoms = limitDoms;
          this->kl = 0;
          this->ku = miv(core[QID], N2 - 1);
          this->ubD = 0;
          this->solFlag = 0;
          this->numRead = 0;
          this->numWrite = 0;
          this->querryId = querryId;
          this->ind = ind;
          this->receiveTimer.restart();


     }

     friend ostream& operator<<(ostream& os, queryData& qd) {
        os << "Querry Id = "<< qd.querryId << ", "
           << "IND = "<<qd.ind<<", "
           << "N1 = " << qd.N1 << ", "
           << "N2 = " << qd.N2 << ", "
           << "QID = " << qd.QID << ", "
           << "isHeu = " << qd.isHeu << ", "
           << "limitDoms = " << qd.limitDoms << ", "
           << "kl = " << qd.kl << ", "
           << "ku = " << qd.ku <<", "
           << "Elapsed Time = " << integer_to_string(qd.receiveTimer.elapsed()).c_str();
        return os;
    }


};

struct queryInfo
{
	int queryId;
	string queryString;


	queryInfo(){}

	queryInfo(int queryId, string queryString)
	{
		this->queryId = queryId;


		this->queryString = queryString;
	}
    friend ostream& operator<<(ostream& os, const queryInfo& q)
    {
        os << "Query ID: " << q.queryId << ", Query String: " << q.queryString;
        return os;
    }

};
vector<queryInfo> messageQueue;
mutex messageQueueMutex;
queryData *queries;

deviceGraphGenPointers deviceGenGraph;
deviceGraphPointers deviceGraph;
deviceInterPointers initialTask;
deviceTaskPointers deviceTask;
deviceBufferPointers deviceBuffer;

void memoryAllocationGenGraph(deviceGraphGenPointers &G);
void memeoryAllocationGraph(deviceGraphPointers &G,ui totalQueries);
void memoryAllocationinitialTask(deviceInterPointers &p, ui numWraps, ui psize);
void memoryAllocationTask(deviceTaskPointers &p, ui numWraps, ui pSize, ui totalQueries, ui factor);
void memoryAllocationBuffer(deviceBufferPointers &p, ui bufferSize,ui totalQueries, ui factor);

void freeGenGraph(deviceGraphGenPointers &p);
void freeGraph(deviceGraphPointers &p);
void freeInterPointer(deviceInterPointers &p);
void freeTaskPointer(deviceTaskPointers &p);
void freeBufferPointer(deviceBufferPointers &p);

struct SystemInfo {
    int rank;
    int numQueriesProcessing;
    bool flag;
};

enum MessageType {
    PROCESS_MESSAGE = 1,
    TERMINATE = 2
};

enum SystemStatus {

    IDLE = 0, // Query was never sent to system
    PROCESSING = 1, // Recived atleast one query, either processing query or waiting for new queries.
    TERMINATED = 2 // Found solution for all sent queries and program terminated
};

