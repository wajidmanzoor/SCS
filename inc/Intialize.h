#pragma once
#include "Utility.h"
#include "Timer.h"
#include <limits.h>
#include <cuda_runtime.h>
#include <mutex>
#include <map>
#include <iomanip>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <thread>
#include "../ipc/msgtool.h"

#include <sys/stat.h>


#define BLK_NUMS 432
#define BLK_DIM 512
#define TOTAL_THREAD (BLK_NUMS*BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM/32)
#define TOTAL_WARPS (BLK_NUMS*WARPS_EACH_BLK)

using namespace std;

int BLK_DIM2 ;
int BLK_NUM2;
int INTOTAL_WARPS;

Timer totalTimer;


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

ui *neighboroffset, *neighborList;

string graphPath;
string fileName;

ui initialPartitionSize;
ui outMemFlag;
ui *queryStopFlag;

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
ui numQueriesProcessing;
ui maxN2;

ui red1;
ui red2;
ui red3;
ui prun1;
ui prun2;


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

