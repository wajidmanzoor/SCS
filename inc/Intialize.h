#include "Utility.h"
#include "Timer.h"
#include <limits.h>
#include <cuda_runtime.h>
#include <string>

#define BLK_NUMS 64
#define BLK_DIM 1024
#define TOTAL_THREAD (BLK_NUMS*BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM/32)
#define TOTAL_WARPS (BLK_NUMS*WARPS_EACH_BLK)

ui BLK_DIM2 = 1024;
ui BLK_NUM2 = 32;
ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;



using namespace std;

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
bool * outOfMemoryFlag;
ui outMemFlag;
bool *queryStopFlag;
ui jump;

ui partitionSize; 
ui bufferSize; 
double copyLimit; 
ui readLimit;
ui limitQueries;

ui tempHost;
ui numReadHost;
ui numTaskHost;
ui startOffset;
ui endOffset;

ui totalQuerry;
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
    bool *flag;
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
     

}deviceTaskPointers;

typedef struct {

    ui *taskOffset;
    ui *taskList;
    ui *statusList;
    ui *degreeInC;
    ui *degreeInR;
    ui  *size;
    ui *numTask;
    ui *temp;
    ui *numReadTasks;
    ui *writeMutex;
    ui *readMutex;
    ui *queryIndicator;

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
     int QID;
     ui kl; //min deg
     ui ku; //max min deg
     ui ubD;
     ui isHeu;
     ui limitDoms;
     ui *numRead;
     ui *numWrite;
     bool solFlag;
     Timer receiveTimer; // Timer to track time from received to processed

	queryData(){

     }
	queryData(ui N1, ui N2, int QID,ui isHeu,ui limitDoms){
          this->N1 = N1;
          this->N2 = N2;
          this->QID = QID;
          this->isHeu = isHeu;
          this->limitDoms = limitDoms;
          this->kl = 0;
          this->ku = miv(core[QID], N2 - 1);
          this->ubD = 0;
          this->numRead = 0;
          this->numWrite = 0;
          this->solFlag = false;


     }
     
     friend ostream& operator<<(ostream& os, queryData& qd) {
        os << "N1 = " << qd.N1 << ", "
           << "N2 = " << qd.N2 << ", "
           << "QID = " << qd.QID << ", "
           << "isHeu = " << qd.isHeu << ", "
           << "limitDoms = " << qd.limitDoms << ", "
           << "kl = " << qd.kl << ", "
           << "ku = " << qd.ku <<", "
           << "Elapsed Time = " << integer_to_string(qd.processTimer.elapsed()).c_str();
        return os;
    }


};

struct queryInfo
{
	int queryId;
	string queryString;


	queryInfo(){}

	queryInfo(int queryId, char* queryString)
	{
		this->queryId = queryId;
		this->queryString = queryString;
	}
    friend ostream& operator<<(ostream& os, queryInfo& q)
    {
        os << "Query ID: " << q.queryId << ", Query String: " << q.queryString;
        return os;
    }

};

vector<queryInfo> messageQueue;
mutex messageQueueMutex;

vector<queryData> queries;

deviceGraphGenPointers deviceGenGraph;
deviceGraphPointers deviceGraph;
deviceInterPointers initialTask;
deviceTaskPointers deviceTask;
deviceBufferPointers deviceBuffer;