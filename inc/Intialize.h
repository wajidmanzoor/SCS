#include "Utility.h"
#include "Timer.h"
#include <limits.h>
#include <cuda_runtime.h>

#define BLK_NUMS 64
#define BLK_DIM 1024
#define TOTAL_THREAD (BLK_NUMS*BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM/32)
#define TOTAL_WARPS (BLK_NUMS*WARPS_EACH_BLK)


ofstream fout;

int MaxTime = 1800;

ui domBr;
ui binBr;



bool EXE_heu2;
bool EXE_heu3;
bool EXE_heu4;

bool EXE_ub1;
bool EXE_ub2;
bool EXE_ub3;
bool EXE_ub3_optimization;
bool EXE_refine_G0;
bool EXE_core_maintenance;
bool EXE_new2VI;
bool EXE_del_from_VR;
bool EXE_dom_ustar;
double total_val_ub1;
double total_val_ub3;
double total_UB;
ui domS_Threshold;
ui srch_ord;
double total_Heu_time;
bool over_time_flag;

ui n; //vertices
ui m; //edges

ui totalQuerry;
vector<ui> H;


ui * pstart; //neighbors offset
ui * edges; //neighbors
ui * peel_sequence;
ui * degree;
ui * core;
ui * q_dist;

vector<ui> G0;
ui * G0_edges;
ui * G0_x;
ui * G0_deg;

vector<ui> VI;
vector<ui> VIVR;
bool * inVI;
bool * inVR;
ui * degVI;
ui * degVIVR;

vector<ui> NEI;
ui * inNEI;
double * NEI_score;
vector<vector<ui>> combs;

vector<queryInfo> message_queue;
mutex message_queue_mutex;

vector<queryData> queries;

ui verbose;

bool cmp_of_domS(const ui x, const ui y)
{
    return degVIVR[x]>degVIVR[y];
}

double time_new2VI;
double time_del_from_VR;
double time_find_NEI;
double time_find_usatr;
double time_comp_ub;

typedef struct  {
    ui *offset;
    ui *neighbors;
    ui *core;
    ui *degree;

}deviceGraphGenPointers;

typedef struct {
    ui *degree;
    ui *distance;
    ui *lowerBoundDegree;
    ui *newNeighbors;
    ui *newOffset;
}deviceGraphPointer;

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
     bool *flag;
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
ui *queryIndicator



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
     double StartTime = 0.0;
     Timer timer;
     ui isHeu;
     ui limitDoms;
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


     }


}