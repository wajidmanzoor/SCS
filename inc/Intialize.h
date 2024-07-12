#include "Utility.h"
#include "Timer.h"
#include <limits.h>
#include <cuda_runtime.h>

#define BLK_NUMS 128
#define BLK_DIM 1024
#define TOTAL_THREAD (BLK_NUMS*BLK_DIM)
#define WARPSIZE 32
#define WARPS_EACH_BLK (BLK_DIM/32)
#define TOTAL_WARPS (BLK_NUMS*WARPS_EACH_BLK)


ofstream fout;

int MaxTime = 1800;

ui domBr;
ui binBr;

double StartTime = 0.0;

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
ui N1; //size LB
ui N2; //size UB
int QID;
ui dMAX;
ui kl; //min deg
ui ku; //max min deg
vector<ui> H;
ui ubD = INF;

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
    ui *degree;
    ui *distance;
    ui *core;
    ui *lowerBoundDegree;

    
}deviceGraphPointers;

typedef struct  {
     ui *taskList;
     ui *statusList;
     ui *taskOffset; 
     ui *size;
     ui *degreeInR;
     ui *degreeInC;
     int *ustar;
     ui *flag;
     ui *doms;
     double *cons;


}deviceTaskPointers;

typedef struct  {
     ui *intialTaskList;
     ui *globalCounter;
     ui *entries;

}deviceInterPointers;