#include "SBS.h"



// Description: SCS Algotithm 
void CSSC_BB()
{
    Timer timer;
    StartTime = (double)clock() / CLOCKS_PER_SEC;
    
    // Description: Peeling algorith to calculate the core values of each vertex.
    core_decomposition_linear_list();

    // Description: upper bound defined
    ku = miv(core[QID], N2-1);


    Timer t_for_heu;

    // Description: Heurestic Algorithm to find starting H
    CSSC_heu();


    total_Heu_time += t_for_heu.elapsed();
    

    // Description: If Klower of H is equal to Kupper. return sol and break 
    if(kl==ku){
        cout<<"heuristic find the OPT!"<<endl;
        cout<<"mindeg = "<<kl<<endl;
        cout<<"H.size = "<<H.size()<<endl;
        cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;
        return;
    }
    
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
    // Description: calculate distance of all vertices from Query vertex in graph
    cal_query_dist();

    // Description: add all vertices with core > klower and distance < ubD (calculated above ) in G0
    // Description: intial reduction of R 
    reduction_g();
        
    
    // Description: this is C (refer to paper)
    VI.clear();
    
    // Description: this is R (refer to paper)
    VIVR.clear();
    
    // Description: bool to indicate if exists in C (VI)
    inVI = new bool[n];
    memset(inVI, 0, sizeof(bool)*n);

    // Description: bool to indicate if exists in R (VR)
    inVR = new bool[n];
    memset(inVR, 0, sizeof(bool)*n);

   // Description: Degree in C (VI)  
    degVI = new ui[n];
    memset(degVI, 0, sizeof(ui)*n);
    
    // Description: Degree in R (VR)
    degVIVR = new ui[n];
    memset(degVIVR, 0, sizeof(ui)*n);

    // Description: Neighbors of all vertices of C (VI) that are in R (VR). 
    inNEI = new ui[n];
    
    memset(inNEI, 0, sizeof(ui)*n);

    // Description: need to add
    NEI_score = new double[n];
    memset(NEI_score, 0, sizeof(double)*n);
    
    // Description: push all elements from intial R (G0) to R (VR).
    for(auto e : G0){
        VIVR.push_back(e);
        inVR[e] = 1;
        degVIVR[e] = G0_deg[e];
    }


    // Description: Push query vertix to C (VI)
    // Different: Started from QID and not the output of Huristic algo
    VI.push_back(QID);
    inVI[QID] = 1;
    inVR[QID] = 0;
    
    over_time_flag = false;

    if(EXE_dom_ustar)
        // Description: Algorithm with branching technique
        BB_dom_ustar(1);
    else
        // Description: Algorithm without branching technique
        BB(1);
    
    // Description: Stop in time exceeds, and return the best solution found so far.
    if(over_time_flag){
        cout<<"overtime"<<endl;
        cout<<"mindeg' = "<<kl<<endl;
        cout<<"H'.size = "<<H.size()<<endl;
        cout<<"time : "<<integer_to_string(timer.elapsed()).c_str()<<endl;
    }
    // Description: Optimal Solution Result 
    else{
        cout<<"mindeg = "<<kl<<endl;
        cout<<"H.size = "<<H.size()<<endl;
        cout<<"time = "<<integer_to_string(timer.elapsed()).c_str()<<endl;
        
        std::cout << "Result ";
        for (size_t i = 0; i < H.size(); ++i) {
            std::cout << H[i] << ", ";
        }
        std::cout << endl;
    }
}




int main(int argc, const char * argv[]) {

    if(argc!=20){
        cout<<"wrong input parameters!"<<endl;exit(1);
    }
    //./SCS ./graph.txt 6 9 2 1 1 1 1 1 1 0 1 1 1 0 0 1800 1
    N1 = atoi(argv[2]); //size LB
    N2 = atoi(argv[3]); //size UB
    QID = atoi(argv[4]); //Query vertex ID
    
    EXE_heu2 = atoi(argv[5]); //Heuristic strategy 1
    EXE_heu3 = atoi(argv[6]); //Heuristic strategy 2
    EXE_heu4 = atoi(argv[7]); //Heuristic strategy 3

    EXE_ub1 = atoi(argv[8]); //UB1
    EXE_ub2 = atoi(argv[9]); //UB2
    EXE_ub3 = atoi(argv[10]); //UB3
    EXE_ub3_optimization = atoi(argv[11]); //UB3 optimization
    
    EXE_core_maintenance = atoi(argv[12]); 
    EXE_new2VI = atoi(argv[13]); 
    EXE_del_from_VR = atoi(argv[14]); 
    EXE_dom_ustar = atoi(argv[15]); //Dominating based branching rule
    domS_Threshold = atoi(argv[16]); //Dom pair threshold
    MaxTime = atoi(argv[17]); //OT
    srch_ord = atoi(argv[18]); //Branching order

    // Added by Wajid if 1 prints the tree structure
    verbose = atoi(argv[19]);

    cout<<"Graph : "<<argv[1]<<", N1 = "<<N1<<", N2 = "<<N2<<", QID = "<<QID<<endl;
    cout<<"    Heu: "<<EXE_heu2<<","<<EXE_heu3<<","<<EXE_heu4;
    cout<<"    UBs: "<<EXE_ub1<<","<<EXE_ub2<<","<<EXE_ub3<<","<<EXE_ub3_optimization;
    cout<<"    Rdt: "<<EXE_core_maintenance<<","<<EXE_new2VI<<","<<EXE_del_from_VR;
    cout<<"    Dom: "<<EXE_dom_ustar<<","<<domS_Threshold;
    cout<<"    Tim: "<<MaxTime;
    cout<<"    Ord: "<<srch_ord;
    cout<<endl;
    // Description: Read graph
    load_graph(argv[1]);
    
    // Description: Algorithm 
    CSSC_BB();
    
    delete [] peel_sequence;
    delete [] degree;
    delete [] core;
    delete [] pstart;
    delete [] edges;
    delete [] q_dist;

    delete [] G0_edges;
    delete [] G0_x;
    delete [] G0_deg;
    
    delete [] inVI;
    delete [] inVR;
    delete [] degVI;
    
    delete [] inNEI;
    delete [] NEI_score;
    
    return 0;
}

