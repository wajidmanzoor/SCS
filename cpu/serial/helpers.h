#include "Heuristic.h"

// Description : calculate distance from QID to all vertices in graph
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

// Description : intial reduction technique to reduce R.
// Add all vertex with core value > k lower and distance <= upper bound distance to G0 (intial R)
void reduction_g()
{   
    // Description : Vector that stores vertecies for Intial R
    G0.clear();
    // Description : Array that stores the neighbors of vertices in G0.  
    G0_edges = new ui[2*m];
    // Description : Array that stores the number of neighbors of vertices in G0.
    G0_x = new ui[n];

    // Description : array that stores the degree of vertices in G0.
    // As this can change, if one of its neighbor is removed from R in future. that is why we stores the number of neighbors in G0_X too. 
    G0_deg  = new ui[n];
    

    memset(G0_x, 0, sizeof(ui)*n);
    memset(G0_deg, 0, sizeof(ui)*n);
    
    // Description : Bool array 1 if in queue Q else 0.
    bool * inQ = new bool[n];
    memset(inQ, 0, sizeof(bool)*n);
    

    // Description : queue to stores vertecies that will be  processed. 
    queue<ui> Q;

    // Description : Push query vertex to queue. 
    Q.push(QID);
    inQ[QID] = 1;
    

    // Description : iterate till queue is empty. 
    while (!Q.empty()) {

        // Description : Get first vertex (v) from queue. 
        ui v = Q.front();
        Q.pop();

        // Description : Push to G0
        G0.push_back(v);

        // Description : iterate through neighbors of V
        for(ui i = pstart[v]; i < pstart[v+1]; i++){

            // Description : if core value of a neighbor is > k lower and distance <= upper bound distance. 
            // Add increament degree of V and neighbor. if neighbor not in queue push to queue. 
            if(core[edges[i]] > kl && q_dist[edges[i]] <= ubD){
                G0_edges[pstart[v] + G0_x[v]] = edges[i];
                ++ G0_x[v];
                ++ G0_deg[v];
                if(!inQ[edges[i]]){
                    Q.push(edges[i]);
                    inQ[edges[i]] = 1;
                }
            }
        }
    }
    delete [] inQ;
}

int find_ustar()
{   
    // Description : return ustar with maximum connection score
    // iterate through NEI (vertices that are in R), get all its neighbors in C and calculate 1/degree in (c)
    int uid = -1;
    double best_score = 0;
    for(auto e : NEI){
        if(inVR[e]){
            double its_score = 0;

            // Desciption : Get neighbors that are in C and add 1 / degree (neighbor) to score 
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                if(inVI[G0_edges[i]] && degVI[G0_edges[i]] != 0)
                  its_score += (double) 1/degVI[G0_edges[i]];
            }

            // Different : Add Degree (vertex) in R / max graph degree. Not mentioned in paper
            its_score += (double)degVIVR[e]/dMAX;

            // Description : update max score and ustar
            if(its_score > best_score){
                best_score = its_score;
                uid = e;
            }
        }
    }

    // Description : ustar
    return uid;
}

int find_ustar_mindeg()
{   

    // Description : reurns utar based on connection score and less expensive that find ustar method. 
    // iterate through VI (C) , get vertecies with same degree, 
    // iterate through same degree vertex, get their neighbors for each neighbor calculate the 1/ degree of neighbor of neighbor 
    // return one with max score
    double best_score;
    int uid = -1;
    // Description : set of degree in VI (C)
    set<ui> dict_deg;


    // Description : get set of unique degree of VI (C)
    for(auto e : VI){
        dict_deg.insert(degVI[e]);
    }


    // Description : itterate through degrees
    for(auto deg : dict_deg){

        // Description : Get vertices of VI (C)  that have the same degree
        vector<ui> vt;
        for(auto e : VI){
            if(degVI[e]==deg){
                vt.push_back(e);
            }
        }

        best_score = 0;

        // Description : itterate through same degree verticies (v) of VI (C) 
        for(auto e : vt){

            // Description : Get neighbors of v in VR (R). (w)
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                ui w = G0_edges[i];
                if(inVR[w]){
                    double its_score = 0;

                    // Description : get neighbors of W in VI (C).
                    for(ui j = pstart[w]; j < pstart[w]+G0_x[w]; j++){
                        if(inVI[G0_edges[j]] && degVI[G0_edges[j]] != 0){

                            // Description : add 1/ degree of neighbors in C to score 
                            its_score += (double) 1/degVI[G0_edges[j]];
                        }
                    }
                    

                    // Different : Add Degree (vertex) in R / max graph degree. Not mentioned in paper
                    its_score += (double)degVIVR[w]/dMAX;

                    // Description : Update the best score and ustar 
                    if(its_score > best_score){
                        best_score = its_score;
                        uid = w;
                    }
                }
            }
        }

        // Description : If any ustar is found, break
        // This doesn't give the ustar with max connection score but saves time
        // Different: never mentioned in paper
        if(uid != -1)
            break;
    }

    // Description : return ustar
    return uid;
}

int find_ustar_2phase()
{
    // Description : if size of VI is greater than that, use the less expensive ustar method.
    // Confusion : Don't know the reason for this number 
    if( VI.size() > (N2*2)/5 )
        return find_ustar_mindeg();
    else
        return find_ustar();
}
int find_ustar_link()
{   

    // Description : returns the vertex with most link if added to C


    int uid = -1;


    double best_score = 0;

    // Description : itterate through NEI (neighbors of vertices of VI (C) that are in VR (R))
    for(auto e : NEI){


        if(inVR[e]){
            double its_score = 0;

            // Description : Get neighbors of vertex (e)
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){

                // Description : if neighbor is VI (C). increament the score 
                if(inVI[G0_edges[i]])
                  its_score ++;
            }

            // Description : update the max score and ustar
            if(its_score >= best_score){
                best_score = its_score;
                uid = e;
            }
        }
    }

    // Description : return ustar
    return uid;
}
int find_ustar_random()
{   

    // Description : returns random vertex from R
    int uid = -1;

    
    for(auto e : NEI){
        if(inVR[e]){
            return  e;
        }
    }
    return uid;
}

// Description : FInd the set of vertiies that are dominated by ustar    
void find_domS_of_ustar(ui ustar, vector<pair<double, ui>> & domS)
{

    // Description :  get dominating set and sort them by connection score

    // Description : itterate through NEI (neighbors of vertices of VI (C) that are in VR (R) )
    for(auto e : NEI){

        // Decription : if degree of vertex <= degree of Ustar in both VI (C) and VR (R).
        // Issue : no need for this condition    
        if( inNEI[e] != 0 && degVI[e] <= degVI[ustar] && degVIVR[e] <= degVIVR[ustar] && e != ustar){
            ui ustar_sidx = pstart[ustar];
            ui ustar_eidx = pstart[ustar] + G0_x[ustar];
            bool be_dom = true;

            // Dscription: get the neighbors
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                ui w = G0_edges[i];

                // Description : if neighbor in VI (C) or VR (R).  
                if( (inVR[w] || inVI[w]) && w != ustar){

                    // Condusion : We should check if all neighbors are neighbors of ustar or ustar itself, 
                    // Didn't understand the condition. 
                    while(G0_edges[ustar_sidx] < w && ustar_sidx < ustar_eidx)
                        ++ ustar_sidx;
                    if(G0_edges[ustar_sidx] == w) continue;
                    else{
                        be_dom = false;
                        break;
                    }
                }
            }

            // Description : if vertex belongs to dominating set, we add the vertex and its connection score to doms. 
            if(be_dom){
                double its_score = 0;

                // Description : Get nieghbors of vertex. 
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){

                    // Description : if neighbors in VI
                    if(inVI[G0_edges[i]] && degVI[G0_edges[i]] != 0)

                      // Description : add to connection score 
                      its_score += (double) 1/degVI[G0_edges[i]];
                }

                its_score += (double)degVIVR[e]/dMAX;

                // Description : add to doms 
                domS.push_back(make_pair(its_score, e));
            }
        }

        // Description : limits dominating set size by a threshold, 
        if(domS.size() > domS_Threshold) break;
    }

    // Description : sort by connection score. 
    if(domS.size() > 1) sort(domS.begin(), domS.end(), greater<>());
}

ui get_ub1()
{
    // itterate through VI, get degree in C+R + min(degree in R, N2-|C| )
    ui min_deg = INF;
    ui r = N2 - (ui)VI.size();
    if(r<0){
        cout<<"??? r < 0 ???"<<endl;
        exit(1);
    }
    for(auto e : VI){
        ui its_deg_ub = 0;
        ui cands = 0;
        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
            if(inVR[G0_edges[i]]) ++ cands;
        }
        its_deg_ub = degVI[e] + min(r, cands);
        if(its_deg_ub < min_deg) min_deg = its_deg_ub;
    }
    return min_deg;
}

ui get_ub2()
{
    vector<ui> nei;
    bool * innei = new bool[n];
    memset(innei, 0, sizeof(bool)*n);
    for(auto e : VI){
        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
            if(inVR[G0_edges[i]] && !innei[G0_edges[i]]){
                nei.push_back(G0_edges[i]);
                innei[G0_edges[i]] = 1;
            }
        }
    }
    delete [] innei;
    
    ui r = N2 - (ui)VI.size();
    vector<ui> cov_power;
    for(auto e : nei){
        ui its_power = 0;
        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
            if(inVI[G0_edges[i]])
                ++ its_power;
        }
        cov_power.push_back(its_power);
    }
    ui r1 = (ui)cov_power.size();
    ui r2 = min(r, r1);
    sort(cov_power.begin(), cov_power.end(), greater<>());//decreasing order
    vector<ui> interm_deg;
    for(auto e : VI){
        interm_deg.push_back(degVI[e]);
    }
    sort(interm_deg.begin(), interm_deg.end(), less<>());//increasing order
    
    for(ui i = 0; i < r2; i++){
        ui budget = cov_power[i];
        for(ui j = 0; j < budget; j++){
            ++ interm_deg[j];
        }
        sort(interm_deg.begin(), interm_deg.end(), less<>());//increasing order
    }
    
    return interm_deg[0];
}

ui get_ub3()
{
    Timer t;
    vector<ui> nei;
    bool * innei = new bool[n];
    memset(innei, 0, sizeof(bool)*n);
    for(auto e : VI){
        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
            if(inVR[G0_edges[i]] && !innei[G0_edges[i]]){
                nei.push_back(G0_edges[i]);
                innei[G0_edges[i]] = 1;
            }
        }
    }
    delete [] innei;
    
    set<ui> ditc_deg;
    for(auto e : VI){
        ditc_deg.insert(degVI[e]);
    }
    vector<ui> deg_lev;
    for(auto e : ditc_deg)
        deg_lev.push_back(e);
    
    ui ditc_deg_num = (ui)deg_lev.size();
    
    ui r = N2 - (ui)VI.size();
    
    ui min_deg = INF;
    ui rcd_i = 0;
    
    for(ui i = 0; i < ditc_deg_num; i++){
        ui t_deg = deg_lev[i];
        vector<ui> interm_deg;
        for(auto e : VI){
            if(degVI[e] <= t_deg){
                interm_deg.push_back(degVI[e]);
            }
        }
        sort(interm_deg.begin(), interm_deg.end(), less<>()); //increasing order
        
        vector<ui> cov_power;
        for(auto e : nei){
            ui its_power = 0;
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                if(inVI[G0_edges[i]] && degVI[G0_edges[i]] <= t_deg){
                    ++ its_power;
                }
            }
            if(its_power > 0){
                cov_power.push_back(its_power);
            }
        }
        sort(cov_power.begin(), cov_power.end(), greater<>()); //decreasing order
        
        ui r1 = (ui)cov_power.size();
        ui r2 = min(r, r1);
        
        for(ui i = 0; i < r2; i++){
            ui budget = cov_power[i];
            for(ui j = 0; j < budget; j++){
                ++ interm_deg[j];
            }
            sort(interm_deg.begin(), interm_deg.end(), less<>());//increasing order
        }
        if(min_deg > interm_deg[0]){
            min_deg = interm_deg[0];
            rcd_i = i;
        }
        if(min_deg <= kl)
            return min_deg;
        
        if(EXE_ub3_optimization){
            if(i < ditc_deg_num-1 && min_deg <= deg_lev[i+1]){
                return min_deg;
            }
        }
    }
    return min_deg;
}



// Description : Calculate Upper Bound of Minimum Degree 
ui compute_ub()
{
    ++ total_UB;
    ui ub1 = INF;

    // Description : Calculate the upper bound using degree (Technique 1 )
    if(EXE_ub1) ub1 = get_ub1();

    ui ub2 = INF;

    //Description : Calcuate the upper bound based on neighbor reconstruction (Technique 2 )
    if(EXE_ub2) ub2 = get_ub2();

    ui ub3 = INF;

    // Description : Calculate the upper bound using technique 3 
    if(EXE_ub3) ub3 = get_ub3();

    total_val_ub1 += ub1;
    total_val_ub3 += ub3;
    return min(ub1, min(ub2,ub3));
}

void core_maintenance(bool & del_v_in_VI, vector<ui> & rVR, ui ustar)
{
    
    queue<ui> Q;
    bool * inQ = new bool[n];
    memset(inQ, 0, sizeof(bool)*n);
    
    for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
        ui w = G0_edges[i];
        if((inVI[w] || inVR[w]) && degVIVR[w] <= kl){
            Q.push(w);
            inQ[w] = 1;
        }
    }
    
    while (!Q.empty()) {
        ui v = Q.front();
        Q.pop();
        if(inVI[v]){
            del_v_in_VI = true;
            break;
        }
        else if(inVR[v]){
            inVR[v] = 0;
            rVR.push_back(v);
        }
        for(ui i = pstart[v]; i < pstart[v]+G0_x[v]; i++){
            ui w = G0_edges[i];
            if(inVI[w] || inVR[w]){
                -- degVIVR[w];
                -- degVIVR[v];
//                cout<<" due to the removal of vertex "<<v<<", --degVIVR["<<w<<"] = "<<degVIVR[w]<<endl;
//                cout<<" due to the removal of vertex "<<v<<", --degVIVR["<<v<<"] = "<<degVIVR[v]<<endl;
                if(degVIVR[w] <= kl && !inQ[w]){
                    Q.push(w);
                    inQ[w] = 1;
                }
            }
        }
    }
    delete [] inQ;
}

void core_maintenance(bool & del_v_in_VI, vector<ui> & rVR, vector<ui> & del_vec)
{
    queue<ui> Q;
    bool * inQ = new bool[n];
    memset(inQ, 0, sizeof(bool)*n);
    
    for(auto e : del_vec){
        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
            ui w = G0_edges[i];
            if( (inVI[w] || inVR[w]) && degVIVR[w] <= kl && !inQ[w]){//?????????
                Q.push(w);
                inQ[w] = 1;
            }
        }
    }

    while (!Q.empty()) {
        ui v = Q.front();
        Q.pop();
        if(inVI[v]){
            del_v_in_VI = true;
            break;
        }
        else if(inVR[v]){
            inVR[v] = 0;
            rVR.push_back(v);
        }
        for(ui i = pstart[v]; i < pstart[v]+G0_x[v]; i++){
            ui w = G0_edges[i];
            if(inVI[w] || inVR[w]){
                -- degVIVR[w];
                -- degVIVR[v];
                if(degVIVR[w] <= kl && !inQ[w]){
                    Q.push(w);
                    inQ[w] = 1;
                }
            }
        }
    }//while
    delete [] inQ;
}

void core_maintenance(bool & del_v_in_VI, vector<ui> & rVR, vector<pair<double, ui>> & del_vec)
{
    queue<ui> Q;
    bool * inQ = new bool[n];
    memset(inQ, 0, sizeof(bool)*n);
    
    for(auto ee : del_vec){
        auto e = ee.second;
        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
            ui w = G0_edges[i];
            if( (inVI[w] || inVR[w]) && degVIVR[w] <= kl && !inQ[w]){//?????????
                Q.push(w);
                inQ[w] = 1;
            }
        }
    }

    while (!Q.empty()) {
        ui v = Q.front();
        Q.pop();
        if(inVI[v]){
            del_v_in_VI = true;
            break;
        }
        else if(inVR[v]){
            inVR[v] = 0;
            rVR.push_back(v);
        }
        for(ui i = pstart[v]; i < pstart[v]+G0_x[v]; i++){
            ui w = G0_edges[i];
            if(inVI[w] || inVR[w]){
                -- degVIVR[w];
                -- degVIVR[v];
                if(degVIVR[w] <= kl && !inQ[w]){
                    Q.push(w);
                    inQ[w] = 1;
                }
            }
        }
    }//while
    delete [] inQ;
}
