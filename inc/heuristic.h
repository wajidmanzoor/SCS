#include "../src/Graph.h"

// Description : Greedy Algorithm to get the intial fesible solution H
// Start from query vertex as H
// It greadily add the vertex from neighbors of H, that will have the higest degree if added to H. 
// and so on 
void heu2(vector<ui> & H2, ui & kl2)
{
    // Description: Final result H and max min degree 
    H2.clear();
    kl2 = 0;
    
    // Description: Temporary H and max min degree
    vector<ui> tH;
    ui tkl = 0;
    ui tH_size = 0;
    
    // Description: array. 0 means not in H or Queue, 1 means in Queue and 2 means in Temporary H
    ui * sta = new ui[n];
    memset(sta, 0, sizeof(ui)*n);
    

    // Decription: Degree of vertex in Temporary H 
    ui * deg = new ui[n];
    memset(deg, 0, sizeof(ui)*n);
    
    // Description: Priority queue of vertices (v) based on degree of vertex if added in Temporary H.i.e degree_(T U v) 
    priority_queue<pair<ui, ui>> Q;

    // Description: Push Query Vertex to Queue. 
    Q.push(make_pair(0, QID));
    sta[QID] = 1;
    
    // Description : Iterate till priority queue is empty
    while(!Q.empty()){

        // Description: Get the top element i.e the one with higest degree
        ui v = Q.top().second;
        Q.pop();

        // Description: If already in Temporray H, move to next
        if(sta[v] == 2) continue;

        // Description : If not in temporary H, push to temporary H
        tH.push_back(v);
        sta[v] = 2;

        // Description : Iterate through all the neghbors 
        for(ui j = pstart[v]; j < pstart[v+1]; j++){
            ui nei = edges[j];

            // Description : If neighbor (nei) is neither in Temporary H nor in priority queue. 
            if(sta[nei] == 0){
                ui d = 0;

                // Description : calculate num of neighbors of nei that are in Temporary H. 
                // I.e The degree of vertex (nei) in Temporary H if added in temporary H. 
                for(ui w = pstart[nei]; w < pstart[nei+1]; w++){
                    if(sta[edges[w]] == 2) ++ d;
                }

                // Description : Push to priority queue. 
                Q.push(make_pair(d, nei));
                sta[nei] = 1;
            }
            else{

                // Description: If neighbor in priority Queue
                if(sta[nei] == 1){
                    ui new_d = 0;

                    // Description: update its degree in prority queue. 
                    for(ui w = pstart[nei]; w < pstart[nei+1]; w++){
                        if(sta[edges[w]] == 2) ++ new_d;
                    }
                    Q.push(make_pair(new_d, nei));
                }

                // Description : if neighbor in temporary H
                else{

                    //Description : Update the degree of neighbor and the vertex in Tempoary H.  
                    ++ deg[nei];
                    ++ deg[v];
                }
            }
        }

        // Description : check size constraint
        if(tH.size()>=N1){
            ui mindeg = INF;

            // Description: calculate the minumum degre of temporary H.
            for(ui i = 0; i < tH.size(); i++){
                if(deg[tH[i]] < mindeg)
                    mindeg = deg[tH[i]];
            }

            // Description: if current minumn degree > max minimum degree. Update max minimum degree.
            // Issue: we should also update the H 
            if(mindeg >= tkl){
                tkl = mindeg;
                tH_size = (ui)tH.size();
            }
        }

        // Description: if size equal to upper bound break.
        if(tH.size()==N2)
            break;
    }
    
    // Description : Copy Temporary H to H. 
    for(ui i=0; i<tH_size; i++){
        H2.push_back(tH[i]);
    }

    // Description: Temporary max min degree  to result max min degree. 
    kl2 = tkl;
    
    delete [] sta;
    delete [] deg;
}

// Description : Greedy Algorithm to get the intial fesible solution H
// Same as heuristic algorithm, except the vertex with higest connection score is added. 
void heu3(vector<ui> & H3, ui & kl3)
{
    // Description: Final result H and max min degree 
    H3.clear();
    kl3 = 0;
    
    // Description: Temporary H and max min degree
    vector<ui> tH;
    ui tkl = 0;
    ui tH_size = 0;
    
    // Description: array. 0 means not in H or Queue, 1 means in Queue and 2 means in Temporary H
    ui * sta = new ui[n];
    memset(sta, 0, sizeof(ui)*n);
    
    
    ui * deg = new ui[n];
    memset(deg, 0, sizeof(ui)*n);
    
    // Decription: Degree of vertex in Temporary H 
    priority_queue<pair<double, ui>> Q;

    // Description: Push Query Vertex to Queue. 
    Q.push(make_pair(0, QID));
    sta[QID] = 1;
    

    // Description : Iterate till priority queue is empty
    while (!Q.empty()) {

        // Description: Get the top element i.e the one with higest degree
        ui v = Q.top().second;
        Q.pop();

        // Description: If already in Temporray H, move to next
        if(sta[v] == 2) continue;

        // Description : If not in temporary H, push to temporary H
        tH.push_back(v);
        sta[v] = 2;

        // Description : Iterate through all the neghbors to update the degree of vertecies that all already in Temporary H 
        for(ui nei = pstart[v]; nei<pstart[v+1]; nei++){
            if(sta[edges[nei]] == 2){
                ++ deg[edges[nei]];
                ++ deg[v];
            }
        }

        // Description : Iterate through all the neghbors
        for(ui nei = pstart[v]; nei<pstart[v+1]; nei++){

            // Description : If neighbor (nei) is neither in Temporary H nor in priority queue. 
            if(!sta[edges[nei]]){
                double score = 0;

                // Description : Iterate through neighbors of nei that are in Temporary H. 
                // To calculate the connection 1/(degree of neighbors). 
                for(ui w = pstart[edges[nei]]; w < pstart[edges[nei]+1]; w++){
                    if(sta[edges[w]] == 2 && deg[edges[w]] != 0){
                        score += (double) 1/deg[edges[w]];
                    }
                }
                score += (double) degree[edges[nei]]/dMAX;

                // Description : Push to priority queue. 
                Q.push(make_pair(score, edges[nei]));
                sta[edges[nei]] = 1;
            }
            else{

                // Description: If neighbor in priority Queue
                if(sta[edges[nei]] == 1){
                    double new_score = 0;

                    // Description: update its connection score in prority queue. 
                    for(ui w = pstart[edges[nei]]; w < pstart[edges[nei]+1]; w++){
                        if(sta[edges[w]] == 2 && deg[edges[w]] != 0){
                            new_score += (double) 1/deg[edges[w]];
                        }
                    }
                    new_score += (double) degree[edges[nei]]/dMAX;
                    Q.push(make_pair(new_score, edges[nei]));
                }
            }
        }

        // Description : check size constraint
        if(tH.size()>=N1){
            ui mindeg = INF;

            // Description: calculate the minumum degre of temporary H.
            for(ui i = 0; i < tH.size(); i++){
                if(deg[tH[i]] < mindeg)
                    mindeg = deg[tH[i]];
            }

            // Description: if current minumn degree > max minimum degree. Update max minimum degree.
            // Issue: we should also update the H 
            if(mindeg >= tkl){
                tkl = mindeg;
                tH_size = (ui)tH.size();
            }
        }

        // Description: if size equal to upper bound break.
        if(tH.size()==N2)
            break;
    }
    
    // Description : Copy Temporary H to H. 
    for(ui i=0; i<tH_size; i++){
        H3.push_back(tH[i]);
    }

    // Description: Temporary max min degree  to result max min degree. 
    kl3 = tkl;
    
    delete [] sta;
    delete [] deg;
}

// Confusion : Didn't understand the logic 
void heu4(vector<ui> & H4, ui & kl4)
{
    H4.clear();
    kl4 = 0;
    
    if( degree[QID] < N1-1){
        return;
    }
    
    vector<ui> S;
    bool * inS = new bool[n];
    memset(inS, 0, sizeof(bool)*n);
    
    ui * deg = new ui[n];
    memset(deg, 0, sizeof(ui)*n);
    
    S.push_back(QID);
    inS[QID] = 1;
    ++ deg[QID];
    
    for(ui j = pstart[QID]; j < pstart[QID+1]; j++){
        S.push_back(edges[j]);
        inS[edges[j]] = 1;
        for(ui w = pstart[edges[j]]; w < pstart[edges[j]+1]; w++){
            if(inS[edges[w]]){
                ++ deg[edges[w]];
                ++ deg[edges[j]];
            }
        }
    }

    priority_queue<pair<ui,ui>, vector<pair<ui,ui>>, greater<>> Q;
    for(auto e : S)
        Q.push(make_pair(deg[e], e));
    vector<ui> rmv;
    ui rmv_idx = 0;
    ui mindeg = 0;
    while (!Q.empty()) {
        ui d = Q.top().first;
        ui v = Q.top().second;
        Q.pop();
        if(inS[v] == 0) continue;
        ui remain = (ui)S.size() - (ui)rmv.size();
        if(remain >= N1 && remain <= N2){
            if(d > mindeg){
                mindeg = d;
                rmv_idx = (ui)rmv.size();
            }
        }
        if(remain == N1) break;

        inS[v] = 0;
        rmv.push_back(v);
        for(ui j = pstart[v]; j < pstart[v+1]; j++){
            if(inS[edges[j]]){
                -- deg[edges[j]];
                Q.push(make_pair(deg[edges[j]], edges[j]));
            }
        }
    }
    
    memset(inS, 0, sizeof(bool)*n);
    for(auto e : S)
        inS[e] = 1;
    for(ui i = 0; i < rmv_idx; i++)
        inS[rmv[i]] = 0;
    
    vector<ui> tH;
    ui tkl = mindeg;
    for(auto e : S)
        if(inS[e]) tH.push_back(e);
    
    for(ui i=0; i<tH.size(); i++){
        H4.push_back(tH[i]);
    }
    kl4 = tkl;
    
    delete [] inS;
    delete [] deg;
    
}



// Description: Calls multiple heusteric algorithms and choses the one what returns max min degree
void CSSC_heu()
{
    H.clear();
    kl = 0;
    
    vector<ui> H2; ui kl2 = 0;
     heu2(H2,kl2);
    vector<ui> H3; ui kl3 = 0;
    heu3(H3,kl3);
    vector<ui> H4; ui kl4 = 0;
     heu4(H4,kl4);
    if(kl4 >= kl3 && kl4 >= kl2){
        H = H4;
        kl = kl4;
    }
    else{
        if(kl3 >= kl2){
            H = H3;
            kl = kl3;
        }
        else{
            H = H2;
            kl = kl2;
        }
    }
}