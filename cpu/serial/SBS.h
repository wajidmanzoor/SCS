#include "helpers.h"
#include "Timer.h"

// Description : SBS algorithm without branching technique
void BB(ui level )
{   if (verbose){
    for (int i = 0; i < level - 1; ++i) {
        cout << "|   ";
    }
    cout << "+--- Level: " << level << "  C ";
    for(auto e : VI){
        cout << e << "  ";
    }
    cout <<endl;
}
    // Description: If algorithm takes too much time stop. 
    if(over_time_flag) return;
    
    double DurTime = (double)clock() / CLOCKS_PER_SEC - StartTime;
    
    if(DurTime > MaxTime) over_time_flag = true;
     

    // Description : If size of C is greater than N2. stop
    // VI is same as C in paper and VR is same as R in paper
    // H is the final result (optimal solution )
    // K lower is maximum minumum degree
    if(VI.size() > N2) return;

    //Description : check if size is between N1 and N2, check and update the K_lower and H
    if(VI.size() >= N1 && VI.size() <= N2){

        ui cur_min_deg = INF;
        // Description : Get current Min degree of VI (C)
        for(auto e : VI){
            if(degVI[e] < cur_min_deg)
                cur_min_deg = degVI[e];
        }

        // Description : check if currrent min degree is greater than k lower. Update the K lower and H.
        if(cur_min_deg > kl){
            kl = cur_min_deg;
            H = VI;

            // Description : Based on K lower and upper bound size. calculate the upper bound of distance. 
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
    }
    

    // Description : If size of VI (C) is N2 break
    if(VI.size() == N2) return;
    

    // Description : Reduction rule 3 . Given an instance (𝐶, 𝑅) and any 𝑢 ∈ 𝐶, if 𝑑𝐶∪𝑅 (𝑢) = ˜𝑘 +1, then we can greedily move to 𝐶 all the vertices in 𝑅 that are neighbors of 𝑢.
    // New set is created to store the verticies that need to be moved
    unordered_set<ui> new2VI;
    if(EXE_new2VI){

        // Description : Iterate through all verticies in VI (C)
        for(auto e : VI){

            // Descrition : if degree in R is Klower + 1
            if(degVIVR[e] == kl+1){
                vector<ui> its_nei;

                // Description : get neighbors
                for(ui i = pstart[e]; i < pstart[e] + G0_x[e]; i++){
                    ui w = G0_edges[i];

                    // Description : push neighbors that are in R to temporary vector
                    if(inVR[w]){
                        its_nei.push_back(w);
                    }
                }

                // Description : Copy from temporary vector to new2VI 
                if(its_nei.size() != 0){
                    for(auto x : its_nei){
                        new2VI.insert(x);
                    }
                }
            }
        }
        // Description : Copy from new2VI to VI
        for(auto e : new2VI){
            if(inVR[e]){

                //Description : Add to VI (C) and Remove from VR (R)
                inVI[e] = 1;
                inVR[e] = 0;
                VI.push_back(e);
                // Descriptin : Increament degree in VI (C)
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++ ){
                    if(inVI[G0_edges[i]]){
                        ++ degVI[G0_edges[i]];
                        ++ degVI[e];
                    }
                }
            }
        }

        // Description : if size of VI (C) becomes greater than N2 remove all 
        // Issue : shouldn't we just add till the size is between N1 to N2. Rather than removing all 
        if(VI.size() > N2){
            for(auto e : new2VI){
                VI.pop_back();

                //Description : remove from VI (C) and add to VR (R)
                inVI[e] = 0;
                inVR[e] = 1;

                //Description : decreament degree in VI (C)
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];

                    }
                }
            }
            return;
        }
    }

    //Description : if Size is between N1 and N2, check and update the K_lower and H
    if(VI.size() >= N1 && VI.size() <= N2){
        ui cur_min_deg = INF;

        // Description: Get current min degree 
        for(auto e : VI){
            if(degVI[e] < cur_min_deg)
                cur_min_deg = degVI[e];
        }

        // Description : compare and update 
        if(cur_min_deg > kl){
            kl = cur_min_deg;
            H = VI;

            // Description : Caculate distance uper bound based on new K lower
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
    }

    //Issue: Remove even  if size == N2. Didn't understand why
    if(VI.size() == N2){
        if(EXE_new2VI){
            for(auto e : new2VI){

                // Description : remove from VI (C) add to VR (R)
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;

                // Description : decreament degree 
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];

                    }
                }
            }
        }
        return;
    }



    // Description : Reduction rule 1 Given an instance (𝐶, 𝑅) and any vertex 𝑣 ∈ 𝑅, if min{𝑑𝐶∪𝑅 (𝑣), 𝑑𝐶∪{𝑣 } (𝑣) + ℎ − |𝐶| − 1} ≤ ˜𝑘, then we can discard 𝑣 from 𝑅, where 𝑑𝐶∪𝑅 (𝑣) is the degree of 𝑣 in the subgraph 𝐺[𝐶 ∪ 𝑅].

    // Description : NEI stores the neighbors of all vertices of VI (C) that are in R 
    NEI.clear();

    // Description : INNEI stores the number of neighbors in C U V for all verticies in NEI (Connection to C )
    memset(inNEI, 0, sizeof(ui)*n);

    // Description : interate through VI (C)
    for(auto e : VI){

        // Description : Get all neighbors of v in graph 
        for(ui i = pstart[e]; i < pstart[e] + G0_x[e]; i++){

            // Decription : if neighbor in R 
            if(inVR[G0_edges[i]]){

                // Description : if neighbor not added in NEI 
                if(inNEI[G0_edges[i]] == 0){

                    // Description : add to nei and set is neighbor count to 1 
                    NEI.push_back(G0_edges[i]);
                    inNEI[G0_edges[i]] = 1;
                }

                // Description : if already in NEI, increament the neighbor count 
                else{
                    ++ inNEI[G0_edges[i]];
                }
            }
        }
    }

    
    // Description : Vector that will store the vertices that will be deleted from R based on the reduction rule 1 
    // Different: Only considers 𝑑𝐶∪{𝑣 } (𝑣) + ℎ − |𝐶| − 1 ≤ ˜𝑘 and not min{𝑑𝐶∪𝑅 (𝑣), 𝑑𝐶∪{𝑣 } (𝑣) + ℎ − |𝐶| − 1} ≤ ˜𝑘
    vector<ui> del_from_VR;
    if(EXE_del_from_VR){

        // Description : iterate through NEI (neighbors of vertices of C that are in R )
        for(auto e : NEI){

            // Description : if neighbor has number of connection with C < k lower + 1 
            if(inNEI[e] < kl+1){

                // Description : Number of lacking connections (K=1 - 𝑑𝐶∪{𝑣 } (v))
                int lack = kl + 1 - inNEI[e];

                // Descripton : Upper bound of connection that might be possible.  (N2- |C| )
                int bugt = N2 - (int)VI.size() - 1;

                // Description: If lack > buget 
                if( lack > bugt ){

                    // Description : push to delete from R vector 
                    del_from_VR.push_back(e);

                    // Description : set 0 in array that indicate if in R and NEI 
                    inVR[e] = 0;
                    inNEI[e] = 0;

                    // Description : decreament degree of neighbors in R. 
                    for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                        ui w = G0_edges[i];
                        if(inVR[w] || inVI[w]){
                            -- degVIVR[w];
                            -- degVIVR[e];

                        }
                    }
                }
            }
        }
    }

    int ustar = -1;

    // Description : Find the vertex from R  that will be used to create branches
    // Connection Score :  Given an instance (𝐶, 𝑅), the connection score of a vertex 𝑣 ∈ 𝑅 is defined as 𝛿 (𝑣) = sum 𝑢 ∈𝑁𝐶∪{𝑣} (1/ 𝑑𝐶 (𝑢))

    //Description : Vertex will be slected based on maximum connection score 
    if(srch_ord == 1) ustar = find_ustar();

    // Description : Using find_ustar_mindeg if size of VI (C) > 2*N2/5 else find_ustar
    // Confusion :  Not sure why it is based on VI size and not VR size
    else if(srch_ord == 2) ustar = find_ustar_2phase();

    //Description : Vertex will be slected based on connection score. we wil check the connection score of neighbors in R of vertices which have same degree in C. 
    // After one iteration if found any, will break to reduce time.  function will alwasys not return the vertex that has highest connection score but will same time
    else if(srch_ord == 3) ustar = find_ustar_mindeg();

    // Description : vertex will be slected based on connection links to C. 
    else if(srch_ord == 4) ustar = find_ustar_link();

    // Description : Returns a random vertex from R 
    else if(srch_ord == 5) ustar = find_ustar_random();

    
    
    //Description : stop scenario. e.g R is empty or none of the vertices in R are connected to vertices in C. 
    if(ustar < 0){

        // Description :  Add vertices deleted from R (reduction rule 1 ) back to R
        if(EXE_del_from_VR){
            for(auto e : del_from_VR){
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVR[w] || inVI[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[e];

                    }
                }
            }
        }
        
        // Description : Remove vertices that were greadly moved to C (reduction rule 3 ) from C
        if(EXE_new2VI){
            for(auto e : new2VI){
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];

                    }
                }
            }
        }
        return;
    }
    

    // Description : Add  ustar to C and remove it from R
    VI.push_back(ustar);
    inVI[ustar] = 1;
    inVR[ustar] = 0;

    // Description: Calculate degree of ustar and neighbors of ustar in C
    for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
        if(inVI[G0_edges[i]]){
            ++ degVI[G0_edges[i]];
            ++ degVI[ustar];

        }
    }

    //Issue:  should have decreamented  1 from neighbors of ustar in R. 

    // Description : Get upper bound of min degree for C + ustar, R-ustar
    // Different : Paper claimes that "In Implementation, compute 𝑈𝑑 ,then 𝑈𝑛𝑟, and finally 𝑈𝑑𝑐 in increasing order of their time complexities. Once a computed upper bound is enough to prune the instance (𝐶, 𝑅), we terminate the upper bound computation immediately."
    ui ub = compute_ub();
    
    if(ub > kl && q_dist[ustar] <= ubD){
        // Description : Resurcive function with C + ustar, R-ustar
    
        BB(level +1 );
    }
    
    // Description : Remove ustar from VI (C)
    VI.pop_back();
    inVI[ustar] = 0;


    // Description: decreament 1 from neighbors of ustar in C and R 
    for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
        if(inVI[G0_edges[i]]){
            -- degVI[G0_edges[i]];
            -- degVI[ustar];

        }
        if(inVI[G0_edges[i]] || inVR[G0_edges[i]]){
            -- degVIVR[G0_edges[i]];
            -- degVIVR[ustar];
        }
    }

    vector<ui> rVR;

    bool del_v_in_VI = false;


    // Description : remove from C and R the neighbors of ustar whose degree is <= K_lower 
    if(EXE_core_maintenance) core_maintenance(del_v_in_VI, rVR, ustar);


     
    if(EXE_core_maintenance){
        if(del_v_in_VI){
            for(ui i = 0; i < rVR.size(); i++){
                ui v = rVR[i];
                inVR[v] = 1;
                for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                    ui w = G0_edges[j];
                    if(inVI[w] || inVR[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[v];

                    }
                }
            }
            
            inVR[ustar] = 1;
            for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
                ui w = G0_edges[i];
                if(inVI[w] || inVR[w]){
                    ++ degVIVR[w];
                    ++ degVIVR[ustar];
                }
            }
            
            if(EXE_del_from_VR){
                for(auto e : del_from_VR){
                    inVR[e] = 1;
                    for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                        ui w = G0_edges[i];
                        if(inVR[w] || inVI[w]){
                            ++ degVIVR[w];
                            ++ degVIVR[e];
                        }
                    }
                }
            }
            
            if(EXE_new2VI){
                for(auto e : new2VI){
                    VI.pop_back();
                    inVI[e] = 0;
                    inVR[e] = 1;
                    for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                        ui w = G0_edges[i];
                        if(inVI[w]){
                            -- degVI[w];
                            -- degVI[e];
                        }
                    }
                }
            }
            return;
        }
    }
    

    // Description : get uper bound of minimum degree for  C , R-ustar
    ub = compute_ub();
        
    if(ub > kl){
        // Description :  Resursive cal for C , r-ustar
        
        BB( level +1  );
    }
    
    if(EXE_core_maintenance){
        for(ui i = 0; i < rVR.size(); i++){
            ui v = rVR[i];
            inVR[v] = 1;
            for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                ui w = G0_edges[j];
                if(inVI[w] || inVR[w]){
                    ++ degVIVR[w];
                    ++ degVIVR[v];

                }
            }
        }
    }
    inVR[ustar] = 1;
    for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
        ui w = G0_edges[i];
        if(inVI[w] || inVR[w]){
            ++ degVIVR[w];
            ++ degVIVR[ustar];

        }
    }
    
    if(EXE_del_from_VR){
        for(auto e : del_from_VR){
            inVR[e] = 1;
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                ui w = G0_edges[i];
                if(inVR[w] || inVI[w]){
                    ++ degVIVR[w];
                    ++ degVIVR[e];

                }
            }
        }
    }
    if(EXE_new2VI){
        for(auto e : new2VI){
            VI.pop_back();
            inVI[e] = 0;
            inVR[e] = 1;
            for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                ui w = G0_edges[i];
                if(inVI[w]){
                    -- degVI[w];
                    -- degVI[e];

                }
            }
        }
    }
}


// Description : SBS algorithm with  branching techniques. 
void BB_dom_ustar(ui level )
{   
    if (verbose){
    for (int i = 0; i < level - 1; ++i) {
        cout << "|   ";
    }

    cout << "+--- Level: " << level << "  C ";
    for(auto e : VI){
        cout << e << "  ";
    }
    cout <<endl;
    }
    // Description: If algorithm takes too much time stop. 
    if(over_time_flag) return;
    
    double DurTime = (double)clock() / CLOCKS_PER_SEC - StartTime;
    
    if(DurTime > MaxTime)
        over_time_flag = true;

    // Description : If size of C is greater than N2. stop
    // VI is same as C in paper and VR is same as R in paper
    // H is the final result (optimal solution )
    // K lower is maximum minumum degree
    
    if(VI.size() > N2) return;

    //Description : check if size is between N1 and N2, check and update the K_lower and H
    if(VI.size() >= N1 && VI.size() <= N2){
        ui cur_min_deg = INF;

        // Description : Get current Min degree of VI (C)
        for(auto e : VI){
            if(degVI[e] < cur_min_deg)
                cur_min_deg = degVI[e];
        }

        // Description : check if currrent min degree is greater than k lower. Update the K lower and H.
        if(cur_min_deg > kl){
            kl = cur_min_deg;
            H = VI;

            // Description : Based on K lower and upper bound size. calculate the upper bound of distance. 
            for(ui d = 1; d <= N2; d++){
                if(d == 1 || d == 2){
                    if(kl + d > N2){
                        ubD = d-1;
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
    }

    // Description : If size of VI (C) is N2 break
    if(VI.size() == N2) return;
    unordered_set<ui> new2VI;
    

    // Description : Reduction rule 3 . Given an instance (𝐶, 𝑅) and any 𝑢 ∈ 𝐶, if 𝑑𝐶∪𝑅 (𝑢) = ˜𝑘 +1, then we can greedily move to 𝐶 all the vertices in 𝑅 that are neighbors of 𝑢.
    // New set is created to store the verticies that need to be moved
    if(EXE_new2VI){

        // Description : Iterate through all verticies in VI (C)
        for(auto e : VI){

            // Descrition : if degree in R is Klower + 1
            if(degVIVR[e] == kl+1){
                vector<ui> its_nei;

                // Description : get neighbors
                for(ui i = pstart[e]; i < pstart[e] + G0_x[e]; i++){
                    ui w = G0_edges[i];

                    // Description : push neighbors that are in R to temporary vector
                    if(inVR[w]){
                        its_nei.push_back(w);
                    }
                }

                // Description : Copy from temporary vector to new2VI 
                if(its_nei.size() != 0){
                    for(auto x : its_nei){
                        new2VI.insert(x);
                    }
                }
            }
        }

        // Description : Copy from new2VI to VI
        for(auto e : new2VI){
            if(inVR[e]){

                //Description : Add to VI (C) and Remove from VR (R)
                inVI[e] = 1;
                inVR[e] = 0;
                VI.push_back(e);
                
                // Descriptin : Increament degree in VI (C)
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++ ){
                    if(inVI[G0_edges[i]]){
                        ++ degVI[G0_edges[i]];
                        ++ degVI[e];
                    }
                }
            }
        }

        // Description : if size of VI (C) becomes greater than N2 remove all 
        // Issue : shouldn't we just add till the size is between N1 to N2. Rather than removing all
        if(VI.size() > N2){
            for(auto e : new2VI){
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];
                    }
                }
            }
            return;
        }
    }
    

    //Description : if Size is between N1 and N2, check and update the K_lower and H
    if(VI.size() >= N1 && VI.size() <= N2){
        ui cur_min_deg = INF;

        // Description: Get current min degree 
        for(auto e : VI){
            if(degVI[e] < cur_min_deg)
                cur_min_deg = degVI[e];
        }

        // Description : compare and update 
        if(cur_min_deg > kl){
            kl = cur_min_deg;
            H = VI;

            // Description : Caculate distance uper bound based on new K lower
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
    }

    //Issue: Remove even  if size == N2. Didn't understand why
    if(VI.size() == N2){

        if(EXE_new2VI){
            for(auto e : new2VI){

                // Description : remove from VI (C) add to VR (R)
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;

                // Description : decreament degree 
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];             }
                }
            }
        }
        return;
    }
     


    // Description : Reduction rule 1 Given an instance (𝐶, 𝑅) and any vertex 𝑣 ∈ 𝑅, if min{𝑑𝐶∪𝑅 (𝑣), 𝑑𝐶∪{𝑣 } (𝑣) + ℎ − |𝐶| − 1} ≤ ˜𝑘, then we can discard 𝑣 from 𝑅, where 𝑑𝐶∪𝑅 (𝑣) is the degree of 𝑣 in the subgraph 𝐺[𝐶 ∪ 𝑅].

    // Description : NEI stores the neighbors of all vertices of VI (C) that are in R 
    
    NEI.clear();

    // Description : INNEI stores the number of neighbors in C U V for all verticies in NEI (Connection to C )
    memset(inNEI, 0, sizeof(ui)*n);

    // Description : interate through VI (C)
    for(auto e : VI){

        // Description : Get all neighbors of v in graph 
        for(ui i = pstart[e]; i < pstart[e] + G0_x[e]; i++){

            // Decription : if neighbor in R 
            if(inVR[G0_edges[i]]){

                // Description : if neighbor not added in NEI
                if(inNEI[G0_edges[i]] == 0){

                    // Description : add to nei and set is neighbor count to 1 
                    NEI.push_back(G0_edges[i]);
                    inNEI[G0_edges[i]] = 1;
                }

                // Description : if already in NEI, increament the neighbor count 
                else{
                    ++ inNEI[G0_edges[i]];
                }
            }
        }
    }
   

    // Description : Vector that will store the vertices that will be deleted from R based on the reduction rule 1 
    // Different: Only considers 𝑑𝐶∪{𝑣 } (𝑣) + ℎ − |𝐶| − 1 ≤ ˜𝑘 and not min{𝑑𝐶∪𝑅 (𝑣), 𝑑𝐶∪{𝑣 } (𝑣) + ℎ − |𝐶| − 1} ≤ ˜𝑘
    vector<ui> del_from_VR;
    if(EXE_del_from_VR){

        // Description : iterate through NEI (neighbors of vertices of C that are in R )
        for(auto e : NEI){

            // Description : if neighbor has number of connection with C < k lower + 1 
            if(inNEI[e] < kl+1){

                // Description : Number of lacking connections (K=1 - 𝑑𝐶∪{𝑣 } (v))
                int lack = kl + 1 - inNEI[e];

                // Descripton : Upper bound of connection that might be possible.  (N2- |C| )
                int bugt = N2 - (int)VI.size() - 1;

                // Description: If lack > buget 
                if( lack > bugt ){

                    // Description : push to delete from R vector 
                    del_from_VR.push_back(e);
                    inVR[e] = 0;
                    inNEI[e] = 0;

                    // Description : decreament degree of neighbors in R. 
                    for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                        ui w = G0_edges[i];
                        if(inVR[w] || inVI[w]){
                            -- degVIVR[w];
                            -- degVIVR[e];
                        }
                    }
                }
            }
        }
    }



    // Description : Find the vertex from R  that will be used to create branches
    // Connection Score :  Given an instance (𝐶, 𝑅), the connection score of a vertex 𝑣 ∈ 𝑅 is defined as 𝛿 (𝑣) = sum 𝑢 ∈𝑁𝐶∪{𝑣} (1/ 𝑑𝐶 (𝑢))

    int ustar = -1;

    //Description : Vertex will be slected based on connection score. we wil check the connection score of neighbors in R of vertices which have same degree in C. 
    // After one iteration if found any, will break to reduce time.  function will alwasys not return the vertex that has highest connection score but will same time
    ustar = find_ustar_mindeg();
    
    

    //Description : stop scenario. e.g R is empty or none of the vertices in R are connected to vertices in C. 
    if(ustar < 0){

        // Description :  Add vertices deleted from R (reduction rule 1 ) back to R
        if(EXE_del_from_VR){
            for(auto e : del_from_VR){
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVR[w] || inVI[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[e];
                    }
                }
            }
        }

        // Description : Remove vertices that were greadly moved to C (reduction rule 3 ) from C
        if(EXE_new2VI){
            for(auto e : new2VI){
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];
                    }
                }
            }
        }
        return;
    }

    vector<pair<double, ui>> domS;

    // Description : find dominated vertex set of Ustar. 
    // Vertex Domination : Given an instance (𝐶, 𝑅), vertex 𝑣 ∈ 𝑅 dominates 𝑣′ ∈ 𝑅, denoted 𝑣 ⪰ 𝑣′, if every neighbor of𝑣′(in 𝐶 ∪ 𝑅) is either a neighbor of 𝑣 or is 𝑣 itself. 
    find_domS_of_ustar(ustar, domS);


    // Description : Use branching technique if dominating set is not empty 
    if(!domS.empty()){
        ++ domBr;

        // Description : branching with C + dominating set[i] + ustar , R - dominating set - ustar 
        // iterte through dominating verticies
        for(ui round = 0; round < domS.size(); round++){
            ui domv = domS[round].second;

            

            // Description : add ustar to VI (C)  and remove from VR (R) 
            // Issue: can be put outside the loop 
            VI.push_back(ustar);
            inVI[ustar] = 1;
            inVR[ustar] = 0;


            // Description : increament degree of  the neighbors of ustar in VI (C) 
            for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
                if(inVI[G0_edges[i]]){
                    ++ degVI[ustar];
                    ++ degVI[G0_edges[i]];
                }
            }

            // Description :  add vertex from dominating set  to VI (C)  and remove from VR (R). 
            VI.push_back(domv);
            inVI[domv] = 1;
            inVR[domv] = 0;

            // Description : increament degree of  neighbors of dom[i] in VI (C) 
            for(ui i = pstart[domv]; i < pstart[domv]+G0_x[domv]; i++){
                if(inVI[G0_edges[i]]){
                    ++ degVI[domv];
                    ++ degVI[G0_edges[i]];
                }
            }
            
            // Description :  remove dominating set upto the current dominating vertex  VR (R) 
            vector<ui> rmv_each_round;
            for(ui i = 0; i < round; i++){
                ui dv = domS[i].second;
                inVR[dv] = 0;
                rmv_each_round.push_back(dv);

                // Description : decreament the degree of neighbors of dominating set[i] in VR (R) 
                for(ui j = pstart[dv]; j < pstart[dv]+G0_x[dv]; j++){
                    if(inVI[G0_edges[j]] || inVR[G0_edges[j]]){
                        -- degVIVR[G0_edges[j]];
                        -- degVIVR[dv];
                    }
                }
            }
            
            vector<ui> rVR;
            bool del_v_in_VI = false;


            if(EXE_core_maintenance) core_maintenance(del_v_in_VI, rVR, rmv_each_round);
            if(EXE_core_maintenance){
                if(del_v_in_VI){
                    for(ui i = 0; i < rVR.size(); i++){
                        ui v = rVR[i];
                        inVR[v] = 1;
                        for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                            ui w = G0_edges[j];
                            if(inVI[w] || inVR[w]){
                                ++ degVIVR[w];
                                ++ degVIVR[v];
                            }
                        }
                    }
                    
                    for(ui i = 0; i < round; i++){
                        ui dv = domS[i].second;
                        inVR[dv] = 1;
                        for(ui j = pstart[dv]; j < pstart[dv]+G0_x[dv]; j++){
                            if(inVI[G0_edges[j]] || inVR[G0_edges[j]]){
                                ++ degVIVR[G0_edges[j]];
                                ++ degVIVR[dv];
                            }
                        }
                    }
                    
                    VI.pop_back();
                    inVI[domv] = 0;
                    inVR[domv] = 1;
                    for(ui i = pstart[domv]; i < pstart[domv]+G0_x[domv]; i++){
                        if(inVI[G0_edges[i]]){
                            -- degVI[domv];
                            -- degVI[G0_edges[i]];
                        }
                    }
                    
                    VI.pop_back();
                    inVI[ustar] = 0;
                    inVR[ustar] = 1;
                    for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
                        if(inVI[G0_edges[i]]){
                            -- degVI[ustar];
                            -- degVI[G0_edges[i]];
                        }
                    }
                    continue;
                }
            }
            // Description : Compute upper bound of C + Ustar + dominating set[i]
            ui ub = compute_ub();
            if(ub > kl){
                // // Description : Resursive on  C + Ustar + dominating set[i], R-ustar - dominating set- ustar  
                BB_dom_ustar(level + 1);
            }
            


            if(EXE_core_maintenance){
                for(ui i = 0; i < rVR.size(); i++){
                    ui v = rVR[i];
                    inVR[v] = 1;
                    for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                        ui w = G0_edges[j];
                        if(inVI[w] || inVR[w]){
                            ++ degVIVR[w];
                            ++ degVIVR[v];
                        }
                    }
                }
            }
            
            for(ui i = 0; i < round; i++){
                ui dv = domS[i].second;
                inVR[dv] = 1;
                for(ui j = pstart[dv]; j < pstart[dv]+G0_x[dv]; j++){
                    if(inVI[G0_edges[j]] || inVR[G0_edges[j]]){
                        ++ degVIVR[G0_edges[j]];
                        ++ degVIVR[dv];
                    }
                }
            }


            // Description : Remove dominating set[i] from VI (C), so that in next iteration new one can be added
            VI.pop_back();
            inVI[domv] = 0;
            inVR[domv] = 1;

            // Description : Decreament the degre of neighbors of dom [i] in VI (C)
            for(ui i = pstart[domv]; i < pstart[domv]+G0_x[domv]; i++){
                if(inVI[G0_edges[i]]){
                    -- degVI[domv];
                    -- degVI[G0_edges[i]];
                }
            }

            // Description : Remove ustar from VI (C) 
            // Issue: can be put outside the loop 
            VI.pop_back();
            inVI[ustar] = 0;
            inVR[ustar] = 1;

            // Description : decreament degree of neighbors of ustar in VI (C)
            for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
                if(inVI[G0_edges[i]]){
                    -- degVI[ustar];
                    -- degVI[G0_edges[i]];
                }
            }
        }
        

        // Description : remove dominating set from R
        for(ui i = 0; i < domS.size(); i++){
            ui dv = domS[i].second;
            inVR[dv] = 0;
            // Description : decreament degree of neighbors of dominating set in R 
            for(ui j = pstart[dv]; j < pstart[dv]+G0_x[dv]; j++){
                if(inVI[G0_edges[j]] || inVR[G0_edges[j]]){
                    -- degVIVR[G0_edges[j]];
                    -- degVIVR[dv];
                }
            }
        }
        
        vector<ui> rVR;
        bool del_v_in_VI = false;
        if(EXE_core_maintenance) core_maintenance(del_v_in_VI, rVR, domS);
        if(EXE_core_maintenance){
            if(del_v_in_VI){
                for(ui i = 0; i < rVR.size(); i++){
                    ui v = rVR[i];
                    inVR[v] = 1;
                    for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                        ui w = G0_edges[j];
                        if(inVI[w] || inVR[w]){
                            ++ degVIVR[w];
                            ++ degVIVR[v];
                        }
                    }
                }
                
                for(ui i = 0; i < domS.size(); i++){
                    ui dv = domS[i].second;
                    inVR[dv] = 1;
                    for(ui j = pstart[dv]; j < pstart[dv]+G0_x[dv]; j++){
                        if(inVI[G0_edges[j]] || inVR[G0_edges[j]]){
                            ++ degVIVR[G0_edges[j]];
                            ++ degVIVR[dv];
                        }
                    }
                }
                
                if(EXE_del_from_VR){
                    for(auto e : del_from_VR){
                        inVR[e] = 1;
                        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                            ui w = G0_edges[i];
                            if(inVR[w] || inVI[w]){
                                ++ degVIVR[w];
                                ++ degVIVR[e];
                            }
                        }
                    }
                }
                if(EXE_new2VI){
                    for(auto e : new2VI){
                        VI.pop_back();
                        inVI[e] = 0;
                        inVR[e] = 1;
                        for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                            ui w = G0_edges[i];
                            if(inVI[w]){
                                -- degVI[w];
                                -- degVI[e];
                            }
                        }
                    }
                }
                return;
            }
        }
        

        // Description :  compute ub for C + ustar 
        ui ub = compute_ub();
        if(ub > kl){

            // Description : resursive C  , R - dominating set
            
            BB_dom_ustar(level + 1);
        }
        
        // Different: mising C + ustar , R- ustar - dominating set[i]

        if(EXE_core_maintenance){
            for(ui i = 0; i < rVR.size(); i++){
                ui v = rVR[i];
                inVR[v] = 1;
                for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                    ui w = G0_edges[j];
                    if(inVI[w] || inVR[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[v];
                    }
                }
            }
        }
        
        for(ui i = 0; i < domS.size(); i++){
            ui dv = domS[i].second;
            inVR[dv] = 1;
            for(ui j = pstart[dv]; j < pstart[dv]+G0_x[dv]; j++){
                if(inVI[G0_edges[j]] || inVR[G0_edges[j]]){
                    ++ degVIVR[G0_edges[j]];
                    ++ degVIVR[dv];

                }
            }
        }

        if(EXE_del_from_VR){
            for(auto e : del_from_VR){
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVR[w] || inVI[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[e];
                    }
                }
            }
        }
        if(EXE_new2VI){
            for(auto e : new2VI){
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];
                    }
                }
            }
        }
    }
    

    // Description : Branching in case domination set is empty 
    else{
        ++ binBr;

        // Description : add ustar in VI (C) and remove from VR (R) 
        VI.push_back(ustar);
        inVI[ustar] = 1;
        inVR[ustar] = 0;

        // Description : Increament degree of neighbors of ustar in C 
        for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
            if(inVI[G0_edges[i]]){
                ++ degVI[G0_edges[i]];
                ++ degVI[ustar];
            }
        }
        

        // Description : compute ub of C+ustar, R-ustar 
        ui ub = compute_ub();
        if(ub > kl){
            
            //Description : resursive c + ustar , R-ustar
            BB_dom_ustar(level +1 );
        }
        
        
        // Description : Remove ustar from C 
        VI.pop_back();
        inVI[ustar] = 0;

        // Description : Decreament degree of neighbors of ustar in C and R 
        for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
            if(inVI[G0_edges[i]]){
                -- degVI[G0_edges[i]];
                -- degVI[ustar];
            }
            if(inVI[G0_edges[i]] || inVR[G0_edges[i]]){
                -- degVIVR[G0_edges[i]];
                -- degVIVR[ustar];
            }
        }
        
        vector<ui> rVR;
        bool del_v_in_VI = false;
        if(EXE_core_maintenance) core_maintenance(del_v_in_VI, rVR, ustar);

        if(EXE_core_maintenance){
            if(del_v_in_VI){
                for(ui i = 0; i < rVR.size(); i++){
                    ui v = rVR[i];
                    inVR[v] = 1;
                    for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                        ui w = G0_edges[j];
                        if(inVI[w] || inVR[w]){
                            ++ degVIVR[w];
                            ++ degVIVR[v];
                        }
                    }
                }
                inVR[ustar] = 1;
                for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w] || inVR[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[ustar];
                    }
                }
                
                for(auto e : del_from_VR){
                    inVR[e] = 1;
                    for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                        ui w = G0_edges[i];
                        if(inVR[w] || inVI[w]){
                            ++ degVIVR[w];
                            ++ degVIVR[e];
                        }
                    }
                }
                
                for(auto e : new2VI){
                    VI.pop_back();
                    inVI[e] = 0;
                    inVR[e] = 1;
                    for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                        ui w = G0_edges[i];
                        if(inVI[w]){
                            -- degVI[w];
                            -- degVI[e];
                        }
                    }
                }
                
                return;
            }
        }
        

        // Description : compute ub of C-ustar, R-ustar
        ub = compute_ub();
        if(ub > kl){
            
            // Description : resursive C-ustar, R-ustar
            BB_dom_ustar(level + 1);
        }
        
        if(EXE_core_maintenance){
            for(ui i = 0; i < rVR.size(); i++){
                ui v = rVR[i];
                inVR[v] = 1;
                for(ui j = pstart[v]; j < pstart[v]+G0_x[v]; j++){
                    ui w = G0_edges[j];
                    if(inVI[w] || inVR[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[v];
                    }
                }
            }
        }
        
        inVR[ustar] = 1;
        for(ui i = pstart[ustar]; i < pstart[ustar]+G0_x[ustar]; i++){
            ui w = G0_edges[i];
            if(inVI[w] || inVR[w]){
                ++ degVIVR[w];
                ++ degVIVR[ustar];
            }
        }
        
        
        if(EXE_del_from_VR){
            for(auto e : del_from_VR){
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVR[w] || inVI[w]){
                        ++ degVIVR[w];
                        ++ degVIVR[e];
                    }
                }
            }
        }
        if(EXE_new2VI){
            for(auto e : new2VI){
                VI.pop_back();
                inVI[e] = 0;
                inVR[e] = 1;
                for(ui i = pstart[e]; i < pstart[e]+G0_x[e]; i++){
                    ui w = G0_edges[i];
                    if(inVI[w]){
                        -- degVI[w];
                        -- degVI[e];
                    }
                }
            }
        }
    }
}