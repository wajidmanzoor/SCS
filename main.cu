#include <stdio.h>
#include "./src/Graph.h"
#include "./src/helpers.cc"


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

int main() {

    load_graph("data/graph.txt");

    QID = 2;
    N1 = 5;
    N2 = 9;

    core_decomposition_linear_list();

    // Description: upper bound defined
    ku = miv(core[QID], N2-1);
    kl = 0;
    ubD = N2-1;
    cal_query_dist();

    ui *d_offset,*d_neighbors,*d_degree, *d_dist;
    ui *d_kl;

    cout << " d max " << dMAX<<endl;


    cudaMalloc((void**)&d_degree, n * sizeof(ui));
    cudaMemcpy(d_degree, degree, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_offset, (n+1) * sizeof(ui));
    cudaMemcpy(d_offset, pstart, (n+1) * sizeof(ui), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&d_neighbors, (2*m) * sizeof(ui));
    cudaMemcpy(d_neighbors, edges, (2*m) * sizeof(ui), cudaMemcpyHostToDevice);

     cudaMalloc((void**)&d_dist, n * sizeof(ui));
    cudaMemcpy(d_dist, q_dist, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_kl, sizeof(ui));
    cudaMemcpy(d_kl, &kl,sizeof(ui),cudaMemcpyHostToDevice);

    ui *d_tasks, *d_status,*d_tasks_offset, *d_global_counter;

    int iter = 0;
    int size_,new_size_;
    size_ = 1 <<iter;

    cudaMalloc((void**)&d_tasks, (size_*n)*sizeof(ui));

    cudaMalloc((void**)&d_status, (size_*n)*sizeof(ui));

    cudaMalloc((void**)&d_global_counter, sizeof(ui));

    ui h_global_counter = 0;
    cudaMemcpy(d_global_counter, &h_global_counter, sizeof(ui), cudaMemcpyHostToDevice);

    int shared_memory_size = 2 * BLK_DIM * sizeof(ui);

    IntialReductionRules<<<BLK_NUMS,BLK_DIM,shared_memory_size>>>(d_offset,d_neighbors,d_degree,d_dist,QID,n ,d_tasks, d_status, N2, d_global_counter);

    cudaMemcpy(&h_global_counter,d_global_counter, sizeof(ui), cudaMemcpyDeviceToHost);

    //cout << "Size after reduction = "<<h_global_counter<<endl;

    ui *tasks_offset;
    tasks_offset = new ui[size_+1];

    ui *d_tasks_new,*d_status_new, *d_tasks_offset_new;
    cudaMalloc((void**)&d_tasks_new,h_global_counter*sizeof(ui));
    cudaMalloc((void**)&d_status_new,h_global_counter*sizeof(ui));



    cudaMemcpy(d_tasks_new,d_tasks,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_status_new,d_status,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaFree(d_tasks);
    cudaFree(d_status);

    tasks_offset[0]=0;
    tasks_offset[1]= h_global_counter;

    cudaMalloc((void**)&d_tasks_offset_new,(size_+1)* sizeof(ui));
    cudaMemcpy(d_tasks_offset_new, tasks_offset, (size_+1)* sizeof(ui), cudaMemcpyHostToDevice);

    bool *flag;
    bool *processed_flag;
    cudaMalloc((void**)&flag,sizeof(bool));

    bool stop_flag;


    int *d_output;
    int *output;
    int start,end;
    while(1){

      cudaMalloc((void**)&d_output,size_*sizeof(ui));
      cudaMemset(d_output, -1, size_ * sizeof(int));
      cudaMemset(flag,1,sizeof(bool));

     cout <<endl<<" size "<< size_<<endl;

     SBS<<<BLK_NUMS,BLK_DIM>>>(d_tasks_offset_new,d_tasks_new,d_status_new,d_neighbors,d_offset,d_kl,h_global_counter,N1,N2,size_, d_output,d_degree, dMAX,d_dist,flag);

    cudaMemcpy(&stop_flag,flag,sizeof(bool),cudaMemcpyDeviceToHost);
    cudaMemcpy(&kl,d_kl,sizeof(ui),cudaMemcpyDeviceToHost);

    cout<< " Max Min degree  "<<kl<<endl;
    if(stop_flag){
      cout << "Final MAx Min degree "<<kl<<endl;
      break;
    }

     output = new int[size_];

    cudaMemcpy(output,d_output,(size_)* sizeof(int),cudaMemcpyDeviceToHost);
    cout << endl;
     for (int i =0;i< size_ ;i++){
      cout << " CPU ustar " << output[i] << " ";
     }
     cout<<endl;

     iter ++;
     new_size_ = 1 <<iter;



    cudaMalloc((void**)&d_tasks, (new_size_*h_global_counter)*sizeof(ui));

    cudaMalloc((void**)&d_status, (new_size_*h_global_counter)*sizeof(ui));

    cudaMalloc((void**)&d_tasks_offset,(new_size_+1)* sizeof(ui));




    shared_memory_size = BLK_DIM * sizeof(ui);
    BranchingFixed <<<BLK_NUMS,BLK_DIM,shared_memory_size>>>(d_output, d_tasks_new, d_tasks_offset_new,d_status_new,h_global_counter, d_tasks_offset,d_tasks,d_status,size_);



    cudaFree(d_tasks_new);
    cudaFree(d_status_new);
    cudaFree(d_tasks_offset_new);



    cudaMemcpy(&h_global_counter,d_global_counter, sizeof(ui), cudaMemcpyDeviceToHost);
    cudaMalloc((void**)&d_tasks_new,(new_size_*h_global_counter)*sizeof(ui));
    cudaMalloc((void**)&d_status_new,(new_size_*h_global_counter)*sizeof(ui));
    cudaMalloc((void**)&d_tasks_offset_new,(new_size_+1)* sizeof(ui));


    cudaMemcpy(d_tasks_new,d_tasks,(new_size_*h_global_counter)*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_status_new,d_status,(new_size_*h_global_counter)*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tasks_offset_new,d_tasks_offset,(new_size_+1)*sizeof(ui),cudaMemcpyDeviceToDevice);

    cudaFree(d_tasks);
    cudaFree(d_status);
    cudaFree(d_tasks_offset);
    cudaFree(d_output);
    cudaFree(processed_flag);

    ui *temp,*temp1,*temp2;
    temp = new ui[new_size_+1];
    temp1 = new ui[new_size_*h_global_counter];
    temp2 = new ui [new_size_*h_global_counter];

    cudaMemcpy(temp1,d_tasks_new,(new_size_*h_global_counter)*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(temp2,d_status_new,(new_size_*h_global_counter)*sizeof(ui),cudaMemcpyDeviceToHost);


    cudaMemcpy(temp,d_tasks_offset_new,(new_size_+1)*sizeof(ui),cudaMemcpyDeviceToHost);
    cout<<endl;



    for(ui i =0;i < (new_size_);i++){

      if(i % 2 ==0 ){
        start = i * h_global_counter;
      }else{
        start = temp[i];

      }
      end =temp[i+1];

      cout<<endl<<" task "<< i << " Offset "<<start<<" end "<<end<<endl;
      cout << "vertex in task "<<endl;
      for(ui j = start; j < end;j++){
        cout << temp1[j] << " ";
      }

      cout << endl<<"status in task"<<endl;

      for(ui j = start; j < end;j++){
        cout << temp2[j] << " ";
      }

    }
    size_ = new_size_;
    cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    return 0;
}




/*int main() {

    // Load graph 

    load_graph("data/graph.txt");

    // Seting Query Vertex, lower size limit and upper size limit
    // TODO : will update so can be passed as Arguement.
    QID = 2;
    N1 = 6;
    N2 = 9;

    // Calculate the core values of each vertex 
    core_decomposition_linear_list();

    // Description: upper bound defined
    ku = miv(core[QID], N2-1);
    kl = 0;
    ubD = N2-1;

    // Calculate distance from query vertex
    cal_query_dist();

    // Declare device pointers  to store graph information 
    ui *d_offset,*d_neighbors,*d_degree, *d_dist;
    ui *d_kl;

    cout << " d max " << dMAX<<endl;

    // Allocate memory (size will be equal to n assuming no vertex will be pruned)and copy from HOST (CPU) to Device (GPU)
    cudaMalloc((void**)&d_degree, n * sizeof(ui));
    cudaMemcpy(d_degree, degree, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_offset, (n+1) * sizeof(ui));
    cudaMemcpy(d_offset, pstart, (n+1) * sizeof(ui), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&d_neighbors, (2*m) * sizeof(ui));
    cudaMemcpy(d_neighbors, edges, (2*m) * sizeof(ui), cudaMemcpyHostToDevice);

     cudaMalloc((void**)&d_dist, n * sizeof(ui));
    cudaMemcpy(d_dist, q_dist, n * sizeof(ui), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_kl, sizeof(ui));
    cudaMemcpy(d_kl, &kl,sizeof(ui),cudaMemcpyHostToDevice);


    // Declare device pointers to store task data. 
    ui *d_tasks, *d_status,*d_tasks_offset, *d_global_counter;


    // Level of the tree 
    int iter = 0;

    // number of current task and number of new task
    int size_,new_size_;

    // current task size = 2**(iter)
    size_ = 1 <<iter;

    // Allocate memory in device. 
    cudaMalloc((void**)&d_tasks, (size_*n)*sizeof(ui));

    cudaMalloc((void**)&d_status, (size_*n)*sizeof(ui));

    cudaMalloc((void**)&d_global_counter, sizeof(ui));

    // Intilaize global counter in HOST.

    ui h_global_counter = 0;

    // Set device global counter to zero. 
    cudaMemcpy(d_global_counter, &h_global_counter, sizeof(ui), cudaMemcpyHostToDevice);


    // Shared memory needed for kernel call. 
    int shared_memory_size = 2 * BLK_DIM * sizeof(ui);

    // Call Intial reduction rules, returns Tasks and status as pointer d_tasks, d_status
    IntialReductionRules<<<BLK_NUMS,BLK_DIM,shared_memory_size>>>(d_offset,d_neighbors,d_degree,d_dist,QID,n ,d_tasks, d_status, N2, d_global_counter);


    // Copy global counter (i.e Number of elements in d_task ) to host 
    cudaMemcpy(&h_global_counter,d_global_counter, sizeof(ui), cudaMemcpyDeviceToHost);

    cout << "Size after reduction = "<<h_global_counter<<endl;

    // Create task offset in HOST 
    ui *tasks_offset;
    tasks_offset = new ui[size_+1];

    // Declare device pointer to store new task data. (size here will be equal to global counter)
    ui *d_tasks_new,*d_status_new, *d_tasks_offset_new;
    cudaMalloc((void**)&d_tasks_new,h_global_counter*sizeof(ui));
    cudaMalloc((void**)&d_status_new,h_global_counter*sizeof(ui));


    // Copy data from task to new task 
    cudaMemcpy(d_tasks_new,d_tasks,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_status_new,d_status,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);

    // Free old task pointers
    cudaFree(d_tasks);
    cudaFree(d_status);

    // Intialize offset to [0,global counter]. (global counter is equal to number of elements in task)
    tasks_offset[0]=0;
    tasks_offset[1]= h_global_counter;

    // allocated memory (size is 2**(iter) +1 )for offset.  
    cudaMalloc((void**)&d_tasks_offset_new,(size_+1)* sizeof(ui));
    cudaMemcpy(d_tasks_offset_new, tasks_offset, (size_+1)* sizeof(ui), cudaMemcpyHostToDevice);


    // Flag for stoping condition, not yet used
    // TODO : will used to add stoping condition in while loop
    bool *flag;
    cudaMalloc((void**)&flag,sizeof(bool));

    // device Pointer to store the ustar of each task 
    int *d_output;

    // Variable used for debuging. will remove later. 
    int c =0;
    // Host pointer used to store ustar (for debuging) will remove later. 
    int *output;

    while(1){

      // Allocate memory for ustar 
      cudaMalloc((void**)&d_output,size_*sizeof(ui));
      cudaMemset(d_output, -1, size_ * sizeof(int));

      cout <<endl<<"size"<<size_<<endl;

    
    // Kernel that applying pruning rules, does comparison for min degree and calculates ustar. 
     SBS<<<BLK_NUMS,BLK_DIM>>>(d_tasks_offset_new,d_tasks_new,d_status_new,d_neighbors,d_offset,d_kl,h_global_counter,N1,N2,size_, d_output,d_degree, dMAX,d_dist);


    // host array to store ustar (used to debuging will remove latter)
     output = new int[size_];

    // copy ustar from device to host 
    cudaMemcpy(output,d_output,(size_)* sizeof(int),cudaMemcpyDeviceToHost);
    cout << endl;
     for (int i =0;i< size_ ;i++){
      cout << "CPU ustar " << output[i] << " ";
     }
     cout<<endl;


    // Increament itter
     iter ++;

     // Calculate number of new tasks  
     new_size_ = 1 <<iter;
    
    // Allocated memory for new task pointers (size will be number of new tasks times h_global counter )
    cudaMalloc((void**)&d_tasks, (new_size_*h_global_counter)*sizeof(ui));

    cudaMalloc((void**)&d_status, (new_size_*h_global_counter)*sizeof(ui));

    cudaMalloc((void**)&d_tasks_offset,(new_size_+1)* sizeof(ui));

    // Set global counter to zero 

    cudaMemset(d_global_counter,0,sizeof(ui));
    shared_memory_size = 2 * BLK_DIM * sizeof(ui);
    
    // Kernel creates new tasks using current task and ustar
    Branching <<<BLK_NUMS,BLK_DIM,shared_memory_size>>>(d_output, d_tasks_new, d_tasks_offset_new,d_status_new, d_global_counter,h_global_counter, d_tasks_offset,d_tasks,d_status,size_,new_size_);

    // Free current tasks used in this iter
    cudaFree(d_tasks_new);
    cudaFree(d_status_new);
    cudaFree(d_tasks_offset_new);

    // Copy global counter to host 
    cudaMemcpy(&h_global_counter,d_global_counter, sizeof(ui), cudaMemcpyDeviceToHost);

    cout<< " Size at Itter  = "<<iter <<" = "<<h_global_counter<<endl;

    // Allocate memory for new tasks based on global counter
    cudaMalloc((void**)&d_tasks_new,h_global_counter*sizeof(ui));
    cudaMalloc((void**)&d_status_new,h_global_counter*sizeof(ui));
    cudaMalloc((void**)&d_tasks_offset_new,(new_size_+1)* sizeof(ui));

    // Copy data to new tasks 
    cudaMemcpy(d_tasks_new,d_tasks,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_status_new,d_status,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tasks_offset_new,d_tasks_offset,(new_size_+1)*sizeof(ui),cudaMemcpyDeviceToDevice);


    
    cudaFree(d_tasks);
    cudaFree(d_status);
    cudaFree(d_tasks_offset);
    cudaFree(d_output);


    // Used for debug purpose
    ui *temp,*temp1,*temp2;
    temp = new ui[new_size_+1];
    temp1 = new ui[h_global_counter];
    temp2 = new ui [h_global_counter];
 
    cudaMemcpy(temp1,d_tasks_new,h_global_counter*sizeof(ui),cudaMemcpyDeviceToHost);
    cudaMemcpy(temp2,d_status_new,h_global_counter*sizeof(ui),cudaMemcpyDeviceToHost);


    cudaMemcpy(temp,d_tasks_offset_new,(new_size_+1)*sizeof(ui),cudaMemcpyDeviceToHost);

    for(ui i =0;i < (new_size_);i++){
      cout<<endl<<" task "<< i << " Offset "<<temp[i]<<endl;
      cout << "vertex in task "<<endl;
      for(ui j = temp[i]; j < temp[i+1];j++){
        cout << temp1[j] << " ";
      }

      cout << endl<<"status in task"<<endl;

      for(ui j = temp[i]; j < temp[i+1];j++){
        cout << temp2[j] << " ";
      }

    }
    cout <<"Vertex after branching "<<endl;
    for(ui i =0;i<h_global_counter;i++){

      cout << "  " <<temp1[i]<<"  ";
    }
    cout<<endl;


    c++;

    if(c==2){
         break;
    }
    // Debug ended 


    // Set size equal to new size 
    size_ = new_size_;
    cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    return 0;
}*/