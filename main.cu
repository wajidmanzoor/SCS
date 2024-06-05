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
    N1 = 6;
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

    cout << "Size after reduction = "<<h_global_counter<<endl;

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
    cudaMalloc((void**)&flag,sizeof(bool));

    int *d_output;
    int c =0;
    int *output;
    while(1){

      cudaMalloc((void**)&d_output,size_*sizeof(ui));
      cudaMemset(d_output, -1, size_ * sizeof(int));

      cout <<endl<<"size"<<size_<<endl;

     SBS<<<BLK_NUMS,BLK_DIM>>>(d_tasks_offset_new,d_tasks_new,d_status_new,d_neighbors,d_offset,d_kl,h_global_counter,N1,N2,size_, d_output,d_degree, dMAX,d_dist);


     output = new int[size_];

    cudaMemcpy(output,d_output,(size_)* sizeof(int),cudaMemcpyDeviceToHost);
    cout << endl;
     for (int i =0;i< size_ ;i++){
      cout << "CPU ustar " << output[i] << " ";
     }
     cout<<endl;

     iter ++;
     new_size_ = 1 <<iter;
    cudaMalloc((void**)&d_tasks, (new_size_*h_global_counter)*sizeof(ui));

    cudaMalloc((void**)&d_status, (new_size_*h_global_counter)*sizeof(ui));

    cudaMalloc((void**)&d_tasks_offset,(new_size_+1)* sizeof(ui));

    cudaMemset(d_global_counter,0,sizeof(ui));
    shared_memory_size = 2 * BLK_DIM * sizeof(ui);

    Branching <<<BLK_NUMS,BLK_DIM,shared_memory_size>>>(d_output, d_tasks_new, d_tasks_offset_new,d_status_new, d_global_counter,h_global_counter, d_tasks_offset,d_tasks,d_status,size_,new_size_);

    cudaFree(d_tasks_new);
    cudaFree(d_status_new);
    cudaFree(d_tasks_offset_new);

    cudaMemcpy(&h_global_counter,d_global_counter, sizeof(ui), cudaMemcpyDeviceToHost);

    cout<< " Size at Itter  = "<<iter <<" = "<<h_global_counter<<endl;

    cudaMalloc((void**)&d_tasks_new,h_global_counter*sizeof(ui));
    cudaMalloc((void**)&d_status_new,h_global_counter*sizeof(ui));
    cudaMalloc((void**)&d_tasks_offset_new,(new_size_+1)* sizeof(ui));


    cudaMemcpy(d_tasks_new,d_tasks,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_status_new,d_status,h_global_counter*sizeof(ui),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tasks_offset_new,d_tasks_offset,(new_size_+1)*sizeof(ui),cudaMemcpyDeviceToDevice);



    cudaFree(d_tasks);
    cudaFree(d_status);
    cudaFree(d_tasks_offset);
    cudaFree(d_output);

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
    size_ = new_size_;
    cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    return 0;
}