#include <stdio.h>
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "./inc/heuristic.h"
#include "./src/gpuMemoryAllocation.cu"
#include "./src/helpers.cc"
#include "./ipc/msgtool.h"

deviceGraphGenPointers deviceGenGraph;
deviceGraphPointers deviceGraph;
deviceInterPointers initialTask;
deviceTaskPointers deviceTask;
ui BLK_DIM2 = 1024;
ui BLK_NUM2 = 32;
ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;
ui initialPartitionSize;

size_t sharedMemorySizeinitial;
size_t  sharedMemoryUpdateNeigh;

void listen_for_messages() {

    msg_queue_server server('g'); 
    long type = 1;
    while (true) {
        if (server.recv_msg(type)) {
            char* msg = server.get_msg();
            auto recieveTime = chrono::high_resolution_clock::now();
            queryInfo query(totalQuerry,msg,recieveTime);
            totalQuerry++;
            message_queue_mutex.lock();
            message_queue.push_back(query);
            message_queue_mutex.unlock();
        }
    }
}

void process_messages() {
    while (true) {
        message_queue_mutex.lock();
        while (!message_queue.empty()) {
            vector<queryInfo> messages = message_queue;
            message_queue.clear();
            message_queue_mutex.unlock();

            for (auto& message : messages) {
                auto processTime = chrono::high_resolution_clock::now();
                message.queryProcessedTime = processTime;

                ui queryId = message.queryId;
                string queryText = message.queryString;

                istringstream iss(queryText);
                vector<ui> argValues;
                ui number;

                while (iss >> number) {
                    argValues.push_back(number);
                }
                queryData query(argValues[0],argValues[1],argValues[2],argValues[3],argValues[4]);
                queries.push_back(query);
                if (queries[queryId].isHeu) 
                CSSC_heu(queryId);
                cout<< "Processing QID : "<<queryId<<" Query Message: "<<queryText<<endl;
                cout << "Heuristic Kl " <<  queries[queryId].kl << " Ku " << queries[queryId].ku << endl;
                if (queries[queryId].kl == queries[queryId].ku) {
                  cout<< "Found Solution for  QID : "<<queryId<<" Query Message: "<<queryText<<endl;
                  cout << "heuristic find the OPT!" << endl;
                  cout << "mindeg = " << queries[queryId].kl << endl;
                  cout << "H.size = " << H.size() << endl;
                  cout << "time = " << integer_to_string(queries[queryId].timer.elapsed()).c_str() << endl;
                  continue;
                }

                cal_query_dist(queries[queryId].QID);
                chkerr(cudaMemcpy(deviceGraph.degree+(queryId*n), degree, n * sizeof(ui), cudaMemcpyHostToDevice));
                chkerr(cudaMemcpy(deviceGraph.distance+(queryId*n), q_dist, n * sizeof(ui), cudaMemcpyHostToDevice));
                chkerr(cudaMemcpy(deviceGraph.lowerBoundDegree+(queryId*n), &(queries[queryId].kl),sizeof(ui),cudaMemcpyHostToDevice));

                chkerr(cudaMemset(initialTask.globalCounter,0,sizeof(ui)));
                chkerr(cudaMemset(initialTask.entries,0,INTOTAL_WARPS*sizeof(ui)));

                if (queries[queryId].kl <= 1)
                  queries[queryId].ubD = queries[queryId].N2 - 1;
                else {
                  for (ui d = 1; d <= queries[queryId].N2; d++) {
                    if (d == 1 || d == 2) {
                      if (queries[queryId].kl + d > queries[queryId].N2) {
                        queries[queryId].ubD = d - 1;
                        break;
                      }
                    } else {
                      ui min_n = queries[queryId].kl + d + 1 + floor(d / 3) * (queries[queryId].kl - 2);
                      if (queries[queryId].N2 < min_n) {
                        queries[queryId].ubD = d - 1;
                        break;
                      }
                    }
                  }
                }

               


            }

            message_queue_mutex.lock();
        }
        message_queue_mutex.unlock();

        
    }
}


struct subtract_functor {
  const ui x;

  subtract_functor(ui _x): x(_x) {}

  __host__ __device__ ui operator()(const ui & y) const {
    return y - x;
  }
};


int main(int argc,
  const char * argv[]) {
  if (argc != 6) {
    cout << "wrong input parameters!" << endl;
    exit(1);
    exit(1);
  }

  cout << "Details" << endl;

  const char * filepath = argv[1]; // Path to the graph file. The graph should be represented as an edge list with tab (\t) separators
  ui partitionSize = atoi(argv[2]); // Defines the partition size, in number of elements, that a single warp will read from and write to.
  ui bufferSize = atoi(argv[3]); // Specifies the size, in number of elements, where warps will write in case the partition overflows.
  double copyLimit = stod(argv[4]); // Specifies that only warps with at most this percentage of their partition space filled will read from the buffer and write to their partition.
  ui readLimit = atoi(argv[5]); // Maximum number of tasks a warp with an empty partition can read from the buffer.
  cout << "File Path = " << filepath << endl;


  load_graph(filepath);
  core_decomposition_linear_list();

 

  memoryAllocationGenGraph(deviceGenGraph);
  totalQuerry = 0;
  q_dist = new ui[n];

  memeoryAllocationGraph(deviceGraph,100);

  ui BLK_DIM2 = 1024;
  ui BLK_NUM2 = 32;
  ui INTOTAL_WARPS = (BLK_NUM2 * BLK_DIM2) / 32;

  initialPartitionSize = (n / INTOTAL_WARPS) + 1;


  memoryAllocationinitialTask(initialTask, INTOTAL_WARPS, initialPartitionSize);

  memoryAllocationTask(deviceTask, TOTAL_WARPS, partitionSize, 100);

  sharedMemorySizeinitial = INTOTAL_WARPS * sizeof(ui);
  sharedMemoryUpdateNeigh = WARPS_EACH_BLK * sizeof(ui);
  
  bool *stopFlag;
  stopFlag = new bool[100];

  memset(stopFlag,true,100*sizeof(bool));
  cudaMemset(deviceTask.ustar, -1, TOTAL_WARPS * partitionSize * sizeof(int));

  bool * outOfMemoryFlag, outMemFlag;

  cudaMalloc((void ** ) & outOfMemoryFlag, sizeof(bool));
  cudaMemset(outOfMemoryFlag, 0, sizeof(bool));

  size_t sharedMemrySizeDoms = WARPS_EACH_BLK * sizeof(ui);

  ui tempHost, numReadHost, numTaskHost, startOffset, endOffset;
  numTaskHost = 0;
  numReadHost = 0;
  tempHost = 0;
  startOffset = 0;
  endOffset = 0;

  thread listener(listen_for_messages);
  thread processor(process_messages);
  listener.join();
  processor.join();
  cudaDeviceSynchronize();

  return 0;
}