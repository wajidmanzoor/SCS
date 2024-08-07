#include "../inc/ListLinearHeap.h"

// Description: Read graph from text file
void load_graph(const char *graph_file) {
  // Description: verticies and their neighbor set
  map<ui, set<ui>> G;
  string buffer;
  ifstream inputFile(graph_file, ios::in);

  // Description: Read file line by line and add to G
  if (!inputFile.is_open()) {
    cout << "Graph file Open Failed " << endl;
    exit(1);
  } else {
    inputFile >> n >> m;

    ui tu, tv;
    while (inputFile >> tu >> tv) {
      if (tu == tv) continue;
      G[tu].insert(tv);
      G[tv].insert(tu);
    }
    inputFile.close();
  }

  // Description: Vertices sorted by Core values
  peel_sequence = new ui[n];

  // Description: Degree of vertices
  degree = new ui[n];
  // Description: Core Values of vertices
  core = new ui[n];
  dMAX = 0;

  // Description: Intialize peel sequence, calculate degree using the set in G
  // and calculate dmax
  for (ui i = 0; i < n; i++) {
    peel_sequence[i] = i;
    if (G.find(i) != G.end()) {
      degree[i] = G.find(i)->second.size();
      if (degree[i] > dMAX) dMAX = degree[i];
    } else
      degree[i] = 0;
  }

  // Description: Neighbor offset
  pstart = new ui[n + 1];
  // Description: Neighbors
  edges = new ui[2 * m];

  // Description: Populate pstart and Edges using G
  pstart[0] = 0;
  for (ui i = 0; i < n; i++) {
    if (G.find(i) != G.end()) {
      ui j = 0;
      for (ui nei : G[i]) {
        edges[pstart[i] + j] = nei;
        j++;
      }
      pstart[i + 1] = pstart[i] + G[i].size();
    } else {
      pstart[i + 1] = pstart[i];
    }
  }
  cout << "n=" << n << ",m=" << m << ",dMAX=" << dMAX << endl;
}

// Description: Function that reads the file
FILE *open_file(const char *file_name, const char *mode) {
  FILE *f = fopen(file_name, mode);
  if (f == nullptr) {
    printf("Can not open file: %s\n", file_name);
    exit(1);
  }

  return f;
}

// Description: Peeling Algorithm to calcuate core values, and update the peel
// sequence
void core_decomposition_linear_list() {
  ui max_core = 0;

  // Description: Create linked lists each containing same degree vertices
  ListLinearHeap *linear_heap = new ListLinearHeap(n, n - 1);
  linear_heap->init(n, n - 1, peel_sequence, degree);

  // Description: One my one remove and least degree vertex to calculate core
  // values.
  memset(core, 0, sizeof(ui) * n);
  for (ui i = 0; i < n; i++) {
    ui u, key;

    // Description: remove least degree vertex
    linear_heap->pop_min(u, key);
    if (key > max_core) max_core = key;
    peel_sequence[i] = u;
    core[u] = max_core;

    // Description: Decrease the degree of neighbors of removed vertex, and
    // rearrange the linked lists.
    for (ui j = pstart[u]; j < pstart[u + 1]; j++)
      if (core[edges[j]] == 0) {
        linear_heap->decrement(edges[j]);
      }
  }
  delete linear_heap;
}

void cal_query_dist() {
  // Description : Intialize querry distance array with INF
  q_dist = new ui[n];
  for (ui i = 0; i < n; i++) q_dist[i] = INF;

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
    for (ui i = pstart[v]; i < pstart[v + 1]; i++) {
      ui w = edges[i];

      // Description : if distance of neighbor is INF, set to dstance of parent
      // + 1. Push neighbor to queue.
      if (q_dist[w] == INF) {
        q_dist[w] = q_dist[v] + 1;
        Q.push(w);
      }
    }
  }
}
