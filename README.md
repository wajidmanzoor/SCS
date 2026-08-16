# size-bounded community search (SCS)

## Problem Statement

Given:

1. graph $G = (V,E)$
2. query $q \in V$
3. size constraint $[l,h]$

Find subgraph $H$ of $G$ that satisfies the bellow conditions

1. Connected: $H$ is connected and contains $q$
2. Size bound : $l < |V(H)| < h $.
3. Minimum degree of H is maximum among all sub graphs that satisfy above conditions

# Execution Guide

## 1. Compilation

### Server Side (in `./SCS` folder)

If you are running on an **NVIDIA A100 (80 GB)**, compile using:

```bash
nvcc main.cu -o SCS -std=c++14 -lpthread -ccbin=mpic++ -lmpi \
    -arch=sm_80 -gencode=arch=compute_80,code=sm_80
```

> 🔹 If using a different GPU, update the `-arch` flag accordingly.

### Client Side (in `./SCS/Client` folder)

For continuous query sending, compile with:

```bash
g++ client.cpp -o client
```

For batch query processing (using a `.txt` file), compile with:

```bash
g++ batchClient.cpp -o batchClient
```

---

## 2. Input Data Format

The input graph must be provided as an **edge list** in `.txt` format:

- **First line**:
  ```
  <#Vertices> <#Edges>
  ```
- **Subsequent lines**:
  ```
  <u> <v>
  ```
  (each line represents an undirected edge, space-separated)

---

## 3. Running the Server

Example run command:

```bash
./SCS data/ego-facebook.txt 30000 100000 0.5 1 8 100 1 1 1 1 1
```

---

## 4. Query Format

Queries should follow the format:

```
<l> <h> <qid> <heuristic_flag> <limit_dominating_set>
```

- `<l>` = lower bound
- `<h>` = upper bound
- `<qid>` = query ID
- `<heuristic_flag>` = whether to run heuristic (0/1)
- `<limit_dominating_set>` = cap for dominating set

To terminate the server, issue:

```
server_exit
```

---

## 5. Running the Client

### Continuous Query Mode

```bash
./client
```

### Batch Processing Mode

Provide a file containing queries in the same format:
Example

```bash
./client client/batch_in.txt
```
