#include <mpi.h>

#include <climits>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include "SBS.h"

struct QuerySpec {
    ui n1;
    ui n2;
    ui qid;
    ui index;
};

struct QueryResult {
    ui index;
    ui n1;
    ui n2;
    ui qid;
    ui degree;
    ui solution_size;
    ui overtime;
    ui heuristic_optimal;
    ui worker_rank;
    ui shard_id;
    ui shard_count;
    ui split_depth;
    std::uint64_t elapsed_us;
};

static_assert(std::is_trivially_copyable<QuerySpec>::value,
              "QuerySpec must be safe to broadcast as bytes");
static_assert(std::is_trivially_copyable<QueryResult>::value,
              "QueryResult must be safe to gather as bytes");

void print_usage(const char *program) {
    if (program == nullptr) program = "cpu_hybrid";
    std::cerr
        << "Usage:\n  mpirun -np <1 + queries*workers> " << program
        << " <graph> <queries.txt> <output.csv>"
        << " <heu2> <heu3> <heu4>"
        << " <ub1> <ub2> <ub3> <ub3opt>"
        << " <coremaint> <new2VI> <delVR>"
        << " <dom> <domThreshold> <MaxTimeSeconds> <searchOrder> <verbose>"
        << " <workersPerQuery> <splitDepth>\n\n"
        << "Rank 0 coordinates. Remaining ranks form equal per-query shard "
           "groups. This target requires dom=0 for exact search.\n";
}

bool read_queries(const char *path, std::vector<QuerySpec> &queries,
                  std::string &error) {
    std::ifstream input(path);
    if (!input.is_open()) {
        error = std::string("Cannot open query file: ") + path;
        return false;
    }

    std::string line;
    ui line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string::size_type first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        if (line.compare(first, 11, "server_exit") == 0) break;

        unsigned long long n1_value = 0;
        unsigned long long n2_value = 0;
        unsigned long long qid_value = 0;
        std::istringstream fields(line.substr(first));
        if (!(fields >> n1_value >> n2_value >> qid_value)) {
            std::ostringstream message;
            message << "Invalid query at " << path << ':' << line_number
                    << " (expected: N1 N2 QID; extra columns are allowed)";
            error = message.str();
            return false;
        }
        if (n1_value > UINT_MAX || n2_value > UINT_MAX ||
            qid_value > UINT_MAX) {
            std::ostringstream message;
            message << "Query value exceeds 32-bit range at " << path << ':'
                    << line_number;
            error = message.str();
            return false;
        }

        QuerySpec query;
        query.n1 = static_cast<ui>(n1_value);
        query.n2 = static_cast<ui>(n2_value);
        query.qid = static_cast<ui>(qid_value);
        query.index = static_cast<ui>(queries.size());
        queries.push_back(query);
    }

    if (queries.empty()) {
        error = std::string("No queries found in: ") + path;
        return false;
    }
    return true;
}

void cleanup_query_memory() {
    delete[] q_dist;
    q_dist = nullptr;
    delete[] G0_edges;
    G0_edges = nullptr;
    delete[] G0_x;
    G0_x = nullptr;
    delete[] G0_deg;
    G0_deg = nullptr;
    delete[] inVI;
    inVI = nullptr;
    delete[] inVR;
    inVR = nullptr;
    delete[] degVI;
    degVI = nullptr;
    delete[] degVIVR;
    degVIVR = nullptr;
    delete[] inNEI;
    inNEI = nullptr;
    delete[] NEI_score;
    NEI_score = nullptr;

    H.clear();
    G0.clear();
    VI.clear();
    VIVR.clear();
    NEI.clear();
    combs.clear();
}

void cleanup_graph_memory() {
    delete[] peel_sequence;
    peel_sequence = nullptr;
    delete[] degree;
    degree = nullptr;
    delete[] core;
    core = nullptr;
    delete[] pstart;
    pstart = nullptr;
    delete[] edges;
    edges = nullptr;
}

void reset_query_counters() {
    domBr = 0;
    binBr = 0;
    total_val_ub1 = 0.0;
    total_val_ub3 = 0.0;
    total_UB = 0.0;
    total_Heu_time = 0.0;
    time_new2VI = 0.0;
    time_del_from_VR = 0.0;
    time_find_NEI = 0.0;
    time_find_usatr = 0.0;
    time_comp_ub = 0.0;
    kl = 0;
    ku = 0;
    ubD = INF;
    over_time_flag = false;
}

QueryResult run_query(const QuerySpec &query, int rank, ui shard_id,
                      ui shard_count, ui split_depth) {
    cleanup_query_memory();
    reset_query_counters();


    bb_shard_id = shard_id;
    bb_shard_count = shard_count;
    bb_shard_split_depth = split_depth;
    N1 = query.n1;
    N2 = query.n2;
    QID = static_cast<int>(query.qid);

    Timer query_timer;
    StartTime = static_cast<double>(clock()) / CLOCKS_PER_SEC;

    ku = miv(core[QID], N2 - 1);

    Timer heuristic_timer;
    CSSC_heu();
    total_Heu_time = heuristic_timer.elapsed();

    bool heuristic_optimal = (kl == ku);
    if (!heuristic_optimal) {
        ubD = 0;
        if (kl <= 1) {
            ubD = N2 - 1;
        } else {
            for (ui d = 1; d <= N2; ++d) {
                if (d == 1 || d == 2) {
                    if (kl + d > N2) {
                        ubD = d - 1;
                        break;
                    }
                } else {
                    const ui min_n = kl + d + 1 + (d / 3) * (kl - 2);
                    if (N2 < min_n) {
                        ubD = d - 1;
                        break;
                    }
                }
            }
        }

        cal_query_dist();
        reduction_g();

        VI.clear();
        VIVR.clear();

        inVI = new bool[n]();
        inVR = new bool[n]();
        degVI = new ui[n]();
        degVIVR = new ui[n]();
        inNEI = new ui[n]();
        NEI_score = new double[n]();

        for (std::vector<ui>::const_iterator it = G0.begin(); it != G0.end();
             ++it) {
            const ui vertex = *it;
            VIVR.push_back(vertex);
            inVR[vertex] = true;
            degVIVR[vertex] = G0_deg[vertex];
        }

        VI.push_back(query.qid);
        inVI[query.qid] = true;
        inVR[query.qid] = false;

        if (EXE_dom_ustar) {
            BB_dom_ustar(1);
        } else {
            BB(1);
        }
    }

    QueryResult result;
    result.index = query.index;
    result.n1 = query.n1;
    result.n2 = query.n2;
    result.qid = query.qid;
    result.degree = kl;
    result.solution_size = static_cast<ui>(H.size());
    result.overtime = over_time_flag ? 1U : 0U;
    result.heuristic_optimal = heuristic_optimal ? 1U : 0U;
    result.worker_rank = static_cast<ui>(rank);
    result.shard_id = shard_id;
    result.shard_count = shard_count;
    result.split_depth = split_depth;
    result.elapsed_us = static_cast<std::uint64_t>(query_timer.elapsed());

    cleanup_query_memory();
    return result;
}

int parse_flag(const char *value) { return std::atoi(value); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 21) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    EXE_heu2 = parse_flag(argv[4]);
    EXE_heu3 = parse_flag(argv[5]);
    EXE_heu4 = parse_flag(argv[6]);
    EXE_ub1 = parse_flag(argv[7]);
    EXE_ub2 = parse_flag(argv[8]);
    EXE_ub3 = parse_flag(argv[9]);
    EXE_ub3_optimization = parse_flag(argv[10]);
    EXE_core_maintenance = parse_flag(argv[11]);
    EXE_new2VI = parse_flag(argv[12]);
    EXE_del_from_VR = parse_flag(argv[13]);
    EXE_dom_ustar = parse_flag(argv[14]);
    domS_Threshold = static_cast<ui>(parse_flag(argv[15]));
    MaxTime = parse_flag(argv[16]);
    srch_ord = static_cast<ui>(parse_flag(argv[17]));
    const int workers_value = parse_flag(argv[19]);
    const int split_depth_value = parse_flag(argv[20]);
    const ui workers_per_query = static_cast<ui>(workers_value);
    verbose = static_cast<ui>(parse_flag(argv[18]));

    if (EXE_dom_ustar) {
        if (rank == 0) {
            std::cerr << "cpu_hybrid requires dom=0 for exact search.\n";
        }
        MPI_Finalize();
        return 1;
    }
    if (MaxTime <= 0 || workers_value <= 0 || split_depth_value <= 0 ||
        split_depth_value > 63) {
        if (rank == 0) {
            std::cerr << "MaxTimeSeconds and workersPerQuery must be positive; "
                         "splitDepth must be in [1,63].\n";
        }
        MPI_Finalize();
        return 1;
    }
    const ui split_depth = static_cast<ui>(split_depth_value);

    std::vector<QuerySpec> queries;
    std::string query_error;
    int query_status = 1;
    if (rank == 0) {
        query_status = read_queries(argv[2], queries, query_error) ? 1 : 0;
        if (!query_status) std::cerr << query_error << '\n';
    }
    MPI_Bcast(&query_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!query_status) {
        MPI_Finalize();
        return 1;
    }

    ui query_count = rank == 0 ? static_cast<ui>(queries.size()) : 0;
    MPI_Bcast(&query_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank != 0) queries.resize(query_count);
    MPI_Bcast(queries.data(), static_cast<int>(queries.size() * sizeof(QuerySpec)),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    const std::uint64_t expected_ranks =
        1ULL + static_cast<std::uint64_t>(query_count) * workers_per_query;
    if (expected_ranks != static_cast<std::uint64_t>(world_size)) {
        if (rank == 0) {
            std::cerr << "Expected " << expected_ranks << " MPI ranks for "
                      << query_count << " queries x " << workers_per_query
                      << " workers plus one coordinator; got " << world_size
                      << ".\n";
        }
        MPI_Finalize();
        return 1;
    }
    if ((1ULL << split_depth) < workers_per_query) {
        if (rank == 0) {
            std::cerr << "splitDepth creates fewer prefixes than workers.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const double total_start = MPI_Wtime();
    if (rank > 0) {
        load_graph(argv[1]);
        core_decomposition_linear_list();
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 1, MPI_COMM_WORLD);

    int local_validation_ok = 1;
    for (std::vector<QuerySpec>::const_iterator it = queries.begin();
         it != queries.end(); ++it) {
        if (it->n1 == 0 || it->n2 == 0 || it->n1 > it->n2 || it->n2 > n ||
            it->qid >= n || it->qid > static_cast<ui>(INT_MAX)) {
            if (rank == 0) {
                std::cerr << "Invalid query " << (it->index + 1) << ": N1="
                          << it->n1 << ", N2=" << it->n2
                          << ", QID=" << it->qid << ", graph vertices=" << n
                          << '\n';
            }
            local_validation_ok = 0;
            break;
        }
    }
    int all_validation_ok = 0;
    MPI_Allreduce(&local_validation_ok, &all_validation_ok, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!all_validation_ok) {
        cleanup_graph_memory();
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "Coordinator loaded " << query_count << " queries; "
                  << workers_per_query << " cores/query, split depth "
                  << split_depth << ", " << world_size
                  << " total MPI ranks, dom=0, timeout=" << MaxTime
                  << " seconds/shard\n";
    }

    QueryResult local_result = {};
    local_result.index = UINT_MAX;

    MPI_Barrier(MPI_COMM_WORLD);
    const double search_start = MPI_Wtime();
    if (rank > 0) {
        const ui worker_index = static_cast<ui>(rank - 1);
        const ui query_index = worker_index / workers_per_query;
        const ui shard_id = worker_index % workers_per_query;
        if (verbose == 0) {
            std::cout << "rank " << rank << " starting query "
                      << (query_index + 1) << '/' << query_count << " shard "
                      << (shard_id + 1) << '/' << workers_per_query << " (N1="
                      << queries[query_index].n1 << ", N2="
                      << queries[query_index].n2 << ", QID="
                      << queries[query_index].qid << ")\n";
        }
        local_result = run_query(queries[query_index], rank, shard_id,
                                 workers_per_query, split_depth);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double search_wall_seconds = MPI_Wtime() - search_start;

    std::vector<QueryResult> gathered_results;
    if (rank == 0) {
        gathered_results.resize(static_cast<std::size_t>(world_size));
    }
    MPI_Gather(&local_result, static_cast<int>(sizeof(QueryResult)), MPI_BYTE,
               rank == 0 ? gathered_results.data() : nullptr,
               static_cast<int>(sizeof(QueryResult)), MPI_BYTE, 0, MPI_COMM_WORLD);

    int output_ok = 1;
    if (rank == 0) {
        const double total_wall_seconds = MPI_Wtime() - total_start;
        std::vector<QueryResult> ordered_results(query_count);
        std::vector<ui> found(query_count, 0);
        for (std::vector<QueryResult>::const_iterator it = gathered_results.begin();
             it != gathered_results.end(); ++it) {
            if (it->index < query_count) {
                QueryResult &combined = ordered_results[it->index];
                if (found[it->index] == 0) {
                    combined = *it;
                } else {
                    if (it->degree > combined.degree) {
                        combined.degree = it->degree;
                        combined.solution_size = it->solution_size;
                        combined.worker_rank = it->worker_rank;
                        combined.shard_id = it->shard_id;
                    }
                    combined.overtime |= it->overtime;
                    combined.heuristic_optimal |= it->heuristic_optimal;
                    if (it->elapsed_us > combined.elapsed_us)
                        combined.elapsed_us = it->elapsed_us;
                }
                ++found[it->index];
            }
        }
        for (ui i = 0; i < query_count; ++i) {
            if (found[i] != workers_per_query) {
                std::cerr << "Expected " << workers_per_query
                          << " shard results for query " << (i + 1) << "; got "
                          << found[i] << '\n';
                output_ok = 0;
            }
        }

        std::ofstream output(argv[3]);
        if (!output.is_open()) {
            std::cerr << "Cannot create output CSV: " << argv[3] << '\n';
            output_ok = 0;
        } else if (output_ok) {
            const std::uint64_t search_wall_us = static_cast<std::uint64_t>(
                search_wall_seconds * 1000000.0);
            const std::uint64_t total_wall_us = static_cast<std::uint64_t>(
                total_wall_seconds * 1000000.0);
            output
                << "query_index,graph,n1,n2,qid,query_limit,total_queries,"
                   "time_us,degree,overtime,heuristic_optimal,solution_size,"
                   "best_worker_rank,best_shard_id,workers_per_query,split_depth,"
                   "batch_search_wall_us,batch_total_wall_us\n";
            for (ui i = 0; i < query_count; ++i) {
                const QueryResult &result = ordered_results[i];
                output << (i + 1) << ",\"" << argv[1] << "\"," << result.n1
                       << ',' << result.n2 << ',' << result.qid << ','
                       << query_count << ',' << query_count << ','
                       << result.elapsed_us << ',' << result.degree << ','
                       << result.overtime << ',' << result.heuristic_optimal << ','
                       << result.solution_size << ',' << result.worker_rank << ','
                       << result.shard_id << ',' << result.shard_count << ','
                       << result.split_depth << ','
                       << search_wall_us << ',' << total_wall_us << '\n';
            }

            std::ofstream shard_output(std::string(argv[3]) + ".shards.csv");
            if (!shard_output.is_open()) {
                std::cerr << "Cannot create per-shard CSV.\n";
                output_ok = 0;
            } else {
                shard_output
                    << "query_index,qid,shard_id,worker_rank,time_us,degree,"
                       "overtime,heuristic_optimal\n";
                for (std::vector<QueryResult>::const_iterator it =
                         gathered_results.begin(); it != gathered_results.end(); ++it) {
                    if (it->index >= query_count) continue;
                    shard_output << (it->index + 1) << ',' << it->qid << ','
                                 << it->shard_id << ',' << it->worker_rank << ','
                                 << it->elapsed_us << ',' << it->degree << ','
                                 << it->overtime << ',' << it->heuristic_optimal << '\n';
                }
            }
            std::cout << "Wrote " << query_count << " results to " << argv[3]
                      << "\nSearch wall time: " << std::fixed
                      << std::setprecision(6) << search_wall_seconds
                      << " s; total including graph load/core: "
                      << total_wall_seconds << " s\n";
        }
    }

    MPI_Bcast(&output_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cleanup_query_memory();
    cleanup_graph_memory();
    MPI_Finalize();
    return output_ok ? 0 : 1;
}
