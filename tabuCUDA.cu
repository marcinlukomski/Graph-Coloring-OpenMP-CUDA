#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

__global__ void count_conflicts(int* d_current_colors, int* d_adjacency_list, int* d_offsets, int num_nodes, int* d_conflict_count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_nodes) {
        int color = d_current_colors[idx];
        int start = d_offsets[idx];
        int end = d_offsets[idx + 1];
        for (int i = start; i < end; ++i) {
            if (color == d_current_colors[d_adjacency_list[i]]) {
                atomicAdd(d_conflict_count, 1);
            }
        }
    }
}

void tabu_search_cuda(int num_nodes, int max_colors, const std::vector<std::vector<int>>& adjacency_list, bool debug = false) {
    srand(time(NULL));

    int max_iterations = 5000;    
    int tabu_list_size = 4;       
    int neighbor_reps = 700;      

    int conflict_count, new_conflict_count;
    int selected_candidate;
    int solutions_checked = 0;

    std::vector<int> tabu_list;
    std::set<int> candidate_set;
    std::vector<int> candidates;
    std::vector<int> current_colors(num_nodes);
    std::vector<int> candidate_colors(num_nodes);
    std::map<int, int> aspiration_criteria;
    std::vector<std::pair<int, int>> debug_info;

    if (max_colors > num_nodes) max_colors = num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        current_colors[i] = rand() % max_colors;
    }

    std::cout << "Initial coloring:";
    for (int i = 0; i < num_nodes; i++) {
        std::cout << " " << current_colors[i];
    }
    std::cout << std::endl;

    int* d_current_colors;
    int* d_adjacency_list;
    int* d_offsets;
    int* d_conflict_count;
    int* d_candidate_colors;

    int adjacency_list_size = 0;
    for (const auto& neighbors : adjacency_list) {
        adjacency_list_size += neighbors.size();
    }

    std::vector<int> flat_adjacency_list(adjacency_list_size);
    std::vector<int> offsets(num_nodes + 1);

    int index = 0;
    for (int i = 0; i < num_nodes; ++i) {
        offsets[i] = index;
        for (int neighbor : adjacency_list[i]) {
            flat_adjacency_list[index++] = neighbor;
        }
    }
    offsets[num_nodes] = index;

    cudaMalloc(&d_current_colors, num_nodes * sizeof(int));
    cudaMalloc(&d_adjacency_list, adjacency_list_size * sizeof(int));
    cudaMalloc(&d_offsets, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_conflict_count, sizeof(int));
    cudaMalloc(&d_candidate_colors, num_nodes * sizeof(int));

    cudaMemcpy(d_current_colors, current_colors.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacency_list, flat_adjacency_list.data(), adjacency_list_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((num_nodes + blockSize.x - 1) / blockSize.x);

    while (max_iterations > 0) {
        candidate_set.clear();
        conflict_count = 0;

        int h_conflict_count = 0;
        cudaMemcpy(d_conflict_count, &h_conflict_count, sizeof(int), cudaMemcpyHostToDevice);

        count_conflicts<<<gridSize, blockSize>>>(d_current_colors, d_adjacency_list, d_offsets, num_nodes, d_conflict_count);
        cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
        conflict_count /= 2;

        if (conflict_count == 0) break;
        solutions_checked++;

        candidates.clear();
        for (int i = 0; i < num_nodes; ++i) {
            if (std::find(tabu_list.begin(), tabu_list.end(), i) == tabu_list.end()) {
                for (int neighbor : adjacency_list[i]) {
                    if (current_colors[i] == current_colors[neighbor]) {
                        candidate_set.insert(i);
                        break;
                    }
                }
            }
        }
        std::copy(candidate_set.begin(), candidate_set.end(), std::back_inserter(candidates));

        for (int i = 0; i < neighbor_reps; i++) {
            selected_candidate = candidates[rand() % candidates.size()];
            candidate_colors = current_colors;
            candidate_colors[selected_candidate] = rand() % (max_colors - 1);

            if (candidate_colors[selected_candidate] == current_colors[selected_candidate]) {
                candidate_colors[selected_candidate] = max_colors - 1;
            }

            cudaMemcpy(d_candidate_colors, candidate_colors.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);
            h_conflict_count = 0;
            cudaMemcpy(d_conflict_count, &h_conflict_count, sizeof(int), cudaMemcpyHostToDevice);

            count_conflicts<<<gridSize, blockSize>>>(d_candidate_colors, d_adjacency_list, d_offsets, num_nodes, d_conflict_count);
            cudaMemcpy(&new_conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
            new_conflict_count /= 2;

            if (new_conflict_count < conflict_count) {
                bool in_tabu_list = false;

                if (aspiration_criteria.find(conflict_count) == aspiration_criteria.end()) {
                    aspiration_criteria.insert({conflict_count, conflict_count - 1});
                }

                if (new_conflict_count <= aspiration_criteria[conflict_count]) {
                    aspiration_criteria[conflict_count] = new_conflict_count - 1;

                    for (auto x : tabu_list) {
                        if (x == selected_candidate) {
                            tabu_list.erase(std::remove(tabu_list.begin(), tabu_list.end(), selected_candidate), tabu_list.end());
                            break;
                        }
                    }
                    break;
                } else {
                    for (auto x : tabu_list) {
                        if (x == selected_candidate) {
                            in_tabu_list = true;
                            break;
                        }
                    }
                    if (in_tabu_list) {
                        continue;
                    } else {
                        break;
                    }
                }
            }
        }

        tabu_list.push_back(selected_candidate);
        if (tabu_list.size() > tabu_list_size) {
            tabu_list.erase(tabu_list.begin());
        }
        current_colors = candidate_colors;

        cudaMemcpy(d_current_colors, current_colors.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        max_iterations--;
    }

    cudaFree(d_current_colors);
    cudaFree(d_adjacency_list);
    cudaFree(d_offsets);
    cudaFree(d_conflict_count);
    cudaFree(d_candidate_colors);

    std::cout << "Final coloring:";
    for (int i = 0; i < num_nodes; i++) {
        std::cout << " " << current_colors[i];
    }
    std::cout << std::endl;

    int total_conflicts = 0;
    for (int i = 0; i < num_nodes; i++) {
        for (auto neighbor : adjacency_list[i]) {
            if (current_colors[i] == current_colors[neighbor]) {
                debug_info.push_back(std::make_pair(neighbor, i));
                if (std::find(debug_info.begin(), debug_info.end(), std::make_pair(i, neighbor)) == debug_info.end()) {
                    std::cout << "Conflict: " << i << " " << neighbor << " - color " << current_colors[i] << std::endl;
                }
                total_conflicts++;
            }
        }
    }
    total_conflicts /= 2;
    std::cout << "Number of conflicts: " << total_conflicts << std::endl;
}

std::vector<std::vector<int>> read_graph_from_file(const std::string& filename, int& num_nodes) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file." << std::endl;
        exit(1);
    }

    infile >> num_nodes;
    std::vector<std::vector<int>> adjacency_list(num_nodes);

    int u, v;
    while (infile >> u >> v) {
        adjacency_list[u - 1].push_back(v - 1);
        adjacency_list[v - 1].push_back(u - 1);
    }

    return adjacency_list;
}

int main() {
    std::string filename = "graph.txt";
    int num_nodes;
    std::vector<std::vector<int>> adjacency_list = read_graph_from_file(filename, num_nodes);

    int max_colors = 3;

    auto start = std::chrono::high_resolution_clock::now();
    tabu_search_cuda(num_nodes, max_colors, adjacency_list);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
