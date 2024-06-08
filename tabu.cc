#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>

void tabu_search(int num_nodes, int max_colors, const std::vector<std::vector<int>>& adjacency_list, bool debug = false) {
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
    std::vector<int> current_colors;
    std::vector<int> candidate_colors;
    std::map<int, int> aspiration_criteria;
    std::vector<std::pair<int, int>> debug_info;

    if (max_colors > num_nodes) max_colors = num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        current_colors.push_back(rand() % max_colors);
    }
    /*
    std::cout << "Initial coloring:";
    for (int i = 0; i < num_nodes; i++) {
        std::cout << " " << current_colors[i];
    }
    std::cout << std::endl;
    */

    while (max_iterations > 0) {
        candidate_set.clear();
        conflict_count = 0;

        for (int i = 0; i < num_nodes; i++) {
            for (auto neighbor : adjacency_list[i]) {
                if (current_colors[i] == current_colors[neighbor]) {
                    candidate_set.insert(neighbor);
                    conflict_count++;
                }
            }
        }
        conflict_count /= 2;

        if (conflict_count == 0) break;
        solutions_checked++;

        candidates.clear();
        std::copy(candidate_set.begin(), candidate_set.end(), std::back_inserter(candidates));

        for (int i = 0; i < neighbor_reps; i++) {
            selected_candidate = candidates[rand() % candidates.size()];
            candidate_colors = current_colors;
            candidate_colors[selected_candidate] = rand() % (max_colors - 1);

            if (candidate_colors[selected_candidate] == current_colors[selected_candidate]) {
                candidate_colors[selected_candidate] = max_colors - 1;
            }

            new_conflict_count = 0;
            for (int i = 0; i < num_nodes; i++) {
                for (auto neighbor : adjacency_list[i]) {
                    if (candidate_colors[i] == candidate_colors[neighbor]) {
                        new_conflict_count++;
                    }
                }
            }
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

        max_iterations--;
    }

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

    int max_colors = 100;

    auto start = std::chrono::high_resolution_clock::now();
    tabu_search(num_nodes, max_colors, adjacency_list);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
