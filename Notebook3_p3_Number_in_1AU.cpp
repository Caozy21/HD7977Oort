#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>

class SegmentTree {
public:
    SegmentTree(int n) : n(n), tree(4 * n, 0), lazy(4 * n, 0) {}

    void update(int l, int r, int ql, int qr, int node, int val) {
        if (ql <= l && r <= qr) {
            tree[node] += val;
            lazy[node] += val;
        } else if (r < ql || qr < l) {
            return;
        } else {
            int mid = (l + r) / 2;
            push_down(node);
            update(l, mid, ql, qr, 2 * node + 1, val);
            update(mid + 1, r, ql, qr, 2 * node + 2, val);
            tree[node] = std::max(tree[2 * node + 1], tree[2 * node + 2]);
        }
    }

    int query(int l, int r, int pos, int node) {
        if (l == r) {
            return tree[node];
        }
        int mid = (l + r) / 2;
        push_down(node);
        if (pos <= mid) {
            return query(l, mid, pos, 2 * node + 1);
        } else {
            return query(mid + 1, r, pos, 2 * node + 2);
        }
    }

private:
    void push_down(int node) {
        if (lazy[node] != 0) {
            tree[2 * node + 1] += lazy[node];
            tree[2 * node + 2] += lazy[node];
            lazy[2 * node + 1] += lazy[node];
            lazy[2 * node + 2] += lazy[node];
            lazy[node] = 0;
        }
    }

    int n;
    std::vector<int> tree;
    std::vector<int> lazy;
};

std::vector<int> count_intervals_with_points(const std::vector<std::pair<float, float>>& intervals, const std::vector<float>& points) {
    std::vector<float> all_points;
    for (const auto& interval : intervals) {
        all_points.push_back(interval.first);
        all_points.push_back(interval.second);
    }
    all_points.insert(all_points.end(), points.begin(), points.end());
    std::sort(all_points.begin(), all_points.end());
    all_points.erase(std::unique(all_points.begin(), all_points.end()), all_points.end());

    std::unordered_map<float, int> point_to_idx;
    for (size_t i = 0; i < all_points.size(); ++i) {
        point_to_idx[all_points[i]] = i;
    }

    int n = all_points.size();
    SegmentTree seg_tree(n);

    for (const auto& interval : intervals) {
        int left_idx = point_to_idx[interval.first];
        int right_idx = point_to_idx[interval.second];
        seg_tree.update(0, n - 1, left_idx, right_idx, 0, 1);
    }

    std::vector<int> result;
    for (size_t i = 0; i < points.size(); ++i) {
        float point = points[i];
        int idx = point_to_idx[point];
        int count = seg_tree.query(0, n - 1, idx, 0);
        result.push_back(count);

        // report progress every 1000 points
        if ((i + 1) % 1000 == 0) {
            std::cout << "Processed " << (i + 1) << " / " << points.size() << " points" << std::endl;
        }
    }

    return result;
}

// Generating a linspace vector
std::vector<float> linspace(float start, float end, int num) {
    std::vector<float> linspaced;
    if (num == 0) { return linspaced; }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        linspaced.push_back(start + delta * i);
    }
    return linspaced;
}

int main() {
    std::ifstream infile("intervals_2300.txt");
    std::vector<std::pair<float, float>> intervals;
    std::string line;
    int count = 1;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        float start, end;
        if (!(iss >> start >> end)) {
            break;
        }
        intervals.emplace_back(start, end);
        // report progress every 1000 intervals
        if (count % 1000 == 0)
            std::cout << "count: " << count << std::endl;
        count++;
    }

    // Example usage:
    // std::vector<float> points = {3, 4};
    float start = -3.47e6;
    float end = 0;
    int num = 10000000;

    std::vector<float> points = linspace(start, end, num);

    // Count the number of intervals each point falls into
    std::vector<int> result = count_intervals_with_points(intervals, points);

    // Store the results in a txt file
    std::ofstream outfile("result_2300.txt");
    for (int i = 0; i < result.size(); ++i) {
        outfile << result[i] << std::endl;
    }
    return 0;
}