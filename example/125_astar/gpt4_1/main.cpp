#include <vector>
#include <queue>
#include <unordered_map>
#include <iostream>

struct Node {
    int x, y;
    int g, h, f;

    bool operator==(const Node& other) const {
        return x == other.x && y == other.y;
    }
};

struct NodeHash {
    std::size_t operator()(const Node& node) const {
        return std::hash<int>()(node.x) ^ std::hash<int>()(node.y);
    }
};

bool a_star(std::vector<std::vector<int>> grid, Node start, Node goal, std::vector<Node>& path) {
    auto cmp = [](const Node& a, const Node& b) { return a.f > b.f; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> open(cmp);
    std::unordered_map<Node, Node, NodeHash> came_from;
    std::unordered_map<Node, int, NodeHash> cost_so_far;

    open.push(start);
    came_from[start] = start;
    cost_so_far[start] = 0;

    while (!open.empty()) {
        auto current = open.top();
        open.pop();

        if (current == goal) {
            auto temp = current;
            while (!(temp == start)) {
                path.push_back(temp);
                temp = came_from[temp];
            }
            path.push_back(start);
            return true;
        }

        for (auto& dir : {Node{-1, 0}, Node{1, 0}, Node{0, -1}, Node{0, 1}}) {
            auto next = Node{current.x + dir.x, current.y + dir.y};
            if (next.x < 0 || next.y < 0 || next.x >= grid.size() || next.y >= grid[0].size() || grid[next.x][next.y] == 1)
                continue;

            auto new_cost = cost_so_far[current] + 1;
            if (cost_so_far.find(next) == cost_so_far.end() || new_cost < cost_so_far[next]) {
                cost_so_far[next] = new_cost;
                auto h = abs(goal.x - next.x) + abs(goal.y - next.y);
                auto f = new_cost + h;
                next.g = new_cost;
                next.h = h;
                next.f = f;
                open.push(next);
                came_from[next] = current;
            }
        }
    }

    return false;
}

int main() {
    auto grid = std::vector<std::vector<int>>{
        {0, 0, 0, 1, 0},
        {1, 1, 0, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 0},
    };
    auto start = Node{0, 0};
    auto goal = Node{4, 4};
    std::vector<Node> path;
    if (a_star(grid, start, goal, path)) {
        for (auto it = path.rbegin(); it != path.rend(); ++it)
            std::cout << "(" << it->x << ", " << it->y << ")\n";
    } else {
        std::cout << "No path found.\n";
    }
    return 0;
}
