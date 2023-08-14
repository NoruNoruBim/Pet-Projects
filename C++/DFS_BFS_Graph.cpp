#include <iostream>
#include <vector>
#include <queue>
#include <map>

using namespace std;

typedef enum {
	WHITE,
	GRAY,
	BLACK,
} color;


map<int, char> NAME = { {0, 'r'}, {1, 's'}, {2, 't'}, {3, 'u'}, {4, 'v'}, {5, 'w'}, {6, 'x'}, {7, 'y'} };

void BFS(vector<vector<int>> graph, int s, vector<int>& p, vector<int>& d, vector<color>& colors);
void DFS_visit(vector<vector<int>> graph, vector<color>& colors, vector<int>& p, int u);
void DFS(vector<vector<int>> graph, vector<color>& colors, vector<int>& p);


int main() {
	vector<vector<int>> graph = { {0, 1, 0, 0, 1, 0, 0, 0}
								, {1, 0, 0, 0, 0, 1, 0, 0,}
								, {0, 0, 0, 1, 0, 1, 1, 0}
								, {0, 0, 1, 0, 0, 0, 1, 1}
								, {1, 0, 0, 0, 0, 0, 0, 0}
								, {0, 1, 1, 0, 0, 0, 1, 0}
								, {0, 0, 1, 1, 0, 1, 0, 1}
								, {0, 0, 0, 1, 0, 0, 1, 0} };
	vector<int> p(8);
	vector<int> d(8);
	for (int i = 0; i < 8; i++) d[i] = -1;
	for (int i = 0; i < 8; i++) p[i] = -1;
	vector<color> colors(8);
	DFS(graph, colors, p);

	for (int i = 0; i < 8; i++) {
		cout << p[i] << " ";
	} cout << endl << endl;

	for (int i = 0; i < 8; i++) {
		cout << colors[i] << " ";
	} cout << endl << endl;

	return 0;
}

void BFS(vector<vector<int>> graph, int s, vector<int>& p, vector<int>& d, vector<color>& colors) {
	queue<int> q;
	q.push(s);
	colors[s] = GRAY;
	d[s] = 0;

	while (!q.empty()) {
		cout << NAME[q.front()] << endl;
		for (int i = 0; i < graph[q.front()].size(); i++) {
			if (graph[q.front()][i] == 1 && colors[i] == WHITE) {
				q.push(i);
				d[i] = d[q.front()] + 1;
				p[i] = q.front();
				colors[i] = GRAY;
			}
		}
		colors[q.front()] = BLACK;
		q.pop();
	}
}

void DFS_visit(vector<vector<int>> graph, vector<color>& colors, vector<int>& p, int u) {
	colors[u] = GRAY;
	for (int i = 0; i < graph[u].size(); i++) {
		if (graph[u][i] == 1 && colors[i] == WHITE) {
			p[i] = u;
			DFS_visit(graph, colors, p, i);
		}
	}
	colors[u] = BLACK;
}


void DFS(vector<vector<int>> graph, vector<color>& colors, vector<int>& p) {
	for (int i = 0; i < graph.size(); i++) {
		if (colors[i] == WHITE) {
			DFS_visit(graph, colors, p, i);
		}
	}
}
