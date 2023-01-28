#include <bits/stdc++.h>
#include <random>
using namespace std;
typedef long long int ll;
typedef long double ld;
typedef pair<int, int> pint;



typedef enum
{
    Unconstructed,
    Occupied,
    Constructed,
} Road_Status;

short N, M, D, K;


struct Edge {
    short edge_id;
    short from;
    short to;
    int cost;
    Road_Status road_status = Unconstructed;
    bool operator==(const Edge& other) { return edge_id == other.edge_id; }
    bool operator!=(const Edge& other) { return edge_id != other.edge_id; }
    bool operator<(const Edge& other) const {
        if(cost == other.cost) {
            if(edge_id == other.edge_id) {
                return false;
            }else {
                return edge_id < other.edge_id;
            }
        }else {
            return cost < other.cost;
        }
    }
    bool operator>(const Edge& other) const {
        if(cost == other.cost) {
            if(edge_id == other.edge_id) {
                return false;
            }else {
                return edge_id > other.edge_id;
            }
        }else {
            return cost > other.cost;
        }
    }
    friend ostream& operator<<(ostream& os, const Edge& edge) {
        os << "(id: " << edge.edge_id << ", cost: " << edge.cost << ")";
        return os;
    }
};

struct Dijkstra {
    private:
        struct Dijkstra_Edge {
            short to;
            int cost;
            Dijkstra_Edge(short _to, int _cost) {
                to = _to;
                cost = _cost;
            }
        };
        vector<Dijkstra_Edge> graph[1010];
        vector<int> dist;

    public:
        
        Dijkstra(vector<Edge> edge_list) {
            for(short i = 0; i < edge_list.size(); i++) {
                graph[edge_list[i].from].push_back(Dijkstra_Edge(edge_list[i].to, edge_list[i].cost));
                graph[edge_list[i].to].push_back(Dijkstra_Edge(edge_list[i].from, edge_list[i].cost));
            }
            for(short i = 0; i < N; i++) dist.push_back(1000000000);
        }
        
        vector<int> get_dist(int s) {
            priority_queue<pint, vector<pint>, greater<pint> > que;
            for(short i = 0; i < N; i++) dist[i] = 1000000000;
            dist[s] = 0;
            que.push(pint(0, s));
            while(!que.empty()){
                pint p = que.top(); que.pop();
                int v = p.second;
                if(dist[v] < p.first) continue;
                for(Dijkstra_Edge e: graph[v]){
                    if(dist[e.to] > dist[v] + e.cost){
                        dist[e.to] = dist[v] + e.cost;
                        que.push(pint(dist[e.to], e.to));
                    }
                }
            }
            return dist;
        }

};

struct UnionFind {
    private:
        vector<short> parent;
    
    public:
        UnionFind(short n) : parent(n, -1) { }
        void init(short n) { parent.assign(n, -1); }
    
        int root(short x) {
            if(parent[x] < 0) return x;
            else return parent[x] = root(parent[x]);
        }
    
        bool issame(short x, short y) {
            return root(x) == root(y);
        }
    
        void merge(short x, short y) { //親、子
            x = root(x);
            y = root(y);
            if(x == y) return;
            if(parent[x] > parent[y]) swap(x, y);
            parent[x] += parent[y]; // sizeを調整
            parent[y] = x; // 大きい木の根に小さい木をつける, yの親はx
        }
        
        short size(short x) {
            return -parent[root(x)];
        }

};



vector<vector<int> > compute_dist_matrix(vector<Edge> edge_list, vector<short> ans, short day) {
    vector<Edge> new_edge_list;
    for(short i = 0; i < M; i++) {
        if(ans[edge_list[i].edge_id] == day) continue;
        new_edge_list.push_back(edge_list[i]);
    }
    Dijkstra djk = Dijkstra(new_edge_list);
    vector<vector<int> > dist;
    for(short s = 0; s < N; s++) {
        dist.push_back(djk.get_dist(s));
    }
    return dist;
}




ll compute_score(vector<short> ans, vector<Edge> edge_list) {
    vector<short> cnt = vector<short>(D + 1, 0);
    for(short i = 0; i < M; i++) {
        cnt[ans[i]] += 1;
    }
    for(short i = 1; i <= D; i++) {
        if(cnt[i] > K) {
            return -1;
        }
    }
    ll num = 0;
    vector<vector<int> > dist0 = compute_dist_matrix(edge_list, ans, 0);
    vector<ll> fs;
    for(short d = 1; d <= D; d++) {
        vector<vector<int> > dist = compute_dist_matrix(edge_list, ans, d);
        ll tmp = 0;
        for(short i = 0; i < N; i++) {
            for(short j = i + 1; j < N; j++) {
                // if(i == 0) cerr << dist[i][j] << endl;
                tmp += (dist[i][j] - dist0[i][j]);
            }
        }
        num += tmp;
        
        fs.push_back(tmp / (N * (N - 1) / 2));
        // cerr << fs.back() << endl;
    }
    
    ll den = D * N * (N - 1) / 2;
    ld avg = (ld)num / den * 1000.0;
    return round(avg);
}


int main() {
    cin >> N >> M >> D >> K;
    vector<Edge> edge_list(M);
    vector<short> unconstructed_edge_list(M);
    vector<short> ans(M);
    for(short i = 0; i < M; i++) {
        edge_list[i].edge_id = i;
        cin >> edge_list[i].from >> edge_list[i].to >> edge_list[i].cost;
        edge_list[i].from--;
        edge_list[i].to--;
        unconstructed_edge_list[i] = edge_list[i].edge_id;
    }
    sort(edge_list.begin(), edge_list.end());
    short unconstructed_number = M;
    for(short d = 0; d < D; d++) {
        UnionFind uf = UnionFind(N);
        
        for(short i = 0; i < M; i++) {
            if(edge_list[i].road_status == Constructed) {
                uf.merge(edge_list[i].from, edge_list[i].to);
            }else {
                edge_list[i].road_status = Unconstructed;
            }
        }

        for(short i = 0; i < M; i++) {
            if(edge_list[i].road_status == Constructed) continue;
            if(uf.issame(edge_list[i].from, edge_list[i].to)) continue;
            uf.merge(edge_list[i].from, edge_list[i].to);
            edge_list[i].road_status = Occupied;
        }
        

        vector<Edge*> unconstructed_edge;
        for(short i = 0; i < M; i++) {
            if(edge_list[i].road_status != Unconstructed) continue;
            unconstructed_edge.push_back(&edge_list[i]);
        }
        short unconstructed_edge_number = unconstructed_edge.size();
        for(short i = 0; i < min(unconstructed_edge_number, K); i++) {
        // for(short i = 0; i < min((int)unconstructed_edge_number, (M + D - 1) / D); i++) {
            short idx = rand()%unconstructed_edge.size();
            unconstructed_edge[idx]->road_status = Constructed;
            ans[unconstructed_edge[idx]->edge_id] = d + 1;
            unconstructed_edge.erase(unconstructed_edge.begin() + idx);
        }
    }
    
    for(short i = 0; i < M; i++) {
        cout << ans[i] << " \n"[i == M - 1];
    }

    // ll score = compute_score(ans, edge_list);
    // cerr << score << endl;

}