#include <bits/stdc++.h>
#include <random>
using namespace std;
typedef long long int ll;
typedef long double ld;
typedef pair<ll, ll> pll;

typedef enum
{
    Unconstructed, // 未工事状態
    Occupied,      // 現在道路として使われている
    Constructed,   // 工事完了
} Road_Status;

ll N, M, D, K;

struct Edge {
    ll edge_id;
    ll from;
    ll to;
    ll cost;
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
            ll to;
            ll cost;
            Dijkstra_Edge(ll _to, ll _cost) {
                to = _to;
                cost = _cost;
            }
        };
        vector<Dijkstra_Edge> graph[1010];
        vector<ll> dist;

    public:
        
        Dijkstra(vector<Edge> edge_list) {
            for(ll i = 0; i < edge_list.size(); i++) {
                graph[edge_list[i].from].push_back(Dijkstra_Edge(edge_list[i].to, edge_list[i].cost));
                graph[edge_list[i].to].push_back(Dijkstra_Edge(edge_list[i].from, edge_list[i].cost));
            }
            for(ll i = 0; i < N; i++) dist.push_back(1000000000);
        }
        
        vector<ll> get_dist(ll s) {
            priority_queue<pll, vector<pll>, greater<pll> > que;
            for(ll i = 0; i < N; i++) dist[i] = 1000000000;
            dist[s] = 0;
            que.push(pll(0, s));
            while(!que.empty()){
                pll p = que.top(); que.pop();
                ll v = p.second;
                if(dist[v] < p.first) continue;
                for(Dijkstra_Edge e: graph[v]){
                    if(dist[e.to] > dist[v] + e.cost){
                        dist[e.to] = dist[v] + e.cost;
                        que.push(pll(dist[e.to], e.to));
                    }
                }
            }
            return dist;
        }

};

struct UnionFind {
    private:
        vector<ll> parent;
    
    public:
        UnionFind(ll n) : parent(n, -1) { }
        void init(ll n) { parent.assign(n, -1); }
    
        ll root(ll x) {
            if(parent[x] < 0) return x;
            else return parent[x] = root(parent[x]);
        }
    
        bool issame(ll x, ll y) {
            return root(x) == root(y);
        }
    
        void merge(ll x, ll y) { //親、子
            x = root(x);
            y = root(y);
            if(x == y) return;
            if(parent[x] > parent[y]) swap(x, y);
            parent[x] += parent[y]; // sizeを調整
            parent[y] = x; // 大きい木の根に小さい木をつける, yの親はx
        }
        
        ll size(ll x) {
            return -parent[root(x)];
        }

};


template <typename T> vector<T> random_sample(vector<T> population, int k) {
    if(population.size() < k) {
        cerr << "Error: population.size() < k" << endl;
        return population;
    }
    int num = population.size() - k;
    for(int i = 0; i < num; i++) {
        int idx = rand()%population.size();
        population.erase(population.begin() + idx);
    }
    return population;
}



vector<vector<ll> > compute_dist_matrix(vector<Edge> edge_list, vector<ll> ans, ll day) {
    vector<Edge> new_edge_list;
    for(ll i = 0; i < M; i++) {
        if(ans[edge_list[i].edge_id] == day) continue;
        new_edge_list.push_back(edge_list[i]);
    }
    Dijkstra djk = Dijkstra(new_edge_list);
    vector<vector<ll> > dist;
    for(ll s = 0; s < N; s++) {
        dist.push_back(djk.get_dist(s));
    }
    return dist;
}


ll compute_score(vector<ll> ans, vector<Edge> edge_list) {
    vector<ll> cnt = vector<ll>(D + 1, 0);
    for(ll i = 0; i < M; i++) {
        if(ans[i] == 0) return -1;
        cnt[ans[i]] += 1;
    }
    for(ll i = 1; i <= D; i++) {
        if(cnt[i] > K) {
            return -1;
        }
    }
    ll num = 0;
    ll penalty_count = 0;
    vector<vector<ll> > dist0 = compute_dist_matrix(edge_list, ans, 0);
    vector<ll> fs;
    for(ll d = 1; d <= D; d++) {
        vector<vector<ll> > dist = compute_dist_matrix(edge_list, ans, d);
        ll tmp = 0;
        for(ll i = 0; i < N; i++) {
            for(ll j = i + 1; j < N; j++) {
                if(dist[i][j] == 1000000000) penalty_count++;
                tmp += (dist[i][j] - dist0[i][j]);
            }
        }
        num += tmp;
        fs.push_back(tmp / (N * (N - 1) / 2));
    }
    if(penalty_count > 0) cerr << "Warning: penalty_count > 0" << endl;
    ll den = D * N * (N - 1) / 2;
    ld avg = (ld)num / den * 1000.0;
    return round(avg);
}


int main() {
    cin >> N >> M >> D >> K;
    vector<Edge> edge_list(M);
    vector<ll> unconstructed_edge_list(M);
    vector<ll> ans(M);
    for(ll i = 0; i < M; i++) {
        edge_list[i].edge_id = i;
        cin >> edge_list[i].from >> edge_list[i].to >> edge_list[i].cost;
        edge_list[i].from--;
        edge_list[i].to--;
        unconstructed_edge_list[i] = edge_list[i].edge_id;
    }
    sort(edge_list.begin(), edge_list.end());
    for(ll d = 0; d < D; d++) {
        UnionFind uf = UnionFind(N);
        
        for(ll i = 0; i < M; i++) {
            if(edge_list[i].road_status == Constructed) {
                uf.merge(edge_list[i].from, edge_list[i].to);
            }else {
                edge_list[i].road_status = Unconstructed;
            }
        }

        for(ll i = 0; i < M; i++) {
            if(edge_list[i].road_status == Constructed) continue;
            if(uf.issame(edge_list[i].from, edge_list[i].to)) continue;
            uf.merge(edge_list[i].from, edge_list[i].to);
            edge_list[i].road_status = Occupied;
        }
        

        vector<Edge*> unconstructed_edge;
        for(ll i = 0; i < M; i++) {
            if(edge_list[i].road_status != Unconstructed) continue;
            unconstructed_edge.push_back(&edge_list[i]);
        }

        unconstructed_edge = random_sample(unconstructed_edge, min((ll)unconstructed_edge.size(), K));

        for(ll i = 0; i < unconstructed_edge.size(); i++) {
            unconstructed_edge[i]->road_status = Constructed;
            ans[unconstructed_edge[i]->edge_id] = d + 1;
        }
    }
    
    for(ll i = 0; i < M; i++) {
        cout << ans[i] << " \n"[i == M - 1];
    }

    ll score = compute_score(ans, edge_list);
    cerr << score << endl;

}