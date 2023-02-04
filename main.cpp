#pragma GCC target("sse,sse2,sse3,ssse3,sse4,fma,abm,mmx,avx,avx2")

#include <bits/stdc++.h>
#include <random>
using namespace std;
#define rep(i, N) for(int i = 0; i < (int)N; i++)
#define IOS ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
typedef long long int ll;
typedef long double ld;


typedef enum {
    Unconstructed, // 未工事状態
    Occupied,      // 現在道路として使われている
    Constructed,   // 工事完了
} Road_Status;

typedef enum {
    FatalError, // 想定外のエラー
    Failed, // 悪化
    OK,      // 改善
    Penalty,   // ペナルティが残っている
} Response;

short N, M, D, K;
ld limit = 3.0;
constexpr ll INF = 1e18;

struct Edge {
    short edge_id;
    short from;
    short to;
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
        os << "(id: " << edge.edge_id << ", cost: " << edge.cost << ", from: " << edge.from << ", to: " << edge.to << ")";
        return os;
    }
};



namespace StaticGraphImpl {

template <typename T, bool Cond = is_void<T>::value>
struct E;
template <typename T>
struct E<T, false> {
    int to;
    T cost;
    E() {}
    E(const int& v, const T& c) : to(v), cost(c) {}
    operator int() const { return to; }
};
template <typename T>
struct E<T, true> {
    int to;
    E() {}
    E(const int& v) : to(v) {}
    operator int() const { return to; }
};

template <typename T = void>
struct StaticGraph {
    private:
    template <typename It>
    struct Es {
        It b, e;
        It begin() const { return b; }
        It end() const { return e; }
        int size() const { return int(e - b); }
        auto&& operator[](int i) const { return b[i]; }
    };
    
    int N, M, ec;
    vector<int> head;
    vector<pair<int, E<T> > > buf;
    vector<E<T> > es;

    void build() {
        partial_sum(begin(head), end(head), begin(head));
        es.resize(M);
        for (auto&& [u, e] : buf) es[--head[u]] = e;
    }

    public:
    StaticGraph(int _n, int _m) : N(_n), M(_m), ec(0), head(N + 1, 0) {
        buf.reserve(M);
    }

    template <typename... Args>
    void add_edge(int u, Args&&... args) {
    #pragma GCC diagnostic ignored "-Wnarrowing"
        buf.emplace_back(u, E<T>(std::forward<Args>(args)...));
    #pragma GCC diagnostic warning "-Wnarrowing"
        ++head[u];
        if ((int)buf.size() == M) build();
    }

    Es<typename vector<E<T> >::iterator> operator[](int u) {
        return {begin(es) + head[u], begin(es) + head[u + 1]};
    }
    const Es<typename vector<E<T> >::const_iterator> operator[](int u) const {
        return {begin(es) + head[u], begin(es) + head[u + 1]};
    }
    int size() const { return N; }
};

}  // namespace StaticGraphImpl

using StaticGraphImpl::StaticGraph;




template <typename Key, typename Val>
struct RadixHeap {
    using uint = typename make_unsigned<Key>::type;
    static constexpr int bit = sizeof(Key) * 8;
    array<vector<pair<uint, Val> >, bit + 1> vs;
    array<uint, bit + 1> ms;

    int s;
    uint last;

    RadixHeap() : s(0), last(0) { fill(begin(ms), end(ms), uint(-1)); }

    bool empty() const { return s == 0; }

    int size() const { return s; }

    __attribute__((target("lzcnt"))) inline uint64_t getbit(uint a) const {
        return 64 - __builtin_clzll(a);
    }

    void push(const uint &key, const Val &val) {
        s++;
        uint64_t b = getbit(key ^ last);
        vs[b].emplace_back(key, val);
        ms[b] = min(key, ms[b]);
    }

    pair<uint, Val> pop() {
        if (ms[0] == uint(-1)) {
            int idx = 1;
            while (ms[idx] == uint(-1)) idx++;
            last = ms[idx];
            for (auto &p : vs[idx]) {
                uint64_t b = getbit(p.first ^ last);
                vs[b].emplace_back(p);
                ms[b] = min(p.first, ms[b]);
            }
            vs[idx].clear();
            ms[idx] = uint(-1);
        }
        --s;
        auto res = vs[0].back();
        vs[0].pop_back();
        if (vs[0].empty()) ms[0] = uint(-1);
        return res;
    }
};



template <typename T>
vector<T> dijkstra(StaticGraph<T>& g, int start = 0) {
    vector<T> d(g.size(), T(1000000000));
    RadixHeap<T, int> Q;
    d[start] = 0;
    Q.push(0, start);
    while (!Q.empty()) {
        auto p = Q.pop();
        int u = p.second;
        if (d[u] < T(p.first)) continue;
        T du = d[u];
        for (auto&& [v, c] : g[u]) {
            if (d[v] == T(1000000000) || du + c < d[v]) {
                d[v] = du + c;
                Q.push(d[v], v);
            }
        }
    }
    return d;
}



template <typename T>
ll dijkstra_get_score(StaticGraph<T>& g, int start = 0) {
    ll score = 0;
    vector<T> d(g.size(), T(1000000000));
    RadixHeap<T, int> Q;
    d[start] = 0;
    Q.push(0, start);
    while (!Q.empty()) {
        auto p = Q.pop();
        int u = p.second;
        if (d[u] < T(p.first)) continue;
        T du = d[u];
        for (auto&& [v, c] : g[u]) {
            if (d[v] == T(1000000000) || du + c < d[v]) {
                d[v] = du + c;
                Q.push(d[v], v);
            }
        }
    }
    rep(i, N) score += d[i];
    return score;
}













struct UnionFind {
    private:
        vector<int> parent;
    
    public:
        UnionFind(int n) : parent(n, -1) { }
        void init(int n) { parent.assign(n, -1); }
    
        int root(int x) {
            if(parent[x] < 0) return x;
            else return parent[x] = root(parent[x]);
        }
    
        bool issame(int x, int y) {
            return root(x) == root(y);
        }
    
        void merge(int x, int y) { //親、子
            x = root(x);
            y = root(y);
            if(x == y) return;
            if(parent[x] > parent[y]) swap(x, y);
            parent[x] += parent[y]; // sizeを調整
            parent[y] = x; // 大きい木の根に小さい木をつける, yの親はx
        }
        
        int size(int x) {
            return -parent[root(x)];
        }

};


template <typename T> vector<T> random_sample(vector<T> population, int k) {
    if(population.size() < k) {
        return population;
    }
    vector<T> ret_population;
    for(int i = 0; i < k; i++) {
        int idx = rand()%population.size();
        ret_population.push_back(population[idx]);
        population.erase(population.begin() + idx);
    }
    return ret_population;
}



struct Info {
    vector<short> construction_day_list;
    vector<Edge> edge_list_per_day[31];

    void dump() {
        for(ll i = 0; i < M; i++) {
            cout << construction_day_list[i] << " \n"[i == M - 1];
        }
    }
};


struct Score {
    
    vector<Edge> edge_list;
    ll penalty_count;
    Info info;
    
    Score(vector<short> _construction_day_list, vector<Edge> _edge_list)  {
        edge_list = _edge_list;
        info.construction_day_list = _construction_day_list;
        rep(i, M) info.edge_list_per_day[info.construction_day_list[edge_list[i].edge_id]].push_back(edge_list[i]);
        penalty_count = 0;
    }

    vector<int> compute_dist_vector(short day) { // O(MlogN) = 3000 * 7 = 2*10**4 = 0.2ms
        short graph_size = M - info.edge_list_per_day[day].size();
        StaticGraph<int> g(N, graph_size*2);
        rep(i, M) {
          	if(info.construction_day_list[edge_list[i].edge_id] == day) continue;
            g.add_edge(edge_list[i].from, edge_list[i].to, edge_list[i].cost);
            g.add_edge(edge_list[i].to, edge_list[i].from, edge_list[i].cost);
        }
        return dijkstra(g, day);
    }

    vector<vector<int> > compute_dist_matrix(short day) { // O(NMlogN) = 1000 * 3000 * 7 = 2*10**7 = 200ms
        short graph_size = M - info.edge_list_per_day[day].size();
        StaticGraph<int> g(N, graph_size*2);
        rep(i, M) {
          	if(info.construction_day_list[edge_list[i].edge_id] == day) continue;
            g.add_edge(edge_list[i].from, edge_list[i].to, edge_list[i].cost);
            g.add_edge(edge_list[i].to, edge_list[i].from, edge_list[i].cost);
        }
        vector<vector<int> > tmp_dist;
        for(ll s = 0; s < N; s++) {
            tmp_dist.push_back(dijkstra(g, s));
        }
        return tmp_dist;
    }

    ll evaluate_score(short day, short s1, short s2) { // O(MlogN) = O(30000) 
      	short graph_size = M - info.edge_list_per_day[day].size();
        StaticGraph<int> g(N, graph_size*2);
        for(short i = 0; i < M; i++) {
            if(info.construction_day_list[edge_list[i].edge_id] == day) continue;
            g.add_edge(edge_list[i].from, edge_list[i].to, edge_list[i].cost);
            g.add_edge(edge_list[i].to, edge_list[i].from, edge_list[i].cost);
        }

        ll score = dijkstra_get_score(g, s1);
      	score += dijkstra_get_score(g, s2);
        return score;
    }


    Response delete_penalty(Edge edge, ll to_day) {
        ll from_day = info.construction_day_list[edge.edge_id]; // 現在の工事日を把握
        if(from_day == to_day) return Failed;
        if(info.edge_list_per_day[to_day].size() >= K) return Failed;
        ll before_penalty_count = 0;
        vector<int> dist_from_day = compute_dist_vector(from_day);
        vector<int> dist_to_day   = compute_dist_vector(to_day);
        for(ll i = 0; i < N; i++) if(dist_from_day[i] == 1000000000) before_penalty_count++;
        for(ll i = 0; i < N; i++) if(dist_to_day[i]   == 1000000000) before_penalty_count++;
        if(before_penalty_count == 0) return OK;

        Info sub_info = info;
        swap(info, sub_info);
        ll after_penalty_count = 0;
        info.construction_day_list[edge.edge_id] = to_day;
        ll idx = -1;
        rep(i, info.edge_list_per_day[from_day].size()) {
            if(info.edge_list_per_day[from_day][i].edge_id == edge.edge_id) {
                idx = i;
                break;
            }
        }
        info.edge_list_per_day[to_day].push_back(info.edge_list_per_day[from_day][idx]);
        info.edge_list_per_day[from_day].erase(info.edge_list_per_day[from_day].begin() + idx);
        
        dist_from_day = compute_dist_vector(from_day);
        dist_to_day   = compute_dist_vector(to_day);
        for(ll i = 0; i < N; i++) if(dist_from_day[i] == 1000000000) after_penalty_count++;
        for(ll i = 0; i < N; i++) if(dist_to_day[i]   == 1000000000) after_penalty_count++;
        if(after_penalty_count == 0) {
            return OK;
        }else if(before_penalty_count > after_penalty_count) {
            return Penalty;
        }else {
            swap(info, sub_info);
            return Failed;
        }
    }


    void edge_move(ll day1, ll day2) {
        if(day1 == day2) return;
        ll num = 1;
        num = min(num, (ll)info.edge_list_per_day[day1].size());
        num = min(num, (ll)(K - info.edge_list_per_day[day2].size()));
        if(num == 0) return;
        Info sub_info = info;
        swap(info, sub_info);
        ll idx1 = rand()%info.edge_list_per_day[day1].size();
        Edge edge = info.edge_list_per_day[day1][idx1];
        

        ll before_day1_score = evaluate_score(day1, edge.from, edge.to);

        ll before_day2_score = evaluate_score(day2, edge.from, edge.to);

        info.construction_day_list[edge.edge_id] = day2;
        info.edge_list_per_day[day2].push_back(edge);
        info.edge_list_per_day[day1].erase(info.edge_list_per_day[day1].begin() + idx1);

        ll after_day1_score = evaluate_score(day1, edge.from, edge.to);

        ll after_day2_score = evaluate_score(day2, edge.from, edge.to);
        
        ll score = (before_day1_score + before_day2_score) - (after_day1_score + after_day2_score);
        if(score >= 0) {
            // 成功
            return;
        }else {
            // 工事日を移動してスコアが悪化する場合
            swap(info, sub_info);
            return;
        }
        
    }
    

    ll compute_score() { // O(DNMlogN) = 30 * 1000 * 3000 * 7 = 6*10**8 = 6000ms
        penalty_count = 0;
        vector<ll> cnt = vector<ll>(D + 1, 0);
        for(ll i = 0; i < M; i++) {
            if(info.construction_day_list[i] == 0) return -2;
            cnt[info.construction_day_list[i]] += 1;
        }
        for(ll i = 1; i <= D; i++) {
            if(cnt[i] > K) {
                return -1;
            }
        }
        ll num = 0;
        vector<ll> fs;
        vector<vector<int> > dist0 = compute_dist_matrix(0);
        for(ll d = 1; d <= D; d++) {
            vector<vector<int> > dist = compute_dist_matrix(d);
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
        ll den = D * N * (N - 1) / 2;
        ld avg = (ld)num / den * 1000.0;
        return round(avg);
    }

};






int main() {
    clock_t start = clock();
    cin >> N >> M >> D >> K;
    vector<Edge> edge_list(M);
    vector<short> construction_day_list(M);
    vector<ll> construction_vacant_count_per_day(D, K);
    for(ll i = 0; i < M; i++) {
        edge_list[i].edge_id = i;
        cin >> edge_list[i].from >> edge_list[i].to >> edge_list[i].cost;
        edge_list[i].from--;
        edge_list[i].to--;
    }

    // sort(edge_list.begin(), edge_list.end());
    // 初期状態のconstruction_day_listを作成
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
        
        vector<Edge*> unconstructed_edge_list;
        for(ll i = 0; i < M; i++) {
            if(edge_list[i].road_status != Unconstructed) continue;
            unconstructed_edge_list.push_back(&edge_list[i]);
        }

        unconstructed_edge_list = random_sample(unconstructed_edge_list, min((int)unconstructed_edge_list.size(), (M + D - 1) / D));
        // 工事日を確定
        for(ll i = 0; i < unconstructed_edge_list.size(); i++) {
            unconstructed_edge_list[i]->road_status = Constructed;
            construction_day_list[unconstructed_edge_list[i]->edge_id] = d + 1;
            construction_vacant_count_per_day[d]--;
        }
    }

    // 最終状態でOccuppiedになっている危険な辺を取得する
    vector<Edge> penalty_edge_list;
    vector<ll> vacant_day;
    for(ll i = 0; i < D; i++) {
        for(ll j = 0; j < construction_vacant_count_per_day[i]; j++) {
            vacant_day.push_back(i);
        }
    }
    // 危険な辺をランダムな位置に振り分ける
    for(ll i = 0; i < M; i++) {
        if(construction_day_list[edge_list[i].edge_id] == 0) {
            ll idx = rand()%vacant_day.size();
            construction_day_list[edge_list[i].edge_id] = vacant_day[idx] + 1;
            vacant_day.erase(vacant_day.begin() + idx);
            penalty_edge_list.push_back(edge_list[i]);
        }
    }

    Score score = Score(construction_day_list, edge_list);
    vector<short> binary_selection1(D), binary_selection2(D);
    while(true) {
        clock_t end = clock();
        
        if((ld)(end - start) / CLOCKS_PER_SEC > limit) break;
        if(penalty_edge_list.size() > 0) {
            ll penalty_edge_idx = rand()%penalty_edge_list.size();
            ll tmp_after_day = (rand()%D) + 1;
            Response response = score.delete_penalty(penalty_edge_list[penalty_edge_idx], tmp_after_day);
            if(response == OK) {
                penalty_edge_list.erase(penalty_edge_list.begin() + penalty_edge_idx);
            }
        }else {
            for(ll d = 1; d <= D; d++) {
                if(d == 1) {
                    binary_selection1[d - 1] = score.info.edge_list_per_day[d].size();
                    binary_selection2[d - 1] = K - score.info.edge_list_per_day[d].size();
                }else {
                    binary_selection1[d - 1] = binary_selection1[d - 2] + score.info.edge_list_per_day[d].size();
                    binary_selection2[d - 1] = binary_selection2[d - 2] + (K - score.info.edge_list_per_day[d].size());
                }
            }
            ll day1 = lower_bound(binary_selection1.begin(), binary_selection1.end(), (rand()%M + 1)) - binary_selection1.begin() + 1;
            ll day2 = lower_bound(binary_selection2.begin(), binary_selection2.end(), (rand()%(K*D - M) + 1)) - binary_selection2.begin() + 1;
            score.edge_move(day1, day2);
        }
        
    }

    score.info.dump();
  	cerr << score.compute_score() << endl;
    rep(i, D) {
        cerr << score.info.edge_list_per_day[i].size() << " \n"[i == D - 1];
    }

}