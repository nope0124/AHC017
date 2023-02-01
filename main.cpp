#include <bits/stdc++.h>
#include <random>
#include <time.h>
using namespace std;
#define rep(i, N) for(ll i = 0; i < (ll)N; i++)
typedef long long int ll;
typedef long double ld;
typedef pair<ll, ll> pll;

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

ll N, M, D, K;
ld limit = 3.0;
const ll INF = 1e18;

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
        os << "(id: " << edge.edge_id << ", cost: " << edge.cost << ", from: " << edge.from << ", to: " << edge.to << ")";
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
        
        ll score;

    public:
        vector<ll> dist;
        
        Dijkstra(vector<Edge> edge_list, bool is_directed=false) {
            for(ll i = 0; i < edge_list.size(); i++) {
                if(edge_list[i].edge_id <= -1) continue;
                graph[edge_list[i].from].push_back(Dijkstra_Edge(edge_list[i].to, edge_list[i].cost));
                if(is_directed == false) graph[edge_list[i].to].push_back(Dijkstra_Edge(edge_list[i].from, edge_list[i].cost));
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

        vector<Edge> get_edge_list(ll s) {
            vector<Edge> edge_list(N);
            for(ll i = 0; i < N; i++) edge_list[i].edge_id = -1;
            priority_queue<pll, vector<pll>, greater<pll> > que;
            for(ll i = 0; i < N; i++) dist[i] = 1000000000;
            dist[s] = 0;
            edge_list[s].edge_id = -2;
            que.push(pll(0, s));
            while(!que.empty()){
                pll p = que.top(); que.pop();
                ll v = p.second;
                if(dist[v] < p.first) continue;
                for(Dijkstra_Edge e: graph[v]) {
                    if(dist[e.to] > dist[v] + e.cost){
                        dist[e.to] = dist[v] + e.cost;
                        Edge tmp_edge;
                        tmp_edge.edge_id = 0;
                        tmp_edge.from = v;
                        tmp_edge.to = e.to;
                        tmp_edge.cost = e.cost;
                        edge_list[e.to] = tmp_edge;
                        que.push(pll(dist[e.to], e.to));
                    }
                }
            }
            return edge_list;
        }

        ll dfs(Dijkstra_Edge v) {
            if(graph[v.to].size() == 0) {
                score += (N - 1) * v.cost;
                return 1;
            }
            ll node_count = 1;
            for(Dijkstra_Edge e: graph[v.to]) {
                node_count += dfs(e);
            }
            score += (N - node_count) * node_count * v.cost; 
            return node_count;
        }

        ll get_score(ll s) {
            score = 0;
            dfs(Dijkstra_Edge(s, 0));
            
            return score;
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
        return population;
    }
    int num = population.size() - k;
    vector<T> ret_population;
    for(int i = 0; i < k; i++) {
        int idx = rand()%population.size();
        ret_population.push_back(population[idx]);
        population.erase(population.begin() + idx);
    }
    return ret_population;
}



struct Info {
    vector<ll> construction_day_list;
    vector<ll> scores;
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
    
    Score(vector<ll> _construction_day_list, vector<Edge> _edge_list)  {
        edge_list = _edge_list;
        info.construction_day_list = _construction_day_list;
        rep(i, M) info.edge_list_per_day[info.construction_day_list[edge_list[i].edge_id]].push_back(edge_list[i]);
        vector<ll> _scores(D + 1, INF);
        for(int i = 1; i < D + 1; i++) _scores[i] = evaluate_score(i);
        info.scores = _scores;
        penalty_count = 0;
    }

    vector<ll> compute_dist_vector(ll day) { // O(MlogN) = 3000 * 7 = 2*10**4 = 0.2ms
        vector<Edge> new_edge_list;
        for(ll i = 0; i < M; i++) {
            if(info.construction_day_list[edge_list[i].edge_id] == day) continue;
            new_edge_list.push_back(edge_list[i]);
        }
        Dijkstra djk = Dijkstra(new_edge_list);
        return djk.get_dist(day);
    }

    vector<vector<ll> > compute_dist_matrix(ll day) { // O(NMlogN) = 1000 * 3000 * 7 = 2*10**7 = 200ms
        vector<Edge> new_edge_list;
        for(ll i = 0; i < M; i++) {
            if(info.construction_day_list[edge_list[i].edge_id] == day) continue;
            new_edge_list.push_back(edge_list[i]);
        }
        Dijkstra djk = Dijkstra(new_edge_list);
        vector<vector<ll> > tmp_dist;
        for(ll s = 0; s < N; s++) {
            tmp_dist.push_back(djk.get_dist(s));
        }
        return tmp_dist;
    }


    Response delete_penalty(Edge edge, ll to_day) {
        vector<ll> dist_from_day, dist_to_day;
        ll from_day = info.construction_day_list[edge.edge_id]; // 現在の工事日を把握
        ll before_penalty_count = 0;
        dist_from_day = compute_dist_vector(from_day);
        dist_to_day   = compute_dist_vector(to_day);
        for(ll i = 0; i < N; i++) if(dist_from_day[i] == 1000000000) before_penalty_count++;
        for(ll i = 0; i < N; i++) if(dist_to_day[i]   == 1000000000) before_penalty_count++;
        if(before_penalty_count == 0) return OK;
        ll after_penalty_count = 0;
        info.construction_day_list[edge.edge_id] = to_day;
        dist_from_day = compute_dist_vector(from_day);
        dist_to_day   = compute_dist_vector(to_day);
        for(ll i = 0; i < N; i++) if(dist_from_day[i] == 1000000000) after_penalty_count++;
        for(ll i = 0; i < N; i++) if(dist_to_day[i]   == 1000000000) after_penalty_count++;
        if(after_penalty_count == 0) {
            return OK;
        }else if(before_penalty_count > after_penalty_count) {
            return Penalty;
        }else {
            info.construction_day_list[edge.edge_id] = from_day;
            return Failed;
        }
    }

    ll evaluate_score(ll day) { // O(M * logN) = O(30000) 
        vector<Edge> new_edge_list;
        for(ll i = 0; i < M; i++) {
            if(info.construction_day_list[edge_list[i].edge_id] == day) continue;
            new_edge_list.push_back(edge_list[i]);
        }
        Dijkstra djk = Dijkstra(new_edge_list);
        vector<Edge> tmp_edge_list = djk.get_edge_list(0);
        Dijkstra new_djk = Dijkstra(tmp_edge_list, true);
        ll score = 0;
        for(ll i = 0; i < N; i++) if(tmp_edge_list[i].edge_id == -1) score += (N - 1) * 1000000000;
        score += new_djk.get_score(0);

        ll tmp_idx = 1;
        ll tmp_min = -1;
        for(ll i = 0; i < N; i++) if(djk.dist[i] != 1000000000) {
            if(tmp_min < djk.dist[i]) tmp_min = djk.dist[i], tmp_idx = i;
        }

        tmp_edge_list = djk.get_edge_list(tmp_idx);
        new_djk = Dijkstra(tmp_edge_list, true);
        for(ll i = 0; i < N; i++) if(tmp_edge_list[i].edge_id == -1) score += (N - 1) * 1000000000;
        score += new_djk.get_score(tmp_idx);

        return score;
    }

    // Response change(Edge edge1, Edge edge2) {
    //     if(edge1.edge_id == edge2.edge_id) return Failed;
    //     if(info.construction_day_list[edge1.edge_id] == info.construction_day_list[edge2.edge_id]) return Failed;
    //     Info sub_info = info;
    //     swap(info, sub_info);
    //     swap(info.construction_day_list[edge1.edge_id], info.construction_day_list[edge2.edge_id]);
    //     info.scores[info.construction_day_list[edge1.edge_id]] = evaluate_score(info.construction_day_list[edge1.edge_id]);
    //     info.scores[info.construction_day_list[edge2.edge_id]] = evaluate_score(info.construction_day_list[edge2.edge_id]);

    //     // 工事日を変更してスコアが悪化する場合
    //     if(info.scores[info.construction_day_list[edge1.edge_id]] + info.scores[info.construction_day_list[edge2.edge_id]] >= sub_info.scores[info.construction_day_list[edge1.edge_id]] + sub_info.scores[info.construction_day_list[edge2.edge_id]]) {
    //         swap(info, sub_info);
    //         return Failed;
    //     }

    //     // 成功
    //     return OK;
    // }
    Response edge_swap(ll day1, ll day2, ll num=1) {
        if(day1 == day2) return Failed;
        Info sub_info = info;
        swap(info, sub_info);
        num = min(num, (ll)info.edge_list_per_day[day1].size());
        num = min(num, (ll)info.edge_list_per_day[day2].size());
        vector<Edge> edge_list1, edge_list2;
        rep(i, num) {
            ll idx1 = rand()%info.edge_list_per_day[day1].size();
            ll idx2 = rand()%info.edge_list_per_day[day2].size();
            edge_list1.push_back(info.edge_list_per_day[day1][idx1]);
            edge_list2.push_back(info.edge_list_per_day[day2][idx2]);
            info.edge_list_per_day[day1].erase(info.edge_list_per_day[day1].begin() + idx1);
            info.edge_list_per_day[day2].erase(info.edge_list_per_day[day2].begin() + idx2);
        }
        rep(i, num) {
            info.edge_list_per_day[day1].push_back(edge_list2[i]);
            info.edge_list_per_day[day2].push_back(edge_list1[i]);
        }
        rep(i, num) info.construction_day_list[edge_list1[i].edge_id] = day2;
        rep(i, num) info.construction_day_list[edge_list2[i].edge_id] = day1;
        
        info.scores[day1] = evaluate_score(day1);
        info.scores[day2] = evaluate_score(day2);

        // 工事日を変更してスコアが悪化する場合
        if(info.scores[day1] + info.scores[day2] >= sub_info.scores[day1] + sub_info.scores[day2]) {
            swap(info, sub_info);
            return Failed;
        }

        // 成功
        return OK;
    }
    

    // Response update(Edge edge, ll to_day) {
    //     ll from_day = info.construction_day_list[edge.edge_id];
    //     // if(from_day == 0 || to_day == 0) {
    //     //     return FatalError;
    //     // }
    //     if(from_day == to_day) return Failed;
        
    //     Info sub_info = info;
    //     swap(info, sub_info);
    //     info.construction_day_list[edge.edge_id] = to_day;
    //     info.scores[from_day] = evaluate_score(from_day);
    //     // 工事を無くしてもスコアが改善しない場合
    //     if(info.scores[from_day] >= sub_info.scores[from_day]) {
    //         swap(info, sub_info);
    //         return Failed;
    //     }
    //     info.scores[to_day] = evaluate_score(to_day);
    //     // 工事日を変更してスコアが悪化する場合
    //     if(info.scores[from_day] + info.scores[to_day] >= sub_info.scores[from_day] + sub_info.scores[to_day]) {
    //         swap(info, sub_info);
    //         return Failed;
    //     }
    //     // 成功
    //     return OK;
    // }

    Response edge_move(ll day1, ll day2, ll num=1) {
        if(day1 == day2) return Failed;
        Info sub_info = info;
        swap(info, sub_info);
        num = min(num, (ll)info.edge_list_per_day[day1].size());
        num = min(num, (ll)(K - info.edge_list_per_day[day2].size()));
        vector<Edge> edge_list1;
        rep(i, num) {
            ll idx1 = rand()%info.edge_list_per_day[day1].size();
            edge_list1.push_back(info.edge_list_per_day[day1][idx1]);
            info.edge_list_per_day[day1].erase(info.edge_list_per_day[day1].begin() + idx1);
        }
        rep(i, num) {
            info.edge_list_per_day[day2].push_back(edge_list1[i]);
        }
        rep(i, num) info.construction_day_list[edge_list1[i].edge_id] = day2;
        
        info.scores[day1] = evaluate_score(day1);
        
        // 工事を無くしてもスコアが改善しない場合
        if(info.scores[day1] >= sub_info.scores[day1]) {
            swap(info, sub_info);
            return Failed;
        }
        info.scores[day2] = evaluate_score(day2);
        // 工事日を変更してスコアが悪化する場合
        if(info.scores[day1] + info.scores[day2] >= sub_info.scores[day1] + sub_info.scores[day2]) {
            swap(info, sub_info);
            return Failed;
        }
        // 成功
        return OK;
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
        vector<vector<ll> > dist0 = compute_dist_matrix(0);
        for(ll d = 1; d <= D; d++) {
            vector<vector<ll> > dist = compute_dist_matrix(d);
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
    vector<ll> unconstructed_edge_list(M);
    vector<ll> construction_day_list(M);
    vector<ll> construction_vacant_count_per_day(D, K);
    for(ll i = 0; i < M; i++) {
        edge_list[i].edge_id = i;
        cin >> edge_list[i].from >> edge_list[i].to >> edge_list[i].cost;
        edge_list[i].from--;
        edge_list[i].to--;
        unconstructed_edge_list[i] = edge_list[i].edge_id;
    }

    // 初期状態のconstruction_day_listを作成
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

        // unconstructed_edge = random_sample(unconstructed_edge, min((ll)unconstructed_edge.size(), K - 1));
        unconstructed_edge = random_sample(unconstructed_edge, min((ll)unconstructed_edge.size(), (M + D - 1) / D));
        for(ll i = 0; i < unconstructed_edge.size(); i++) {
            unconstructed_edge[i]->road_status = Constructed;
            construction_day_list[unconstructed_edge[i]->edge_id] = d + 1;
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
            construction_vacant_count_per_day[vacant_day[idx]]--;
            construction_day_list[edge_list[i].edge_id] = vacant_day[idx] + 1;
            vacant_day.erase(vacant_day.begin() + idx);
            penalty_edge_list.push_back(edge_list[i]);
        }
    }

    Score score = Score(construction_day_list, edge_list);
    ll cnt = 0;
    
    while(true) {
        clock_t end = clock();
        if((ld)(end - start) / CLOCKS_PER_SEC > limit) break;
        
        if(penalty_edge_list.size() > 0) {
            
            ll penalty_edge_idx = rand()%penalty_edge_list.size();
            vector<ll> to_day_selection;
            for(ll j = 0; j < D; j++) {
                if(score.info.construction_day_list[penalty_edge_list[penalty_edge_idx].edge_id] == j + 1) continue;
                for(ll k = 0; k < construction_vacant_count_per_day[j]; k++) {
                    to_day_selection.push_back(j + 1);
                }
            }
            ll tmp_before_day = score.info.construction_day_list[penalty_edge_list[penalty_edge_idx].edge_id];
            ll tmp_after_day = to_day_selection[rand()%to_day_selection.size()];
            Response response = score.delete_penalty(penalty_edge_list[penalty_edge_idx], tmp_after_day);
            if(response == OK) {
                construction_vacant_count_per_day[tmp_before_day - 1]++;
                construction_vacant_count_per_day[tmp_after_day - 1]--;
                penalty_edge_list.erase(penalty_edge_list.begin() + penalty_edge_idx);
            }else if(response == Penalty) {
                construction_vacant_count_per_day[tmp_before_day - 1]++;
                construction_vacant_count_per_day[tmp_after_day - 1]--;
            }else if(response == Failed) {
            }else if(response == FatalError) {
                return 0;
            }
        }else {
            if(rand()%100 < 100) {
                ll day1 = (rand()%D) + 1;
                ll day2 = (rand()%D) + 1;

                Response response = score.edge_move(day1, day2);
                if(response == OK) {
                    // cnt++;
                    // if(cnt % 10 == 0) {
                    //     cerr << score.compute_score() << endl;
                    // }
                    // construction_vacant_count_per_day[tmp_before_day - 1]++;
                    // construction_vacant_count_per_day[tmp_after_day - 1]--;
                }else if(response == Penalty) {
                    // construction_vacant_count_per_day[tmp_before_day - 1]++;
                    // construction_vacant_count_per_day[tmp_after_day - 1]--;
                }else if(response == Failed) {
                }else if(response == FatalError) {
                    return 0;
                }
            }else {
                ll day1 = (rand()%D) + 1;
                ll day2 = (rand()%D) + 1;

                Response response = score.edge_swap(day1, day2);
                // cerr << setprecision(10) << "update time: " << (ld)(tmp_end - tmp_start) / CLOCKS_PER_SEC << endl;
                if(response == OK) {
                    // cnt++;
                    // if(cnt % 10 == 0) {
                    //     cerr << score.compute_score() << endl;
                    // }
                }else if(response == Penalty) {
                }else if(response == Failed) {
                }else if(response == FatalError) {
                    return 0;
                }
            }
            
        }
        
    }
    // // if(penalty_edge_list.size() > 0) cerr << "################" << endl;
    // // clock_t end = clock();
    // // cerr << setprecision(10) << (ld)(end - start) / CLOCKS_PER_SEC << endl;
    score.info.dump();
    // 注意点、入れ替え実装できていないから本番2000ケースで落ちる可能性高い
    // day=5とか危ない、普通にバカ重い
    // cerr << cnt << endl;
    cerr << score.compute_score() << endl;
    // for(ll i = 0; i < D; i++) {
    //     cerr << construction_vacant_count_per_day[i] << " \n"[i == D - 1];
    // }


    // N = 4, N = 6;
    // vector<ll> from_nodes = {0, 0, 0, 1, 1, 2};
    // vector<ll> to_nodes =   {1, 2, 3, 2, 3, 3};
    // vector<ll> cost_nodes = {1, 2, 3, 4, 5, 6};
    // vector<Edge> edge_nodes;
    // for(ll i = 0; i < N; i++) {
    //     Edge tmp_edge;
    //     tmp_edge.edge_id = 0;
    //     tmp_edge.from = from_nodes[i];
    //     tmp_edge.to = to_nodes[i];
    //     tmp_edge.cost = cost_nodes[i];
    //     edge_nodes.push_back(tmp_edge);
    // }
    // Dijkstra djk = Dijkstra(edge_nodes);
    // vector<ll> dist = djk.get_dist(0);
    // for(ll i = 0; i < N; i++) cout << dist[i] << endl;
    // vector<Edge> next_edges = djk.get_edge_list(0);
    // for(int i = 0; i < next_edges.size(); i++) cerr << "edges "<< next_edges[i] << endl;
    // Dijkstra next_djk = Dijkstra(next_edges, true);
    // cout << next_djk.get_score(0) << endl;

}