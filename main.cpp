#include <bits/stdc++.h>
#include <random>
#include <time.h>
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
const ll INF = 1e16;

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
    vector<T> ret_population;
    for(int i = 0; i < k; i++) {
        int idx = rand()%population.size();
        ret_population.push_back(population[idx]);
        population.erase(population.begin() + idx);
    }
    return ret_population;
}


struct WarshallFloyd {
    ll initial_total_cost;
    WarshallFloyd(vector<vector<ll> > dist0) {
        initial_total_cost = 0;
        for(ll i = 0; i < N; i++) {
            for(ll j = i + 1; j < N; j++) {
                initial_total_cost += dist0[i][j];
            }
        }
    }
    ll add_edge(vector<vector<ll> > &dist, ll s, ll t, ll cost) {
        dist[s][t] = dist[t][s] = min(dist[s][t], cost);
        set<ll> st;
        st.insert(s);
        st.insert(t);
        for(ll k : st) {
            for(ll i = 0; i < N; i++) {
                for(ll j = 0; j < N; j++) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
        ll total_cost = -initial_total_cost;
        for(ll i = 0; i < N; i++) {
            for(ll j = i + 1; j < N; j++) {
                total_cost += dist[i][j];
            }
        }
        return total_cost;
    }
};



struct Score {
    
    vector<Edge> edge_list;
    vector<ll> ans;
    vector<ll> scores;
    vector<vector<vector<ll> > > dist;
    ll penalty_count;
    
    Score(vector<ll> _ans, vector<Edge> _edge_list) {
        ans = _ans;
        edge_list = _edge_list;
        vector<ll> _scores(D + 1);
        vector<vector<vector<ll> > > _dist(D + 1, vector<vector<ll> >(N, vector<ll>(N, 1000000000)));
        dist = _dist;
        for(ll i = 0; i < D + 1; i++) _scores[i] = INF;
        scores = _scores;
        dist[0] = compute_dist_matrix(0);
        penalty_count = 0;
    }

    vector<vector<ll> > compute_dist_matrix(ll day) { // O(NMlogN) = 1000 * 3000 * 7 = 2*10**7 = 200ms
        vector<Edge> new_edge_list;
        for(ll i = 0; i < M; i++) {
            if(ans[edge_list[i].edge_id] == day) continue;
            new_edge_list.push_back(edge_list[i]);
        }
        Dijkstra djk = Dijkstra(new_edge_list);
        vector<vector<ll> > tmp_dist;
        for(ll s = 0; s < N; s++) {
            tmp_dist.push_back(djk.get_dist(s));
        }
        return tmp_dist;
    }

    ll compute_score_per_day(ll day) {
        ll tmp = 0;
        for(ll i = 0; i < N; i++) {
            for(ll j = i + 1; j < N; j++) {
                if(dist[day][i][j] == 1000000000) penalty_count++;
                tmp += (dist[day][i][j] - dist[0][i][j]);
            }
        }
        return tmp;
    }

    ll update_construction_day(Edge edge, ll after_day) {
        ll before_day = ans[edge.edge_id];
        if(before_day == 0 || after_day == 0) {
            // cerr << "Error: before_day == 0 || after_day == 0" << endl;
            return -1;
        }
        if(before_day == after_day) {
            // cerr << "Error: before_day == after_day" << endl;
            return -1;
        }
        if(scores[before_day] == INF) {
            dist[before_day] = compute_dist_matrix(before_day);
            scores[before_day] = compute_score_per_day(before_day);
        }
        if(scores[after_day]  == INF) {
            dist[after_day] = compute_dist_matrix(after_day);
            scores[after_day]  = compute_score_per_day(after_day);
        }
        penalty_count = 0;
        vector<ll> sub_ans = ans;
        vector<ll> sub_scores = scores;
        vector<vector<vector<ll> > > sub_dist = dist;
        swap(ans, sub_ans);
        swap(scores, sub_scores);
        swap(dist, sub_dist);
        ans[edge.edge_id] = after_day;
        WarshallFloyd wf = WarshallFloyd(dist[0]);
        scores[before_day] = wf.add_edge(dist[before_day], edge.from, edge.to, edge.cost);
        // 工事を無くしてもスコアが改善しない場合
        if(scores[before_day] >= sub_scores[before_day]) {
            // cerr << "Error: scores[before_day] >= sub_scores[before_day]" << endl;
            swap(ans, sub_ans);
            swap(scores, sub_scores);
            swap(dist, sub_dist);
            return -1;
        }
        dist[after_day] = compute_dist_matrix(after_day);
        scores[after_day] = compute_score_per_day(after_day);
        // 工事日を変更してスコアが悪化する場合
        if(scores[before_day] + scores[after_day] >= sub_scores[before_day] + sub_scores[after_day]) {
            // cerr << "Error: scores[before_day] + scores[after_day] >= sub_scores[before_day] + sub_scores[after_day]" << endl;
            swap(ans, sub_ans);
            swap(scores, sub_scores);
            swap(dist, sub_dist);
            return -1;
        }
        // 成功
        return penalty_count;
    }

    

    ll compute_score() { // O(DNMlogN) = 30 * 1000 * 3000 * 7 = 6*10**8 = 6000ms
        penalty_count = 0;
        vector<ll> cnt = vector<ll>(D + 1, 0);
        for(ll i = 0; i < M; i++) {
            if(ans[i] == 0) return -2;
            cnt[ans[i]] += 1;
        }
        for(ll i = 1; i <= D; i++) {
            if(cnt[i] > K) {
                return -1;
            }
        }
        ll num = 0;
        vector<ll> fs;
        for(ll d = 1; d <= D; d++) {
            dist[d] = compute_dist_matrix(d);
            ll tmp = compute_score_per_day(d);
            num += tmp;
            fs.push_back(tmp / (N * (N - 1) / 2));
        }
        // if(penalty_count > 0) cerr << "Warning: penalty_count > 0" << endl;
        ll den = D * N * (N - 1) / 2;
        ld avg = (ld)num / den * 1000.0;
        // if(penalty_count > 0) cerr << "Error: " << penalty_count << endl;
        return round(avg);
    }

};






int main() {
    clock_t start = clock();
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

    // 初期状態のansを作成
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

        unconstructed_edge = random_sample(unconstructed_edge, min((ll)unconstructed_edge.size(), K - 1));
        for(ll i = 0; i < unconstructed_edge.size(); i++) {
            unconstructed_edge[i]->road_status = Constructed;
            ans[unconstructed_edge[i]->edge_id] = d + 1;
        }
    }

    vector<Edge> tmp_vec;
    for(ll i = 0; i < M; i++) {
        if(ans[edge_list[i].edge_id] == 0) {
            ans[edge_list[i].edge_id] = D;
            tmp_vec.push_back(edge_list[i]);
        }
    }


    Score score = Score(ans, edge_list);

    while(true) {
        clock_t end = clock();
        if((ld)(end - start) / CLOCKS_PER_SEC > 3.0) break;
        vector<ll> cnt(D, 0);
        for(ll j = 0; j < M; j++) {
            cnt[score.ans[j] - 1]++;
        }
        
        if(tmp_vec.size() > 0) {
            ll tmp_idx = rand()%tmp_vec.size();
            vector<ll> selection;
            for(ll j = 0; j < D; j++) {
                if(ans[tmp_vec[tmp_idx].edge_id] == j + 1) continue;
                for(ll k = 0; k < K - cnt[j]; k++) {
                    selection.push_back(j + 1);
                }
            }
            // for(ll i = 0; i < selection.size(); i++) cerr << selection[i] << " \n"[i == selection.size() - 1];
            ll ret = score.update_construction_day(tmp_vec[tmp_idx], selection[rand()%selection.size()]);
            // cerr << ret << endl;
            if(ret == 0) {
                selection.erase(selection.begin() + tmp_idx);
            }
        }else {
            vector<ll> selection;
            for(ll j = 0; j < D; j++) {
                for(ll k = 0; k < K - cnt[j]; k++) {
                    selection.push_back(j + 1);
                }
            }
            score.update_construction_day(edge_list[rand()%edge_list.size()], selection[rand()%selection.size()]);
        }
        
    }

    for(ll i = 0; i < M; i++) {
        cout << score.ans[i] << " \n"[i == M - 1];
    }

    
    cerr << score.compute_score() << endl;

}