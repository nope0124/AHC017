## やったこと

### 1/28
 - 完全ランダム
 - 工事をできるだけ分けたほうがいいので、工事回数を $ceil(M/D)$ にしておく
 - ローカル実行(0000.txt-0099.txt)スコア合計252,320,706,119、この時点で14B
<br>

 - 全域木を作りながら最大まで工事回数を消費する
 - $10^9$のペナルティを出来る限り回避する
 - ローカル実行(0000.txt-0099.txt)~~スコア合計3,256,265,636~~(バグがありました、何で通ったんだ)、この時点で28B、最終日の上位4割がこの辺になりそう
 - Pythonのrandom.sampleに当たる操作を多用しているから纏めたほうがいいかも
 <br>
 
 ### 1/29
  - ローカル実行(0000.txt-0999.txt)で $10^9$ のペナルティを受けてないことを確認
  - dayによってスコアは独立している→スコアの差分更新できそう
  - 恐らく1dayあたり100msかかるから高速化が鍵？
  - (卒論を修正する→しました)
  - ローカル実行(0000.txt-0099.txt)スコア合計6,003,942,654(一旦)
  - デバッグしました
  - ワーシャルフロイドで辺追加の差分更新だけ $O(N^2)$ でできるようになった
  - ローカル実行(0000.txt-0099.txt)スコア合計3,239,209,934、この時点で23B、ほとんど変化なし
  - リモートスコア1Bを切らないときつそう
  - あと20B-30Bあたりが団子状態になってる
  - 
 
## 次やること
 - とにかく綺麗なコードを心掛けよう
 - LowLinkで差分更新の候補を絞る
 - AnsInfoの構造体を作る
 - 辺削除の高速化も調べる
