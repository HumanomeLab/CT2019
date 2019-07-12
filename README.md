# CT検診学会 2019年夏期セミナー講演資料

* [スライド](https://github.com/HumanomeLab/CT2019/blob/master/20190713_CT.pdf)
* [Jupyter Notebook](https://github.com/HumanomeLab/CT2019/blob/master/CT2019_screening.ipynb)
  * 文章の記載は Google Colaboratory で読み込んだ場合を想定した記載です。

---
## この資料について

この資料は、[CT検診学会 2019年夏期セミナー](https://www.jscts.org/index.php?page=seminar_index)の講演のために作成した資料です。
データは [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)で公開されている肺炎の方と肺炎でない方のX線画像を利用しています（事例がCTでなくて、すみません。。。） 

多くのAIに関連した入門演習資料は、AIプログラムの開発に重きを置いていますが、実際の開発では、プログラムの作成後に行われる「**育てる**」フェーズも重要です。育てるフェーズでは

* モデル：学習時のハイパーパラメータをチューニングする
* データ：「必要な注釈を付ける」「利用するデータを選別・加工する」

などが実施されます。この演習では、プログラム開発自身ではなく、これらのハイパーパラメータを触ることや、データの加工で遊んでみることを中心にして組んであります。

発表時利用の資料は以下のとおりです。

* [スライド](https://github.com/HumanomeLab/CT2019/blob/master/20190713_CT.pdf)に詳細の内容が書かれています。
* 上記スライドの演習項目にある通り、Google Colaboratory の画面で、このGitHubを指定した後、指示に従って [CT2019.zip](https://github.com/HumanomeLab/CT2019/blob/master/CT2019.zip) をアップロードして演習を開始してください。
* アノテーションツールに[VoTT](https://github.com/Microsoft/VoTT/releases)を利用しています。

## 免責事項

全ての記述は 2019年7月12日現在の情報を元にしており、各ツール、頻繁にアップデートされるので、その時の状況に応じて読み替えてください。
また、本資料によって不具合等が起きても、何らかの保証をするものではありません。自己責任でお願いいたします。
