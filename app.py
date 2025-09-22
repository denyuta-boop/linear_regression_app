import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# === ここから追加 ===
# iframeを使ってads.htmlを埋め込む
st.markdown('<iframe src="ads.html" width="100%" height="200px" style="border:0;"</iframe>', unsafe_allow_html=True)
# === ここまで追加 ===

# 以下、あなたのアプリの元のコード
st.title('CSVから線形回帰分析を行うアプリ')
st.markdown('CSVファイルをアップロードすると、線形回帰モデルを作成し、結果を可視化します。')

# ファイルアップローダーを設置
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    # CSVファイルを読み込む
    df = pd.read_csv(uploaded_file)
    st.write("アップロードされたデータ:")
    st.write(df.head())

    # 列名を取得
    columns = df.columns.tolist()

    # ユーザーに説明変数（x）と目的変数（y）を選択させる
    st.sidebar.header('分析設定')
    x_column = st.sidebar.selectbox('説明変数（X軸）を選択してください', columns)
    y_column = st.sidebar.selectbox('目的変数（Y軸）を選択してください', columns)

    # === ここから新しいコードを追加 ===
    st.sidebar.markdown('---')
    st.sidebar.header('グラフ設定')
    # 軸の色を選ぶカラーピッカー
    axis_color = st.sidebar.color_picker('軸とラベルの色を選んでください', '#000000')

    # matplotlibとseabornの軸の色設定を更新
    plt.rc('axes', edgecolor=axis_color) # 軸の枠線の色
    sns.set(rc={'axes.edgecolor': axis_color, 'xtick.color': axis_color, 'ytick.color': axis_color})
    
    # === ここまで新しいコードを追加 ===

    # 選択された列を数値型に変換
    try:
        x_data = df[x_column].values.reshape(-1, 1)
        y_data = df[y_column].values
    except Exception as e:
        st.error(f"選択した列に数値データが含まれていない可能性があります。別の列を選択してください。エラー: {e}")
        st.stop()

    # 線形回帰モデルを作成
    model = LinearRegression()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)

    # グラフを可視化
    st.subheader('線形回帰分析の結果')

    # Matplotlibの図（Figure）を作成
    fig, ax = plt.subplots(figsize=(10, 6))

    # 散布図をプロット
    sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax, label='実測データ')

    # 回帰直線をプロット
    ax.plot(df[x_column], y_pred, color='red', linewidth=2, label='回帰直線')

    # グラフのタイトルとラベルを設定
    ax.set_title(f'線形回帰 ({y_column} vs {x_column})', color=axis_color) # タイトルの色
    ax.set_xlabel(x_column, color=axis_color) # X軸ラベルの色
    ax.set_ylabel(y_column, color=axis_color) # Y軸ラベルの色
    ax.legend()
    st.pyplot(fig)

    # モデルの係数を表示
    st.subheader('モデルの係数')
    st.write(f'傾き (coef_): **{model.coef_[0]:.4f}**')
    st.write(f'切片 (intercept_): **{model.intercept_:.4f}**')

    # R^2スコアを表示
    st.subheader('モデルの評価')
    st.write(f'決定係数（R^2）: **{model.score(x_data, y_data):.4f}**')
    st.info('決定係数が1に近いほど、モデルの当てはまりが良いことを示します。')