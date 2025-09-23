import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io # ダウンロード機能用

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="高機能 線形回帰分析アプリ")

# --- タイトル ---
st.title('📈 高機能 線形回帰分析アプリ')
st.markdown('CSVをアップロードするだけで、詳細な回帰分析と予測ができます。')

# --- サイドバー ---
with st.sidebar:
    st.header("1. ファイルアップロード")
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
    
    st.markdown("---")
    st.header("2. 変数選択")
    # ファイルがアップロードされてから選択肢を表示
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        x_column = st.selectbox('説明変数（X軸）を選択してください', columns)
        y_column = st.selectbox('目的変数（Y軸）を選択してください', columns)

    st.markdown('---')
    st.header("3. グラフの見た目設定")
    plot_style = st.selectbox('グラフのスタイル',('darkgrid', 'whitegrid', 'dark', 'white', 'ticks'), index=1)
    scatter_color = st.color_picker('散布図の点の色', '#1f77b4')
    line_color = st.color_picker('回帰直線の色', '#ff7f0e')
    scatter_size = st.slider('点のサイズ', 10, 200, 50)


# --- メイン画面 ---
if uploaded_file is not None:
    # データ型のチェックと変換
    try:
        # 数値型に変換できない値をNaNにする
        df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        # NaNを含む行を削除
        df.dropna(subset=[x_column, y_column], inplace=True)
        
        x_data = df[x_column]
        y_data = df[y_column]
        
        if x_data.empty or y_data.empty:
            st.error("選択された列に有効な数値データがありません。")
        else:
            # --- モデル作成と分析 ---
            X = sm.add_constant(x_data)
            model = sm.OLS(y_data, X)
            results = model.fit()
            y_pred = results.predict(X)

            # --- レイアウト設定 ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader('📈 散布図と回帰直線')
                sns.set_style(plot_style)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=x_data, y=y_data, ax=ax, label='実測データ', color=scatter_color, s=scatter_size)
                ax.plot(x_data, y_pred, color=line_color, linewidth=2, label='回帰直線')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.legend()
                st.pyplot(fig)
                
                # グラフダウンロード機能
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label="グラフをダウンロード (PNG)",
                    data=buf,
                    file_name=f"regression_plot_{y_column}_vs_{x_column}.png",
                    mime="image/png"
                )


            with col2:
                st.subheader('📊 分析結果サマリー')
                st.text(results.summary())

                st.subheader('💡 結果のポイント解説')
                coef_summary = results.summary2().tables[1]
                st.write(f"**傾き (coef):** {coef_summary['Coef.'][1]:.4f}")
                st.write(f"**切片 (intercept):** {coef_summary['Coef.'][0]:.4f}")
                
                st.markdown("---")
                
                st.subheader('モデルの評価')
                st.write(f"**決定係数 (R-squared):** {results.rsquared:.4f}")

                with st.expander("📝 各指標の簡単な説明を見る"):
                    st.markdown("""
                    - **R-squared (決定係数):** 1に近いほど、モデルが実際のデータをうまく説明できていることを示します。
                    - **coef (係数):** Xが1増えた時に、Yがどれだけ増えるか(傾き)を示します。
                    - **P>|t| (P値):** 係数が「偶然そうなっただけ」という可能性です。一般的に0.05より小さいと「統計的に意味のある関係」と判断します。
                    - **[0.025, 0.975]:** 信頼区間。95%の確率で本当の係数がこの範囲にあることを示します。
                    """)

            # --- 未来の予測 ---
            st.markdown("---")
            st.subheader(f'🚀 「{x_column}」の値から「{y_column}」を予測')
            new_x_value = st.number_input(f'予測したい「{x_column}」の値を入力してください', format="%.4f")
            
            if st.button('予測する'):
                new_x_with_const = sm.add_constant([new_x_value])
                prediction = results.predict(new_x_with_const)
                st.success(f'予測結果: **{prediction[0]:.4f}**')
    
    except Exception as e:
        st.error(f"エラーが発生しました。選択した列が数値データか確認してください。エラー詳細: {e}")

else:
    st.info('サイドバーからCSVファイルをアップロードしてください。')