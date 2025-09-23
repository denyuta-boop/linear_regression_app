import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io

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
    try:
        df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        df.dropna(subset=[x_column, y_column], inplace=True)
        x_data = df[x_column]
        y_data = df[y_column]
        
        if x_data.empty or y_data.empty:
            st.error("選択された列に有効な数値データがありません。")
        else:
            X = sm.add_constant(x_data)
            model = sm.OLS(y_data, X)
            results = model.fit()
            y_pred = results.predict(X)
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader('Scatter plot with Regression Line') # 英語表記に変更
                sns.set_style(plot_style)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.set_title(f'Relationship between {y_column} and {x_column}') # 英語表記に変更
                sns.scatterplot(x=x_data, y=y_data, ax=ax, label='Actual Data', color=scatter_color, s=scatter_size)
                ax.plot(x_data, y_pred, color=line_color, linewidth=2, label='Regression Line')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.legend()
                
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label="Download Plot (PNG)",
                    data=buf,
                    file_name=f"regression_plot.png",
                    mime="image/png"
                )

            with col2:
                st.subheader('Analysis Summary') # 英語表記に変更
                st.text(results.summary())
                st.subheader('Key Metrics') # 英語表記に変更
                coef_summary = results.summary2().tables[1]
                st.write(f"**Coefficient (slope):** {coef_summary['Coef.'][1]:.4f}")
                st.write(f"**Intercept:** {coef_summary['Coef.'][0]:.4f}")
                st.markdown("---")
                st.subheader('Model Evaluation') # 英語表記に変更
                st.write(f"**R-squared:** {results.rsquared:.4f}")

                with st.expander("See explanations of metrics"): # 英語表記に変更
                    st.markdown("""
                    - **R-squared:** Indicates how well the model explains the data. Closer to 1 is better.
                    - **Coefficient (slope):** Shows how much Y changes for a one-unit increase in X.
                    - **P>|t| (p-value):** If below 0.05, the relationship is generally considered statistically significant.
                    """)

            st.markdown("---")
            st.subheader(f'🚀 Predict "{y_column}" from "{x_column}"') # 英語表記に変更
            new_x_value = st.number_input(f'Enter a value for "{x_column}" to predict', format="%.4f")
            
            if st.button('Predict'): # 英語表記に変更
                prediction_data = [[1, new_x_value]] 
                prediction = results.predict(prediction_data)
                st.success(f'Predicted Result: **{prediction[0]:.4f}**')
    
    except Exception as e:
        st.error(f"An error occurred. Please check if the selected columns contain numeric data. Error: {e}")

else:
    st.info('Please upload a CSV file from the sidebar to get started.')