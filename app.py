import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io
import platform
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_hex

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Advanced Linear Regression App")

# --- User-Selected Palettes ---
PALETTE_LIST = [
    "Accent", "Accent_r", "BrBG", "CMRmap", "CMRmap_r", "Dark2", "Dark2_r",
    "Greys", "Greys_r", "PRGn", "PRGn_r", "Paired", "Paired_r", "Pastel1",
    "Pastel2", "PiYG", "PiYG_r", "RdBu", "RdBu_r", "Set2", "Set2_r",
    "Set3", "Set3_r", "Spectral", "Spectral_r"
]
# --- End User-Selected Palettes ---

# --- App Title and Description ---
st.title('📈 Advanced Linear Regression App')
st.markdown("Upload a CSV file to perform a detailed regression analysis and make predictions.")

# --- Sidebar ---
with st.sidebar:
    st.header("🎨 General Theme")
    palette_choice = st.selectbox(
        "Select a color palette",
        PALETTE_LIST, # Use your curated list
        help="Select a color scheme. The colors below will update automatically."
    )
    palette = sns.color_palette(palette_choice)
    
    st.header("1. File Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    st.markdown("---")
    st.header("2. Variable Selection")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        x_column = st.selectbox('Select the X-axis (Independent Var.)', columns)
        y_column = st.selectbox('Select the Y-axis (Dependent Var.)', columns)

    st.markdown('---')
    st.header("3. Plot Customization")
    
    st.subheader("General Design")
    plot_context = st.selectbox('Plot Context', ('notebook', 'paper', 'talk', 'poster'))
    plot_style = st.selectbox('Plot Style', ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks'), index=1)
    face_color = st.color_picker('Plot Background Color', '#FFFFFF')

    st.subheader("Scatter Point Elements")
    marker_options = {
        "Circle ('o')": "o", "Point ('.')": ".", "Pixel (',')": ",", 
        "Triangle Up ('^')": "^", "Triangle Down ('v')": "v", 
        "Triangle Left ('<')": "<", "Triangle Right ('>')": ">",
        "Square ('s')": "s", "Diamond ('D')": "D", "Thin Diamond ('d')": "d",
        "Plus ('+')": "+", "Cross ('x')": "x", "Star ('*')": "*",
    }
    marker_style_key = st.selectbox("Point Style (Marker)", list(marker_options.keys()))
    marker_style = marker_options[marker_style_key]

    scatter_fill_color = st.color_picker('Point Fill Color', value=to_hex(palette[0]))
    scatter_edge_color = st.color_picker('Point Edge Color', value=to_hex(palette[3]))
    scatter_edge_width = st.slider('Point Edge Width', 0.0, 5.0, 1.0, 0.25)
    scatter_size = st.slider('Point Size', 10, 400, 50)
    
    st.subheader("Regression Line Elements")
    line_color = st.color_picker('Regression Line Color', value=to_hex(palette[1]))
    line_width = st.slider('Line Width', 0.5, 10.0, 2.0, 0.5)
    
    linestyle_options = {
        "Solid (-)": "-", "Dashed (--)": "--",
        "Dotted (:)": ":", "Dash-dot (-.)": "-."
    }
    line_style_key = st.selectbox("Line Style", list(linestyle_options.keys()))
    line_style = linestyle_options[line_style_key]

    st.subheader("Font Sizes")
    title_fontsize = st.slider('Title Font Size', 8, 40, 18)
    label_fontsize = st.slider('Axis Label Font Size', 6, 30, 12)
    legend_fontsize = st.slider('Legend Font Size', 6, 30, 10)
    tick_fontsize = st.slider('Tick Label Font Size', 5, 25, 10)

# --- OS-dependent Font Setup ---
font_options, legend_options, title_options = {}, {}, {}
try:
    if platform.system() == 'Windows':
        font_path = r'C:\Windows\Fonts\meiryo.ttc'
        jp_title_font = FontProperties(fname=font_path, size=title_fontsize)
        jp_label_font = FontProperties(fname=font_path, size=label_fontsize)
        jp_legend_font = FontProperties(fname=font_path, size=legend_fontsize)
        title_options, font_options, legend_options = {'fontproperties': jp_title_font}, {'fontproperties': jp_label_font}, {'prop': jp_legend_font}
    else:
        import japanize_matplotlib
        title_options, font_options, legend_options = {'fontsize': title_fontsize}, {'fontsize': label_fontsize}, {'fontsize': legend_fontsize}
except Exception:
    title_options, font_options, legend_options = {'fontsize': title_fontsize}, {'fontsize': label_fontsize}, {'fontsize': legend_fontsize}


# --- Main Content ---
if uploaded_file is not None:
    try:
        df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        df.dropna(subset=[x_column, y_column], inplace=True)
        x_data = df[x_column]
        y_data = df[y_column]
        
        if x_data.empty or y_data.empty or len(x_data) < 2:
            st.error("The selected columns need at least 2 valid numeric data points to perform analysis.")
        else:
            with st.sidebar:
                st.markdown('---')
                st.header("4. Axis Range (Limits)")
                use_manual_limits = st.checkbox("Set axis limits manually")

                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()

                if use_manual_limits:
                    st.subheader("Manual Limits")
                    col1_limit, col2_limit = st.columns(2)
                    with col1_limit:
                        x_limit_min = st.number_input("X-min", value=x_min)
                        y_limit_min = st.number_input("Y-min", value=y_min)
                    with col2_limit:
                        x_limit_max = st.number_input("X-max", value=x_max)
                        y_limit_max = st.number_input("Y-max", value=y_max)
                    x_limits = (x_limit_min, x_limit_max)
                    y_limits = (y_limit_min, y_limit_max)
                else:
                    st.subheader("Auto-Range Sliders")
                    x_pad = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 1
                    y_pad = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1
                    
                    x_limits = st.slider('X-axis range', min_value=float(x_min - x_pad), max_value=float(x_max + x_pad), value=(float(x_min - x_pad), float(x_max + x_pad)))
                    y_limits = st.slider('Y-axis range', min_value=float(y_min - y_pad), max_value=float(y_max + y_pad), value=(float(y_min - y_pad), float(y_max + y_pad)))

            X = sm.add_constant(x_data)
            model = sm.OLS(y_data, X)
            results = model.fit()
            y_pred = results.predict(X)
            col1_main, col2_main = st.columns([2, 1])

            with col1_main:
                st.subheader('Scatter plot with Regression Line')
                
                sns.set_context(plot_context)
                sns.set_style(plot_style)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_facecolor(face_color)
                
                ax.set_title(f'Relationship between {y_column} and {x_column}', **title_options)
                
                sns.scatterplot(
                    x=x_data, y=y_data, ax=ax, label='Actual Data', 
                    color=scatter_fill_color, s=scatter_size, marker=marker_style,
                    edgecolor=scatter_edge_color, linewidth=scatter_edge_width
                )

                plot_df = pd.DataFrame({'x': x_data, 'y_pred': y_pred})
                sorted_plot_df = plot_df.sort_values(by='x')

                ax.plot(sorted_plot_df['x'], sorted_plot_df['y_pred'], color=line_color, linewidth=line_width, label='Regression Line', linestyle=line_style)
                
                ax.set_xlabel(x_column, **font_options)
                ax.set_ylabel(y_column, **font_options)
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                
                ax.legend(**legend_options)
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(label="Download Plot (PNG)", data=buf, file_name="regression_plot.png", mime="image/png")

            with col2_main:
                st.subheader('Analysis Summary')
                st.text(results.summary())
                st.subheader('Key Metrics')
                
                r_value = x_data.corr(y_data)
                st.write(f"**Correlation Coefficient (r):** {r_value:.4f}")
                
                st.write(f"**R-squared:** {results.rsquared:.4f}")
                st.markdown("---")
                
                coef_summary = results.summary2().tables[1]
                st.write(f"**Coefficient (slope):** {coef_summary['Coef.'][1]:.4f}")
                st.write(f"**Intercept:** {coef_summary['Coef.'][0]:.4f}")

                with st.expander("See detailed explanations of metrics"):
                    st.markdown("""
                    #### **Is the Relationship Real or Just a Coincidence?**
                    Statistical analysis helps us determine if a relationship in the data is real or just a random occurrence.
                    
                    ---
                    **Correlation Coefficient (r)**
                    - **What it is:** It measures the **strength and direction** of a linear relationship between two variables.
                    - **Range:** It goes from -1 to +1.
                    - **How to interpret:**
                        - Close to **+1**: Strong **positive correlation**. As one variable increases, the other also increases.
                        - Close to **-1**: Strong **negative correlation**. As one variable increases, the other decreases.
                        - Close to **0**: Little to no linear correlation.

                    ---
                    **R-squared (Coefficient of Determination)**
                    - **What it is:** It shows how well the regression line **fits the data**.
                    - **Range:** It goes from 0 to 1.
                    - **How to interpret:** An R-squared of 0.7 means that 70% of the variation in the Y-variable can be explained by the X-variable. The closer to 1, the better the model fits the data.

                    ---
                    **P-value (P>|t|)**
                    - **What it asks:** "If there were **no real relationship** between X and Y, what is the probability that we would see a relationship this strong (or stronger) just by **random chance**?"
                    - **How to interpret:**
                        - A **small p-value** (typically < 0.05) suggests: "This is very unlikely to happen by chance. Therefore, the relationship we're seeing is likely real and **statistically significant**."
                        - A **large p-value** (typically > 0.05) suggests: "A relationship this strong could easily happen by random chance. We cannot conclude that a real relationship exists."
                    """)
            
            # Prediction section
            st.markdown("---")
            st.subheader(f'🚀 Predict "{y_column}" from "{x_column}"')
            new_x_value = st.number_input(f'Enter a value for "{x_column}" to predict', format="%.4f")
            if st.button('Predict'):
                prediction_data = [[1, new_x_value]] 
                prediction = results.predict(prediction_data)
                st.success(f'Predicted Result: **{prediction[0]:.4f}**')
    
    except Exception as e:
        st.error(f"An error occurred. Please check if the selected columns contain numeric data. Error: {e}")

else:
    st.info('Please upload a CSV file from the sidebar to get started.')