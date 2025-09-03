# markowitz_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title=" Green Markowitz Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# SIDEBAR - SETTINGS
# -------------------------------
st.sidebar.header("Portfolio Settings")

all_assets = ["PG", "^GSPC", "AAPL", "MSFT", "GOOG"]
assets = st.sidebar.multiselect("Select Assets:", all_assets, default=["PG", "^GSPC"])

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

num_portfolios = st.sidebar.slider("Number of Random Portfolios", 100, 5000, 1000)
rf_rate = st.sidebar.slider("Risk-free Rate (%)", 0.0, 5.0, 0.5)

# -------------------------------
# TITLE
# -------------------------------
st.title("Markowitz Portfolio Optimization Dashboard")

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
if assets:
    # Download data
    pf_data = yf.download(assets, start=start_date, end=end_date, auto_adjust=True)
    adj_close = pf_data["Close"]
    normalized = adj_close / adj_close.iloc[0] * 100

    # -------------------------------
    # Layout: Price Performance
    # -------------------------------
    st.subheader("Price Performance")
    st.line_chart(normalized, use_container_width=True)

    # -------------------------------
    # Portfolio simulation
    # -------------------------------
    returns = adj_close.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    portfolio_results = np.zeros((num_portfolios, 3))
    weight_array = []
    for i in range(num_portfolios):
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)
        weight_array.append(weights)
        port_return = np.sum(weights * mean_returns) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (port_return - rf_rate/100) / port_volatility
        portfolio_results[i] = [port_volatility, port_return, sharpe_ratio]

    portfolios = pd.DataFrame(portfolio_results, columns=["Volatility", "Return", "Sharpe"])
    portfolios["Weights"] = weight_array

    # -------------------------------
    # Layout: Tabs
    # -------------------------------
    tab1, tab2, tab3 = st.tabs(["2D Efficient Frontier", "Portfolio Data", "3D Efficient Frontier"])
    
    with tab1:
        st.subheader("2D Efficient Frontier")
        fig, ax = plt.subplots(figsize=(10,6))
        scatter = ax.scatter(
            portfolios["Volatility"], portfolios["Return"],
            c=portfolios["Sharpe"], cmap="Greens", alpha=0.8
        )
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        ax.set_title("Efficient Frontier (color = Sharpe Ratio)", fontsize=14)
        cbar = fig.colorbar(scatter)
        cbar.set_label("Sharpe Ratio")
        st.pyplot(fig)

        st.markdown("""
         
        - Darker green points indicate higher Sharpe Ratio (better risk-adjusted return).  
        - Lighter green points indicate lower Sharpe Ratio.
        """)

    with tab2:
        st.subheader("Simulated Portfolio Data")
        st.dataframe(portfolios.style.background_gradient(cmap="Greens", axis=0))

    with tab3:
        st.subheader("3D Efficient Frontier")
        max_sharpe_idx = portfolios["Sharpe"].idxmax()
        max_sharpe_portfolio = portfolios.loc[max_sharpe_idx]

        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(
            x=portfolios["Volatility"],
            y=portfolios["Return"],
            z=portfolios["Sharpe"],
            mode='markers',
            marker=dict(size=5, color=portfolios["Sharpe"], colorscale='Greens', opacity=0.8),
            name='Portfolios'
        ))

        # Highlight Max Sharpe Portfolio
        fig_3d.add_trace(go.Scatter3d(
            x=[max_sharpe_portfolio["Volatility"]],
            y=[max_sharpe_portfolio["Return"]],
            z=[max_sharpe_portfolio["Sharpe"]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Max Sharpe'
        ))

        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Volatility',
                yaxis_title='Return',
                zaxis_title='Sharpe Ratio'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("""
          
        - The 3D plot shows Volatility (X), Return (Y), and Sharpe Ratio (Z).  
        - Green shades indicate risk-adjusted performance. Darker green = better Sharpe.  
        - Red diamond marks the Max Sharpe portfolio.
        """)
