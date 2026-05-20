import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    layout="wide"
)

st.title("📈 Retail Demand Forecasting KPI Dashboard")
st.markdown("AI-Powered Retail Sales Analytics & Forecasting")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("csv/train.csv/train.csv")
    store = pd.read_csv("E:\Sales Forcasting\csv\store.csv")

    # Merge datasets
    df = train.merge(store, on="Store", how="left")

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Remove closed stores
    df = df[df["Open"] == 1]

    return df


df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

store_filter = st.sidebar.multiselect(
    "Select Store",
    options=df["Store"].unique(),
    default=df["Store"].unique()[:10]
)

promo_filter = st.sidebar.selectbox(
    "Promo",
    ["All", "Promo", "No Promo"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["Date"].min(), df["Date"].max()]
)

# Apply filters
filtered_df = df[df["Store"].isin(store_filter)]

if promo_filter == "Promo":
    filtered_df = filtered_df[filtered_df["Promo"] == 1]
elif promo_filter == "No Promo":
    filtered_df = filtered_df[filtered_df["Promo"] == 0]

filtered_df = filtered_df[
    (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
    (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
]

# -----------------------------
# KPI Metrics
# -----------------------------
total_sales = filtered_df["Sales"].sum()
avg_sales = filtered_df["Sales"].mean()
total_customers = filtered_df["Customers"].sum()
num_stores = filtered_df["Store"].nunique()
promo_sales = filtered_df[
    filtered_df["Promo"] == 1
]["Sales"].sum()

promo_percentage = (
    promo_sales / total_sales * 100
    if total_sales > 0 else 0
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "💰 Total Sales",
    f"₹ {total_sales:,.0f}"
)

col2.metric(
    "📊 Avg Sales",
    f"₹ {avg_sales:,.0f}"
)

col3.metric(
    "🛒 Customers",
    f"{total_customers:,.0f}"
)

col4.metric(
    "🏪 Stores",
    num_stores
)

col5.metric(
    "🔥 Promo Sales %",
    f"{promo_percentage:.1f}%"
)

st.markdown("---")

# -----------------------------
# Sales Trend
# -----------------------------
daily_sales = filtered_df.groupby(
    "Date"
)["Sales"].sum().reset_index()

fig_sales = px.line(
    daily_sales,
    x="Date",
    y="Sales",
    title="Daily Sales Trend"
)

st.plotly_chart(fig_sales, use_container_width=True)

# -----------------------------
# Monthly Sales Trend
# -----------------------------
filtered_df["Month"] = filtered_df["Date"].dt.to_period("M").astype(str)

monthly_sales = filtered_df.groupby(
    "Month"
)["Sales"].sum().reset_index()

fig_monthly = px.bar(
    monthly_sales,
    x="Month",
    y="Sales",
    title="Monthly Demand Trend"
)

st.plotly_chart(fig_monthly, use_container_width=True)

# -----------------------------
# Store Type Performance
# -----------------------------
store_sales = filtered_df.groupby(
    "StoreType"
)["Sales"].sum().reset_index()

fig_store = px.pie(
    store_sales,
    names="StoreType",
    values="Sales",
    title="Sales by Store Type"
)

st.plotly_chart(fig_store, use_container_width=True)

# -----------------------------
# Top Performing Stores
# -----------------------------
top_stores = (
    filtered_df.groupby("Store")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_top = px.bar(
    top_stores,
    x="Store",
    y="Sales",
    title="Top 10 Performing Stores"
)

st.plotly_chart(fig_top, use_container_width=True)

# -----------------------------
# Promo vs Non Promo
# -----------------------------
promo_analysis = filtered_df.groupby(
    "Promo"
)["Sales"].mean().reset_index()

promo_analysis["Promo"] = (
    promo_analysis["Promo"]
    .map({0: "No Promo", 1: "Promo"})
)

fig_promo = px.bar(
    promo_analysis,
    x="Promo",
    y="Sales",
    title="Promo vs Non-Promo Sales"
)

st.plotly_chart(fig_promo, use_container_width=True)

# -----------------------------
# Forecasting Section
# -----------------------------
st.subheader("📈 Retail Demand Forecast")

forecast_df = daily_sales.rename(
    columns={"Date": "ds", "Sales": "y"}
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(forecast_df)

future = model.make_future_dataframe(
    periods=30
)

forecast = model.predict(future)

fig_forecast = px.line(
    forecast,
    x="ds",
    y="yhat",
    title="30-Day Sales Forecast"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# -----------------------------
# Forecast Table
# -----------------------------
st.subheader("Forecasted Demand")

forecast_table = forecast[
    ["ds", "yhat"]
].tail(30)

forecast_table.columns = [
    "Date",
    "Predicted Sales"
]

st.dataframe(forecast_table)

st.success("Dashboard Loaded Successfully!")