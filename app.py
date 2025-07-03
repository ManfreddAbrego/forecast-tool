import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from io import BytesIO

# Set up the web page
st.set_page_config(page_title="📊 Accurate Forecast Tool", layout="wide")
st.title("📈 Call Volume & AHT Forecast (Accurate Model)")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Excel file with Date, Calls, AHT", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=1)
    df.columns = ["Date", "Calls", "AHT"]
    df = df[df["Date"].notna()]
    df["Date"] = pd.to_datetime(df["Date"])
    df[["Calls", "AHT"]] = df[["Calls", "AHT"]].apply(pd.to_numeric, errors='coerce')
    
    st.success("✅ File loaded and processed successfully!")

    def run_holt_forecast(df, column_name, seasonal_periods=39, forecast_horizon=270):
        series = df[["Date", column_name]].dropna()
        series.set_index("Date", inplace=True)

        model = ExponentialSmoothing(
            series[column_name],
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods
        )
        fit = model.fit()
        forecast = fit.forecast(forecast_horizon)

        # Plotting
        plt.figure(figsize=(12, 4))
        plt.plot(series, label="Actual", color="blue")
        plt.plot(forecast, label="Forecast", linestyle='--', color="orange")
        plt.title(f"{column_name} Forecast (9 Months)")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        forecast_df = forecast.reset_index()
        forecast_df.columns = ["Date", f"{column_name}_Forecast"]
        return forecast_df

    # Run forecasts
    forecast_calls = run_holt_forecast(df, "Calls")
    forecast_aht = run_holt_forecast(df, "AHT")

    # Merge
    merged = pd.merge(forecast_calls, forecast_aht, on="Date")
    merged["FTE"] = (merged["Calls_Forecast"] * merged["AHT_Forecast"] / 60 / 8).round(2)

    st.line_chart(merged.set_index("Date")[["FTE"]], height=250)

    # Download as Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        forecast_calls.to_excel(writer, sheet_name="Calls Forecast", index=False)
        forecast_aht.to_excel(writer, sheet_name="AHT Forecast", index=False)
        merged.to_excel(writer, sheet_name="Merged Forecast", index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Forecast Data",
        data=output,
        file_name="accurate_forecast_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
