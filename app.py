import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta
import streamlit as st

# Ліва бічна панель
st.sidebar.title('🔧 Налаштування')
period_options = {
    '1 рік': '1y',
    '2 роки': '2y',
    '5 років': '5y'
}
selected_period_label = st.sidebar.radio('Оберіть період:', list(period_options.keys()))
selected_period = period_options[selected_period_label]
show_forecast = st.sidebar.checkbox('Показати прогноз', value=False)

# Верхня панель
with st.container():
    st.markdown(
        """
        <div style='
            background-color: #f0f2f6;
            padding: 15px 25px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        '>
            <h2 style='margin: 0; color: #336699;'>📊 Лінійна регресія Прогнозування акцій</h2>
            <div style='flex: 1; text-align: right; font-size: 14px; color: #666;'>
                Оберіть компанії нижче ⬇️
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# Вибір компаній
company_options = {
    'Apple (AAPL)': 'AAPL',
    'Microsoft (MSFT)': 'MSFT',
    'Tesla (TSLA)': 'TSLA',
    'Amazon (AMZN)': 'AMZN',
    'Google (GOOGL)': 'GOOGL',
    'Meta (META)': 'META',
    'NVIDIA (NVDA)': 'NVDA',
    'Netflix (NFLX)': 'NFLX'
}
selected = st.multiselect(
    'Оберіть компанії для перегляду:',
    options=list(company_options.keys())
)

forecast_figs = []
volume_latest = {}

if selected:
    symbols = [company_options[name] for name in selected]
    close_prices = pd.DataFrame()

    for symbol in symbols:
        stock_data = yf.download(symbol, period=selected_period)
        st.subheader(f'📌 Дані для {symbol}')
        if not stock_data.empty:
            available_columns = stock_data.columns.tolist()
            columns_to_show = st.multiselect(
                f'Оберіть колонки для {symbol}:',
                options=available_columns,
                default=available_columns
            )
            st.write('📊 Початкові фінансові дані', stock_data[columns_to_show].head())
            close_prices[symbol] = stock_data['Close']

            last_volume = stock_data['Volume'].dropna().iloc[-1]
            volume_latest[symbol] = last_volume

            df = stock_data[['Close']].copy().dropna()
            df.index = pd.to_datetime(df.index)
            last_2_months = df.last('60D')

            try:
                # Лінійна регресія
                last_n = 60
                df_lr = df.tail(last_n)
                X = (df_lr.index - df_lr.index[0]).days.values.reshape(-1, 1)
                y = df_lr['Close'].values
                model = LinearRegression()
                model.fit(X, y)

                # Прогноз на 7 днів вперед
                future_days = 7
                last_day = (df_lr.index[-1] - df_lr.index[0]).days
                X_future = [[last_day + i + 1] for i in range(future_days)]
                forecast = model.predict(X_future)
                future_dates = [df_lr.index[-1] + timedelta(days=i + 1) for i in range(future_days)]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(last_2_months.index, last_2_months['Close'], label='Останні 2 місяці')
                ax.plot(future_dates, forecast, label='Прогноз (7 днів)', color='red')
                ax.set_title(f'Прогноз ціни для {symbol} (лінійна регресія)')
                ax.set_xlabel('Дата')
                ax.set_ylabel('Ціна')
                ax.legend()
                plt.xticks(rotation=45)

                forecast_figs.append((symbol, fig, forecast, future_dates))

            except Exception as e:
                st.error(f'❌ Помилка при побудові моделі лінійної регресії для {symbol}: {e}')
        else:
            st.error(f'⚠️ Дані не знайдено для {symbol}.')

    if not close_prices.empty:
        st.subheader("📈 Динаміка цін")
        st.line_chart(close_prices)

        last_close_prices = close_prices.iloc[-1]
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(
            last_close_prices,
            labels=last_close_prices.index,
            autopct='%1.1f%%',
            startangle=90
        )
        ax_pie.set_title('Відношення цін акцій на останній день')
        st.pyplot(fig_pie)

        if show_forecast:
            st.subheader('🔮 Прогнози')
            for symbol, fig, forecast, future_dates in forecast_figs:
                st.pyplot(fig)

                forecast_df = pd.DataFrame({
                    'Дата': pd.to_datetime(future_dates),
                    'Прогнозована ціна': forecast.flatten()  # <--- Виправлення тут
                })
                forecast_df.set_index('Дата', inplace=True)
                st.write(f'📄 Таблиця прогнозу для {symbol}:')
                st.dataframe(forecast_df)
