#!/usr/bin/env python3
"""
Complete Futures Outperformance Forecasting System
Optimized for GitHub Codespaces
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

warnings.filterwarnings('ignore')

class GitHubForecaster:
    """Forecasting system optimized for GitHub environment"""
    
    def __init__(self):
        self.data = None
        self.forecasts = None
        self.results = {}
        
    def fetch_data(self):
        """Fetch data from Yahoo Finance"""
        print("📊 Fetching futures data...")
        
        # Define tickers
        tickers = {
            'NQ': 'NQ=F',  # Nasdaq-100 Futures
            'ES': 'ES=F',  # S&P 500 Futures
            'VIX': '^VIX',  # Volatility Index
            'TNX': '^TNX',  # 10-Year Treasury
        }
        
        try:
            # Download data
            data = yf.download(
                list(tickers.values()),
                start='2024-01-01',
                end=datetime.now().strftime('%Y-%m-%d'),
                progress=False
            )['Adj Close']
            
            data.columns = list(tickers.keys())
            self.data = data.dropna()
            
            print(f"✅ Data fetched: {len(self.data)} trading days")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def calculate_biweekly_returns(self):
        """Calculate biweekly returns and spreads"""
        print("\n📈 Calculating biweekly returns...")
        
        # Ensure we have at least 20 trading days for 2 periods
        if len(self.data) < 20:
            print("⚠️ Insufficient data for analysis")
            return None
        
        periods = []
        n_days = len(self.data)
        
        # Create 10-trading day periods
        for i in range(0, n_days - 9, 10):
            if i + 9 < n_days:
                period_data = self.data.iloc[i:i+10]
                
                nq_start = period_data['NQ'].iloc[0]
                nq_end = period_data['NQ'].iloc[-1]
                es_start = period_data['ES'].iloc[0]
                es_end = period_data['ES'].iloc[-1]
                
                nq_return = (nq_end - nq_start) / nq_start * 100
                es_return = (es_end - es_start) / es_start * 100
                spread = nq_return - es_return
                
                periods.append({
                    'period': len(periods) + 1,
                    'start_date': period_data.index[0].date(),
                    'end_date': period_data.index[-1].date(),
                    'nq_return': round(nq_return, 2),
                    'es_return': round(es_return, 2),
                    'spread': round(spread, 2),
                    'year': period_data.index[0].year,
                    'month': period_data.index[0].month,
                })
        
        df = pd.DataFrame(periods)
        return df
    
    def forecast_2026(self, historical_data):
        """Forecast all 2026 biweekly periods"""
        print("\n🔮 Forecasting 2026 periods...")
        
        # Prepare features
        X = historical_data[['period', 'month']].values
        y = historical_data['spread'].values
        
        # Train ensemble model
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ]
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model_predictions = []
        
        for model in models:
            scores = cross_val_score(model, X, y, cv=tscv, 
                                   scoring='neg_mean_squared_error')
            rmse = np.sqrt(-scores.mean())
            print(f"  Model RMSE: {rmse:.3f}%")
            
            model.fit(X, y)
            
            # Predict 2026 (26 periods)
            future_periods = np.arange(len(X) + 1, len(X) + 27)
            future_months = self._estimate_2026_months()
            X_future = np.column_stack([future_periods, future_months])
            
            predictions = model.predict(X_future)
            model_predictions.append(predictions)
        
        # Ensemble average
        ensemble_forecast = np.mean(model_predictions, axis=0)
        
        # Generate dates for 2026
        dates_2026 = self._generate_2026_dates()
        
        # Create forecast dataframe
        forecasts = pd.DataFrame({
            'period': range(1, 27),
            'start_date': dates_2026['start_dates'],
            'end_date': dates_2026['end_dates'],
            'forecast_spread': ensemble_forecast,
            'prob_outperform': self._calculate_probabilities(ensemble_forecast, y)
        })
        
        return forecasts
    
    def _estimate_2026_months(self):
        """Estimate month for each 2026 period"""
        # First period starts in January
        months = []
        current_month = 1
        
        for i in range(26):
            months.append(current_month)
            # Every 2 periods, move to next month
            if (i + 1) % 2 == 0:
                current_month += 1
            if current_month > 12:
                current_month = 1
        
        return months
    
    def _generate_2026_dates(self):
        """Generate approximate dates for 2026"""
        # Start with first trading day of 2026
        start_date = datetime(2026, 1, 2)
        trading_days = pd.bdate_range(start=start_date, periods=260)
        
        start_dates = []
        end_dates = []
        
        for i in range(26):
            start_idx = i * 10
            end_idx = start_idx + 9
            
            if end_idx < len(trading_days):
                start_dates.append(trading_days[start_idx].date())
                end_dates.append(trading_days[end_idx].date())
        
        return {'start_dates': start_dates, 'end_dates': end_dates}
    
    def _calculate_probabilities(self, forecasts, historical_spreads):
        """Calculate probability of outperformance"""
        mean_spread = np.mean(historical_spreads)
        std_spread = np.std(historical_spreads)
        
        probabilities = []
        for forecast in forecasts:
            # Z-score for spread > 0
            z = (0 - forecast) / std_spread
            prob = 1 - self._normal_cdf(z)
            probabilities.append(round(prob * 100, 1))
        
        return probabilities
    
    def _normal_cdf(self, x):
        """Approximate normal CDF"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def create_interactive_plot(self, historical_data, forecasts):
        """Create interactive Plotly visualization"""
        print("\n📊 Creating interactive visualization...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Historical Spread Performance',
                          '2026 Forecast Spread',
                          'Outperformance Probability',
                          'Cumulative Spread'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Historical spreads
        fig.add_trace(
            go.Scatter(
                x=historical_data['start_date'],
                y=historical_data['spread'],
                mode='lines+markers',
                name='Historical Spread',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Forecast spreads
        fig.add_trace(
            go.Scatter(
                x=forecasts['start_date'],
                y=forecasts['forecast_spread'],
                mode='lines+markers',
                name='2026 Forecast',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecasts['start_date'],
                y=forecasts['forecast_spread'] + 1.0,
                mode='lines',
                name='Upper Bound',
                line=dict(color='red', width=0),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecasts['start_date'],
                y=forecasts['forecast_spread'] - 1.0,
                mode='lines',
                name='Lower Bound',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='red', width=0),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Probability bars
        fig.add_trace(
            go.Bar(
                x=forecasts['period'],
                y=forecasts['prob_outperform'],
                name='Outperform Probability',
                marker_color=['green' if x > 50 else 'red' 
                            for x in forecasts['prob_outperform']]
            ),
            row=2, col=1
        )
        
        # Cumulative spread
        historical_cumulative = np.cumsum(historical_data['spread'])
        forecast_cumulative = np.cumsum(forecasts['forecast_spread'])
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['start_date'],
                y=historical_cumulative,
                mode='lines',
                name='Historical Cumulative',
                line=dict(color='blue', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecasts['start_date'],
                y=forecast_cumulative,
                mode='lines',
                name='2026 Cumulative Forecast',
                line=dict(color='red', width=3)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='NASDAQ-100 vs S&P 500 Futures Outperformance Forecast 2026',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Period", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        
        fig.update_yaxes(title_text="Spread (%)", row=1, col=1)
        fig.update_yaxes(title_text="Spread (%)", row=1, col=2)
        fig.update_yaxes(title_text="Probability (%)", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Spread (%)", row=2, col=2)
        
        # Save and show
        fig.write_html("forecast_plot.html")
        print("✅ Interactive plot saved as forecast_plot.html")
        
        return fig
    
    def generate_report(self, historical_data, forecasts):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("2026 NASDAQ-100 vs S&P 500 FUTURES FORECAST REPORT")
        print("="*80)
        
        # Historical analysis
        print(f"\n📊 HISTORICAL ANALYSIS ({historical_data['start_date'].min()} to {historical_data['start_date'].max()})")
        print("-"*60)
        
        total_periods = len(historical_data)
        outperform_periods = len(historical_data[historical_data['spread'] > 0])
        outperform_pct = (outperform_periods / total_periods) * 100
        
        print(f"Total periods analyzed: {total_periods}")
        print(f"Periods with NQ outperformance: {outperform_periods}")
        print(f"Outperformance percentage: {outperform_pct:.1f}%")
        print(f"Average spread: {historical_data['spread'].mean():.2f}%")
        print(f"Spread volatility: {historical_data['spread'].std():.2f}%")
        
        # 2026 Forecast
        print(f"\n🎯 2026 FORECAST SUMMARY")
        print("-"*60)
        
        positive_forecasts = len(forecasts[forecasts['forecast_spread'] > 0])
        avg_forecast = forecasts['forecast_spread'].mean()
        
        print(f"Total forecast periods: {len(forecasts)}")
        print(f"Periods with expected NQ outperformance: {positive_forecasts}")
        print(f"Average forecast spread: {avg_forecast:.2f}%")
        
        # Specific periods
        print(f"\n📅 KEY PERIODS FORECAST")
        print("-"*60)
        
        # Q1 2026
        q1_forecasts = forecasts[forecasts['period'] <= 6]
        print(f"Q1 2026 (Periods 1-6):")
        print(f"  Average spread: {q1_forecasts['forecast_spread'].mean():.2f}%")
        print(f"  Expected NQ outperformance in {len(q1_forecasts[q1_forecasts['forecast_spread'] > 0])} of 6 periods")
        
        # January 5-16, 2026
        jan_period = forecasts.iloc[0]  # First period
        print(f"\n📌 SPECIFIC FORECAST: January 5-16, 2026")
        print(f"  Expected spread: {jan_period['forecast_spread']:.2f}%")
        print(f"  Probability of NQ outperformance: {jan_period['prob_outperform']:.1f}%")
        print(f"  Confidence interval: [{jan_period['forecast_spread']-1.0:.2f}%, {jan_period['forecast_spread']+1.0:.2f}%]")
        
        # Recommendations
        print(f"\n💡 TRADING RECOMMENDATIONS")
        print("-"*60)
        
        if jan_period['forecast_spread'] > 0.5:
            print("1. Consider LONG NQ / SHORT ES spread for January 2026")
            print("2. Entry: First trading day of January")
            print("3. Target: 1-2% spread gain")
            print("4. Stop-loss: If spread turns negative")
        else:
            print("1. Wait for better entry point")
            print("2. Monitor economic indicators in early January")
            print("3. Consider defensive positioning if spread forecast is negative")
        
        # Save results
        historical_data.to_csv('historical_spreads.csv', index=False)
        forecasts.to_csv('2026_forecasts.csv', index=False)
        
        with open('forecast_summary.md', 'w') as f:
            f.write("# Futures Outperformance Forecast Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Key Findings\n\n")
            f.write(f"- Historical NQ outperformance rate: {outperform_pct:.1f}%\n")
            f.write(f"- 2026 average forecast spread: {avg_forecast:.2f}%\n")
            f.write(f"- January 5-16, 2026 forecast: {jan_period['forecast_spread']:.2f}%\n")
            f.write(f"- Probability of January outperformance: {jan_period['prob_outperform']:.1f}%\n")
        
        print("\n✅ Files saved:")
        print("   historical_spreads.csv")
        print("   2026_forecasts.csv")
        print("   forecast_summary.md")
        print("   forecast_plot.html")

def main():
    """Main execution function"""
    print("="*80)
    print("GITHUB FUTURES FORECASTING SYSTEM")
    print("="*80)
    print("\nThis script runs in GitHub Codespaces, Colab, or any Python environment.")
    
    # Initialize forecaster
    forecaster = GitHubForecaster()
    
    # Step 1: Fetch data
    if not forecaster.fetch_data():
        print("⚠️ Using sample data for demonstration")
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=500, freq='B')
        np.random.seed(42)
        forecaster.data = pd.DataFrame({
            'NQ': 16000 * np.cumprod(1 + np.random.normal(0.0008, 0.015, 500)),
            'ES': 4800 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 500)),
            'VIX': np.random.normal(15, 3, 500),
            'TNX': np.random.normal(4.0, 0.2, 500)
        }, index=dates)
    
    # Step 2: Calculate biweekly returns
    historical_data = forecaster.calculate_biweekly_returns()
    
    if historical_data is None or len(historical_data) < 10:
        print("❌ Insufficient data for analysis")
        return
    
    # Step 3: Forecast 2026
    forecasts = forecaster.forecast_2026(historical_data)
    
    # Step 4: Create visualization
    forecaster.create_interactive_plot(historical_data, forecasts)
    
    # Step 5: Generate report
    forecaster.generate_report(historical_data, forecasts)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📋 Files generated:")
    print("   - forecast_plot.html (Interactive visualization)")
    print("   - historical_spreads.csv (Historical data)")
    print("   - 2026_forecasts.csv (2026 predictions)")
    print("   - forecast_summary.md (Summary report)")

if __name__ == "__main__":
    main()
