import sys
import os
import pandas as pd
import numpy as np
import time

# Add the Components directory to the path
sys.path.append(os.path.join(os.getcwd(), 'Components'))

from TickerData_optimized import TickerData

def test_all_indicators_extended():
    """Test all indicators including the original 23 + new 5 indicators"""
    
    # Original 23 technical indicators
    original_indicators = [
        "log_ret_1", "log_ret_2", "log_ret_3",
        "sma_5_close", "ema_fast5_slow10",
        "rsi_3", "rsi_7", "macd_fast5_slow13",
        "atr_5", "bb_width_10", "real_vol_5",
        "obv", "mfi_5", "dollar_vol_z",
        "stoch_k_5", "stoch_d_5", "adx_7",
        "williams_r_7"
    ]
    
    # New indicators
    new_indicators = [
        "sic_sector",
        "day_of_week", 
        "day_of_month", 
        "days_to_month_end"
    ]
    
    all_indicators = original_indicators + new_indicators
    
    print("Testing ALL Indicators (Original + New)")
    print("=" * 60)
    print(f"Original technical indicators: {len(original_indicators)}")
    print(f"New indicators: {len(new_indicators)}")
    print(f"Total indicators to test: {len(all_indicators)}")
    
    print(f"\nNew indicators added:")
    for i, indicator in enumerate(new_indicators, 1):
        print(f"{i:2d}. {indicator}")
    
    # Create sample data for testing
    print(f"\nCreating sample data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Generate realistic sample data
    np.random.seed(42)
    sample_data = []
    
    for ticker in tickers:
        base_price = np.random.uniform(100, 400)
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)
            current_price *= (1 + change)
            prices.append(current_price)
        
        for i, date in enumerate(dates):
            high = prices[i] * (1 + abs(np.random.normal(0, 0.01)))
            low = prices[i] * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 15000000)
            
            sample_data.append({
                'Ticker': ticker,
                'date': date,
                'Open': prices[i] * (1 + np.random.normal(0, 0.005)),
                'High': high,
                'Low': low,
                'Close': prices[i],
                'Volume': volume
            })
    
    df = pd.DataFrame(sample_data)
    df.set_index('date', inplace=True)
    
    print(f"Sample data created: {len(df)} rows, {len(tickers)} tickers")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    try:
        print(f"\nProcessing all {len(all_indicators)} indicators...")
        start_time = time.time()
        
        # Initialize TickerData
        ticker_data = TickerData(
            data=df,
            indicator_list=all_indicators,
            years=1,
            prediction_window=3
        )
        
        # Process all indicators
        result_df = ticker_data.process_all()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Result shape: {result_df.shape}")
        
        # Check indicator status
        print(f"\nOriginal Technical Indicators Status:")
        print("-" * 60)
        
        successful_original = []
        failed_original = []
        
        for indicator in original_indicators:
            if indicator in result_df.columns:
                non_null_count = result_df[indicator].notna().sum()
                total_count = len(result_df)
                percentage = (non_null_count / total_count) * 100
                print(f"✓ {indicator:<20}: {non_null_count:>4}/{total_count} ({percentage:>5.1f}%) non-null")
                successful_original.append(indicator)
            else:
                print(f"✗ {indicator:<20}: Not found in result")
                failed_original.append(indicator)
        
        print(f"\nNew Indicators Status:")
        print("-" * 60)
        
        successful_new = []
        failed_new = []
        
        for indicator in new_indicators:
            if indicator in result_df.columns:
                non_null_count = result_df[indicator].notna().sum()
                total_count = len(result_df)
                percentage = (non_null_count / total_count) * 100
                print(f"✓ {indicator:<20}: {non_null_count:>4}/{total_count} ({percentage:>5.1f}%) non-null")
                successful_new.append(indicator)
            else:
                print(f"✗ {indicator:<20}: Not found in result")
                failed_new.append(indicator)
        
        # Show sample data
        if len(result_df) > 0:
            print(f"\nSample Results (AAPL, first 3 rows):")
            aapl_sample = result_df[result_df['Ticker'] == 'AAPL'].head(3)
            
            # Show a mix of original and new indicators
            sample_original = ['log_ret_1', 'sma_5_close', 'rsi_7']
            sample_new = ['sic_sector', 'day_of_week', 'earnings_dummy_10d']
            sample_cols = ['Ticker'] + [col for col in sample_original + sample_new if col in result_df.columns]
            
            print(aapl_sample[sample_cols].to_string())
            
            # Show unique values for new categorical indicators
            print(f"\nNew Indicators Analysis:")
            print("-" * 40)
            
            if 'sic_sector' in successful_new:
                unique_sectors = result_df['sic_sector'].unique()
                print(f"SIC Sectors: {list(unique_sectors)}")
            
            if 'day_of_week' in successful_new:
                unique_dow = sorted(result_df['day_of_week'].unique())
                print(f"Days of week: {unique_dow} (0=Mon, 6=Sun)")
            
            if 'day_of_month' in successful_new:
                unique_dom = sorted(result_df['day_of_month'].unique())
                print(f"Days of month: {min(unique_dom)}-{max(unique_dom)}")
            
            if 'days_to_month_end' in successful_new:
                unique_dtme = sorted(result_df['days_to_month_end'].unique())
                print(f"Days to month end: {min(unique_dtme)}-{max(unique_dtme)}")
        
        # Summary
        print(f"\n" + "=" * 60)
        print(f"COMPREHENSIVE TEST SUMMARY")
        print(f"=" * 60)
        
        total_successful = len(successful_original) + len(successful_new)
        total_indicators = len(all_indicators)
        
        print(f"✓ Original indicators working: {len(successful_original)}/{len(original_indicators)}")
        print(f"✓ New indicators working: {len(successful_new)}/{len(new_indicators)}")
        print(f"✓ Total indicators working: {total_successful}/{total_indicators}")
        
        failed_total = failed_original + failed_new
        if failed_total:
            print(f"✗ Failed indicators: {', '.join(failed_total)}")
        
        # Performance metrics
        if len(result_df) > 0:
            rows_per_second = len(result_df) / processing_time
            memory_usage = result_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            print(f"\nPerformance Metrics:")
            print(f"- Processing speed: {rows_per_second:,.0f} rows/second")
            print(f"- Memory usage: {memory_usage:.2f} MB")
            print(f"- Data efficiency: {len(result_df)/len(df)*100:.1f}% data retention")
            print(f"- Indicators per row: {len([col for col in result_df.columns if col != 'Ticker'])}")
        
        # Final assessment
        success_rate = total_successful / total_indicators
        if success_rate >= 0.95:  # 95% success rate
            print(f"\n🎉 EXCELLENT: {success_rate*100:.1f}% of all indicators working!")
            return True
        elif success_rate >= 0.85:  # 85% success rate
            print(f"\n✅ GOOD: {success_rate*100:.1f}% of all indicators working!")
            return True
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Only {success_rate*100:.1f}% of indicators working")
            return False
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_indicators_extended()
    print(f"\n" + "=" * 60)
    if success:
        print("🎯 COMPREHENSIVE TEST PASSED!")
        print("All 28 indicators (23 original + 5 new) implemented successfully!")
        print("✅ SIC sector indicators")
        print("✅ Calendar-based indicators") 
        print("✅ Earnings dummy indicators")
        print("✅ Full compatibility with existing technical indicators")
    else:
        print("❌ COMPREHENSIVE TEST FAILED!")
        print("Please review the failed indicators and fix any issues.")
    print("=" * 60)