import onnxruntime as ort
import numpy as np
import pandas as pd

def onnx_predict(model_path, input_df, window_size):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    predictions = []
    tickers = []
    dates = []
    for i in range(window_size, len(input_df)):
        date = input_df.index[i]
        ticker = input_df['Ticker'].iloc[i] if 'Ticker' in input_df.columns else None
        values = input_df.drop(columns=['Ticker']).values.astype(np.float32)

        input_window = values[i - window_size:i]

        # Fix: add batch dimension → shape = (1, window_size, num_features)
        input_window = np.expand_dims(input_window, axis=0)

        output = session.run(None, {input_name: input_window})
        predictions.append(float(output[0][0][0]))
        tickers.append(ticker)
        dates.append(date)

    # Create DataFrame with predictions
    preds_df = pd.DataFrame({
        'Ticker': tickers,
        'Predicted': predictions
    },index=dates)

    return preds_df
