"""One h_n LSTM across ten equally represented lengths, pooled w2 calibration."""
from pathlib import Path
from scripts.run_lstm_variable_length import main

if __name__=='__main__':
    main(default_config=Path('configs/lstm_multilength_grid50_equalN.json'))
