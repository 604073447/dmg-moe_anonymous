
PARAM_PREFIX = r'^(?:(?:module|_orig_mod|model)\.)*'
MODULE_NAME = [
    'stem',
    'layers.0.encoder',
    'layers.0.gate',
    'layers.0.head',
    'layers.1.encoder',
    'layers.1.gate',
    'layers.1.head',
    'layers.2.encoder',
    'layers.2.gate',
    'layers.2.head',
    'layers.3.encoder',
    'layers.3.gate',
    'layers.3.head',
]

SPACE2 = '  '
SPACE3 = '   '
SPACE4 = '    '
TAB = '\t'

SHOT_LIST = ['All', 'Many', 'Med.', 'Few', 'Macro']
metric_of_interest = ['MAE', 'MSE', 'GM']
lower_better_metrics = ['mae', 'mse', 'rmse', 'loss', 'gm',
                        'MAE', 'MSE', 'RMSE', 'Loss', 'GM',]
met_map = {
    'loss': 'Loss',
    'mae': 'MAE',
    'mse': 'MSE',
    'rmse': 'RMSE',
    'gmean': 'GM',
    'within_3': 'WT3',
    'within_5': 'WT5',
    'within_10': 'WT10',
    'cs': 'CS',
}
met_map_inv = {
    'Loss': 'loss',
    'MAE': 'mae',
    'MSE': 'mse',
    'RMSE': 'rmse',
    'GM': 'gmean',
    'WT3': 'within_3',
    'WT5': 'within_5',
    'WT10': 'within_10',
    'CS': 'cs',
}

__all__ = [
    "SPACE2",
    "SPACE3",
    "SPACE4",
    "TAB",
    "lower_better_metrics",
]
