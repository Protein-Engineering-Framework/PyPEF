# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import os
import numpy as np

import logging
logger = logging.getLogger('pypef.utils.to_file')

def predictions_out(
        predictions,
        model,
        prediction_set,
        path: str = ''
):
    """
    Writes predictions (of the new sequence space) to text file(s).
    """
    name, value = [], []
    for (val, nam) in predictions:
        name.append(nam)
        value.append('{:f}'.format(val))

    data = np.array([name, value]).T
    col_width = max(len(str(value)) for row in data for value in row) + 5

    head = ['Name', 'Prediction']
    txt_file_out_path = os.path.abspath(
        os.path.join(
            path, 'Predictions_' + str(os.path.basename(model)) + 
            '_' + str(os.path.splitext(os.path.basename(prediction_set))[0]) + '.txt'
            )
    )
    logger.info(txt_file_out_path)
    with open(txt_file_out_path, 'w') as f:
        f.write("".join(caption.ljust(col_width) for caption in head) + '\n')
        f.write(len(head)*col_width*'-' + '\n')
        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')
    logger.info(f'Wrote predictions to {txt_file_out_path}.')
