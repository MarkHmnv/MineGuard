import logging
from datetime import datetime

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S.%f'
LOG_FILENAME = datetime.now().strftime(DATE_FORMAT) + '_mineguard.log'

log = logging

log.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[
    logging.StreamHandler(),
    logging.FileHandler(LOG_FILENAME)
])
