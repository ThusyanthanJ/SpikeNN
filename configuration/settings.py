from datetime import datetime

CHECKPOINT_PATH = 'checkpoint'

# total training epoches
EPOCH = 25
# time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
# tensorboard log dir
LOG_DIR = 'runs'
# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 5
EPOCH_INDEX = 0
IS_TRAIN = True

A_JESTER_DATA_PATH = ""  # Need to configure
S_JESTER_DATA_PATH = "/dataset/Jester/events_np/"
