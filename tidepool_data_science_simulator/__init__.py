import os
import logging

# Reference
# https://stackoverflow.com/questions/13479295/python-using-basicconfig-method-to-log-to-console-and-file

# Silence matplotlib - so many messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

THIS_DIR = os.path.dirname(__file__)

logging.basicConfig(
     filename=os.path.join(THIS_DIR, 'simulator.log'),
     level=logging.DEBUG,
     format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)

# add the handler to the root logger
logging.getLogger('').addHandler(console)


# For Development. When True, use local dev code
USE_LOCAL_PYLOOPKIT = False

if USE_LOCAL_PYLOOPKIT:
    logger = logging.getLogger(__name__)
    logger.debug("========== Importing Local Pyloopkit ============")
    import sys
    this_dir = os.path.dirname(__file__)
    local_pyloopkit_path = os.path.join("../../PyLoopKit/")  # Assume simulator and
    assert os.path.isdir(local_pyloopkit_path)
    sys.path.insert(0, local_pyloopkit_path)