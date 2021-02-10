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