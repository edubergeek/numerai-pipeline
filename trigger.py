from numerapi import NumerAPI
from time import sleep
from datetime import datetime
import logging

logging.basicConfig(filename='trigger.log', level=logging.DEBUG)

napi = NumerAPI(public_id="your_public_id", secret_key="your_secret_key")

# wait until current round is open and get current_round number
wait_time = 300 # Poll interval in seconds
current_round = 0
while current_round == 0:
    log_message = "Waiting for opening of current round at %s" % (datetime.now())
    logging.info(log_message)
    try:
        if napi.check_round_open():
            current_round = napi.get_current_round() 
            log_message = "Current round %d is open at %s" % (current_round, datetime.now())
            logging.info(log_message)
        else:
            sleep(wait_time)
        
    except:
        sleep(wait_time)

print(current_round)
