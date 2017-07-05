#!/bin/bash

# the three command line arguments are the input batch file, the input stream file, and the output flagged file
./src/process_log ./log_input/batch_log.json ./log_input/stream_log.json ./log_output/flagged_purchases.json 2> ./log_output/error_log
