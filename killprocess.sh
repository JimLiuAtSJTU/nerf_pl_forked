

kill -9 $(nvidia-smi -g 0 | grep python | awk ' {print $5}')


