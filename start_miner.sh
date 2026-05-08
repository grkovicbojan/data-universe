python3 ./neurons/miner.py \
  --wallet.name hanibalbtwallet \
  --wallet.hotkey hanibalwallethotkey \
  2>&1 | tee -a "logs/miner_$(date +%F_%H-%M-%S).log"