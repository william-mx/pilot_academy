# scripts/run_train_pilotnet.py
from pathlib import Path

from pilot_academy.train_pilotnet import train_pilotnet_from_run_config

from dotenv import load_dotenv
load_dotenv()

def main():
    RUN_CFG_PATH = Path("/workspaces/pilot_academy/config/runs/pilotnet_baseline.yaml")
    train_pilotnet_from_run_config(str(RUN_CFG_PATH))


if __name__ == "__main__":
    main()
