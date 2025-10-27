# omniGS_P/utils/logging.py
import logging, os
from datetime import datetime

def setup_logging(results_dir, run_mode, geno_base, log_level="INFO"):
    log_path = os.path.join(
    results_dir,
    f"gs_{run_mode}_{geno_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s(): %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_path
