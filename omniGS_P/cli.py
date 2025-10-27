# omniGS_P/cli.py
import argparse

def parse_cli():
    """
    Parse command-line arguments for omniGS_P pipeline.
    """
    parser = argparse.ArgumentParser(
        description="omniGS_P: Genomic Selection Pipeline"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to configuration file (e.g., config.ini)"
    )
    return parser.parse_args()