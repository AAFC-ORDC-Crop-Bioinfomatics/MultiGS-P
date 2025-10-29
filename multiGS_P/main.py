# multiGS_P/main.py

from multiGS_P.start_pipeline import start_pipeline
from multiGS_P.cli import parse_cli

def main():
    """
    Entry point for running multiGS_P-P via `python -m multiGS_P.main --config config.ini`.
    """
    args = parse_cli()
    start_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()