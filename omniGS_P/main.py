# omniGS_P/main.py

from omniGS_P.start_pipeline import start_pipeline
from omniGS_P.cli import parse_cli

def main():
    """
    Entry point for running omniGS_P-P via `python -m omniGS_P.main --config config.ini`.
    """
    args = parse_cli()
    start_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()