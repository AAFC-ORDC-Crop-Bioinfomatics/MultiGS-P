# omniGS/main.py

from omniGS.start_pipeline import start_pipeline
from omniGS.cli import parse_cli

def main():
    """
    Entry point for running OmniGS-P via `python -m omniGS.main --config config.ini`.
    """
    args = parse_cli()
    start_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()