import sys
from omniGS.start_pipeline import start_pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.ini>")
        sys.exit(1)

    config_path = sys.argv[1]
    start_pipeline(config_path=config_path)

if __name__ == "__main__":
    main()