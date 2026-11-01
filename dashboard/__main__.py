import argparse

from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="dashboard", description="Performance and research dashboards")
    parser.add_argument("--port", type=int, default=8766)
    serve(parser.parse_args().port)


if __name__ == "__main__":
    main()
