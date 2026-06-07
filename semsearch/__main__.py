import sys

from . import cli, crawler, indexer


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: semsearch <query>")
        print("   or: semsearch index")
        print("   or: semsearch crawl")
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == "index":
        indexer.main()
    elif subcommand == "crawl":
        crawler.main()
    else:
        cli.main()


if __name__ == "__main__":
    main()
