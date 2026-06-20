import argparse

from rich.console import Console

from .model_download import download_embedding_model
from .index.embedding_model import LOCAL_MODEL_DIR, is_model_installed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download the local ONNX embedding model for semantic search"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the model even if it is already installed",
    )
    args = parser.parse_args(argv)

    console = Console()
    if is_model_installed() and not args.force:
        console.print(
            f"[green]Embedding model already installed[/green] at [bold]{LOCAL_MODEL_DIR}[/bold]"
        )
        return

    console.print(
        f"[dim]Downloading {LOCAL_MODEL_DIR.name} from Hugging Face"
        f"{'' if args.force else ''}...[/dim]"
    )
    if not args.force:
        console.print(
            "[dim]Optional: export HF_TOKEN=... for faster, more reliable downloads[/dim]"
        )

    model_dir = download_embedding_model(force=args.force)
    console.print(f"[green]Embedding model ready[/green] at [bold]{model_dir}[/bold]")


if __name__ == "__main__":
    main()
