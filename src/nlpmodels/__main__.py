"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """NLP Models."""


if __name__ == "__main__":
    main(prog_name="NLP-Models")  # pragma: no cover
