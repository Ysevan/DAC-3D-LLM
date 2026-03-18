"""Main entry point for the DAC-3D IIM assistant."""

from config import AppConfig


def main() -> None:
    """Start the assistant application."""
    config = AppConfig()
    print(f"Starting assistant with model: {config.model_name}")
    print("Application skeleton is in place. Functionality is not implemented yet.")


if __name__ == "__main__":
    main()
