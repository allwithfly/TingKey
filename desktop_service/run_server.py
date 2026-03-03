from __future__ import annotations

import argparse

import uvicorn

from desktop_service.api_server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Desktop voice input backend service",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="desktop_service_config.json")
    parser.add_argument("--audio-root", default="recordings")
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(config_path=args.config, audio_root=args.audio_root)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

