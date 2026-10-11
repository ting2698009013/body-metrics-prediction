"""Command-line launcher for the local workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from body_soft_sensor.inference import ModelRegistry
from body_soft_sensor import __version__
from body_soft_sensor.webapp import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动身体成分估算台")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--check-models", action="store_true", help="校验模型并输出模型卡，不启动服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机可访问")
    parser.add_argument("--port", default=8000, type=int, help="监听端口")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="可选的模型目录；省略时使用项目内当前激活模型",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_models:
        try:
            registry = ModelRegistry(args.model_dir)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Model check failed: {error}") from error
        print(json.dumps(registry.model_info(), ensure_ascii=False, indent=2))
        return
    uvicorn.run(create_app(model_dir=args.model_dir), host=args.host, port=args.port)

