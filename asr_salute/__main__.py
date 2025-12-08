import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path

from wyoming.info import AsrProgram, Attribution, Info, AsrModel
from wyoming.server import AsyncServer

from .handler import SberEventHandler
from .auth import SaluteSpeechAuth

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10305", help="Wyoming server URI")
    parser.add_argument("--token", required=True, help="Sber SaluteSpeech Auth Key")
    parser.add_argument("--cert", default="api/russian_trusted_root_ca.cer", help="Path to Russian Trusted Root CA")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cert_path = Path(__file__).parent / args.cert
    if not cert_path.exists():
        cert_path = Path(args.cert)
    
    if not cert_path.exists():
        _LOGGER.critical("Certificate not found at: %s", cert_path)
        sys.exit(1)
    
    with open(cert_path, 'rb') as f:
        ca_cert_content = f.read()

    auth = SaluteSpeechAuth(auth_key=args.token, ca_cert_path=str(cert_path))

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="sber-cloud",
                description="Sber SaluteSpeech Cloud ASR",
                attribution=Attribution(name="SberDevices", url="https://developers.sber.ru/"),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name="cloud-ru",
                        description="Sber Russian Cloud Model",
                        attribution=Attribution(name="Sber", url=""),
                        installed=True,
                        languages=["ru"],
                        version="1.0",
                    )
                ],
            )
        ],
    )

    _LOGGER.info("Starting Sber Wyoming ASR on %s", args.uri)

    server = AsyncServer.from_uri(args.uri)
    
    handler_factory = partial(
        SberEventHandler,
        wyoming_info=wyoming_info,
        auth=auth,
        ca_cert_content=ca_cert_content
    )

    await server.run(handler_factory)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass