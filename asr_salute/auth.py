import ssl
import uuid
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Optional

_LOGGER = logging.getLogger(__name__)

class SaluteSpeechAuth:
    def __init__(self, auth_key: str, ca_cert_path: str) -> None:
        self._auth_key: str = auth_key
        self._ca_cert_path: str = ca_cert_path
        self._scope: str = "SALUTE_SPEECH_PERS"
        self._token_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        self._access_token: Optional[str] = None
        self._expires_at: Optional[datetime] = None
        self._ssl_context: Optional[ssl.SSLContext] = None

    def _generate_rquid(self) -> str:
        return str(uuid.uuid4())

    def get_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for REST API calls."""
        if self._ssl_context:
            return self._ssl_context
        
        ctx = ssl.create_default_context(cafile=self._ca_cert_path)
        self._ssl_context = ctx
        return self._ssl_context

    async def get_access_token(self) -> str:
        """Get valid access token (refreshes if needed)."""
        if self._access_token and self._expires_at and datetime.now() < self._expires_at:
            return self._access_token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": self._generate_rquid(),
            "Authorization": f"Basic {self._auth_key}",
        }

        ssl_context = self.get_ssl_context()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self._token_url,
                    headers=headers,
                    ssl=ssl_context,
                    data={"scope": self._scope},
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Auth failed: {response.status} {error_text}")

                    data = await response.json()
                    self._access_token = data.get("access_token")
                    # Token usually lives 30 min, refresh a bit earlier
                    self._expires_at = datetime.now() + timedelta(seconds=1700)
                    _LOGGER.debug("Sber token refreshed successfully")
                    return self._access_token
            except Exception as e:
                _LOGGER.error("Error getting Sber token: %s", e)
                raise