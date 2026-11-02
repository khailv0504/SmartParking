"""RFID abstraction. Nothing touches SPI until RC522Reader is constructed."""
from __future__ import annotations

import time
from typing import Protocol


class Reader(Protocol):
    def read_uid(self) -> str | None: ...


class MockReader:
    def __init__(self, uids: list[str]):
        self.uids = iter(uids)

    def read_uid(self) -> str | None:
        return next(self.uids, None)


class RC522Reader:
    """Requires SPI enabled and mfrc522 package on Raspberry Pi only.

    The simple MFRC522.ReadCardSerial() API returns UID array or None.
    This adapter never manipulates GPIO pin levels directly.
    """
    def __init__(self):
        try:
            from mfrc522 import MFRC522
        except ImportError as exc:
            raise RuntimeError('RC522 needs a compatible mfrc522 Python library on Pi') from exc
        self.reader = MFRC522()

    def read_uid(self) -> str | None:
        status, _ = self.reader.MFRC522_Request(self.reader.PICC_REQIDL)
        if status != self.reader.MI_OK:
            return None
        status, uid = self.reader.MFRC522_Anticoll()
        if status != self.reader.MI_OK or not uid:
            return None
        # Common MFRC522 Anticoll API returns 4 UID bytes + 1 XOR BCC.
        # The checksum is NOT part of the user-facing card identifier.
        if len(uid) == 5 and (uid[0] ^ uid[1] ^ uid[2] ^ uid[3]) == uid[4]:
            uid = uid[:4]
        return ''.join(f'{v:02X}' for v in uid)
