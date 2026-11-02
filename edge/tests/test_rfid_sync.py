import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from edge.database import Database
from edge.sync import SyncWorker


class CardSyncTests(unittest.TestCase):
    def test_preexisting_cards_and_new_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'parking.db'
            db = Database(path)
            db.register_card('a1b2c3d4')
            snapshot, _ = db.card_snapshot_due()
            self.assertEqual(snapshot['cards'][0]['uid'], 'A1B2C3D4')
            version = snapshot['snapshot_version']
            worker = SyncWorker(db, 'https://example.com/api/v1/edge/events', 'secret')
            self.assertEqual(worker.card_url, 'https://example.com/api/v1/edge/card-snapshots')
            with patch.object(worker, 'send_card_snapshot') as send:
                self.assertEqual(worker.run_cards_once(), (1, 0))
                self.assertEqual(send.call_count, 1)
            self.assertIsNone(db.card_snapshot_due())
            db.register_card('a1b2c3d4', enabled=False)
            newer, _ = db.card_snapshot_due()
            self.assertGreater(newer['snapshot_version'], version)
            self.assertFalse(newer['cards'][0]['enabled'])

    def test_offline_retry_and_concurrent_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / 'parking.db')
            db.register_card('ABC12345')
            worker = SyncWorker(db, 'https://example.com/api/v1/edge/events', 'secret')
            with patch.object(worker, 'send_card_snapshot', side_effect=OSError('offline')):
                self.assertEqual(worker.run_cards_once(), (0, 1))
            self.assertGreater(db.card_sync_status()['attempts'], 0)
            old_version = db.card_sync_status()['snapshot_version']
            db.register_card('DEF12345')
            db.mark_card_snapshot_synced(old_version)
            self.assertLess(db.card_sync_status()['synced_version'], db.card_sync_status()['snapshot_version'])
            with patch.object(worker, 'send_card_snapshot'):
                self.assertEqual(worker.run_cards_once(), (1, 0))
            self.assertIsNone(db.card_snapshot_due())


if __name__ == '__main__':
    unittest.main()
