import logging
import json
from pathlib import Path
from typing import Dict, Any, Set, List

logger = logging.getLogger(__name__)


class ThreatIntelligenceFeed:
    def __init__(self, feed_config: Dict[str, Any]):
        self.ip_blacklist_path = Path(
            feed_config.get(
                "ip_blacklist_path", "data_vault/threat_intel/ip_blacklist.json"
            )
        )
        self.device_blacklist_path = Path(
            feed_config.get(
                "device_blacklist_path", "data_vault/threat_intel/device_blacklist.json"
            )
        )
        self.promo_watchlist_path = Path(
            feed_config.get(
                "promo_watchlist_path", "data_vault/threat_intel/promo_watchlist.json"
            )
        )

        self.ip_blacklist: Set[str] = set()
        self.device_blacklist: Set[str] = set()
        self.promo_watchlist: Set[str] = set()

        self._load_all_feeds()
        logger.info("ThreatIntelligenceFeed initialized.")

    def _load_feed(self, file_path: Path) -> Set[str]:
        if not file_path.exists():
            logger.warning(
                f"Threat intelligence file not found: {file_path}. Creating empty file."
            )
            with open(file_path, "w") as f:
                json.dump([], f)
            return set()
        try:
            with open(file_path, "r") as f:
                return set(json.load(f))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return set()
        except Exception as e:
            logger.error(f"Error loading feed from {file_path}: {e}")
            return set()

    def _load_all_feeds(self):
        self.ip_blacklist = self._load_feed(self.ip_blacklist_path)
        self.device_blacklist = self._load_feed(self.device_blacklist_path)
        self.promo_watchlist = self._load_feed(self.promo_watchlist_path)
        logger.info(
            f"Loaded {len(self.ip_blacklist)} blacklisted IPs, {len(self.device_blacklist)} blacklisted devices, {len(self.promo_watchlist)} promo codes on watchlist."
        )

    def update_feed(self, feed_type: str, new_items: List[str]):
        if feed_type == "ip_blacklist":
            self.ip_blacklist.update(new_items)
            with open(self.ip_blacklist_path, "w") as f:
                json.dump(list(self.ip_blacklist), f, indent=2)
        elif feed_type == "device_blacklist":
            self.device_blacklist.update(new_items)
            with open(self.device_blacklist_path, "w") as f:
                json.dump(list(self.device_blacklist), f, indent=2)
        elif feed_type == "promo_watchlist":
            self.promo_watchlist.update(new_items)
            with open(self.promo_watchlist_path, "w") as f:
                json.dump(list(self.promo_watchlist), f, indent=2)
        else:
            logger.warning(f"Unknown feed type for update: {feed_type}")
            return
        logger.info(f"Updated {feed_type} with {len(new_items)} new items.")

    def check_event_for_threats(self, event: Dict[str, Any]) -> Dict[str, bool]:
        threat_status = {
            "is_ip_blacklisted": False,
            "is_device_blacklisted": False,
            "is_promo_on_watchlist": False,
        }

        if event.get("ip_address") in self.ip_blacklist:
            threat_status["is_ip_blacklisted"] = True
            logger.warning(
                f"Blacklisted IP detected for event {event.get('event_id')}: {event['ip_address']}"
            )

        if event.get("device_info") in self.device_blacklist:
            threat_status["is_device_blacklisted"] = True
            logger.warning(
                f"Blacklisted device detected for event {event.get('event_id')}: {event['device_info']}"
            )

        if event.get("promo_code_used") in self.promo_watchlist:
            threat_status["is_promo_on_watchlist"] = True
            logger.warning(
                f"Promo code on watchlist detected for event {event.get('event_id')}: {event['promo_code_used']}"
            )

        return threat_status
