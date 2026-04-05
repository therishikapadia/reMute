import pychromecast
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromecastController:
    def __init__(self, friendly_name="Hall TV"):
        self.friendly_name = friendly_name
        self.cast = None
        self.browser = None
        self.is_muted = False

    def connect(self):
        """Discover and connect to the specify Chromecast device."""
        logger.info(f"Searching for Chromecast: {self.friendly_name}...")
        chromecasts, self.browser = pychromecast.get_chromecasts()
        
        for cc in chromecasts:
            if cc.name == self.friendly_name:
                self.cast = cc
                logger.info(f"Found {self.friendly_name}. Connecting...")
                self.cast.wait()
                self.is_muted = self.cast.status.volume_muted
                logger.info(f"Connected to {self.friendly_name}. Initial mute state: {self.is_muted}")
                return True
        
        logger.error(f"Could not find Chromecast with name: {self.friendly_name}")
        return False

    def set_mute(self, mute: bool):
        """Set the mute state of the TV."""
        if not self.cast:
            logger.warning("Chromecast not connected. Cannot set mute.")
            return

        if self.is_muted == mute:
            return  # Avoid redundant calls

        try:
            self.cast.set_volume_muted(mute)
            self.is_muted = mute
            logger.info(f"TV {'Muted' if mute else 'Unmuted'}")
        except Exception as e:
            logger.error(f"Error setting mute state: {e}")

    def disconnect(self):
        """Stop discovery and disconnect."""
        if self.browser:
            pychromecast.discovery.stop_discovery(self.browser)
            logger.info("Discovery stopped.")

if __name__ == "__main__":
    # Test script
    controller = ChromecastController(friendly_name="Hall TV")
    if controller.connect():
        controller.set_mute(True)
        time.sleep(2)
        controller.set_mute(False)
        controller.disconnect()
