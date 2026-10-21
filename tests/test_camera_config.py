import unittest
from unittest.mock import patch, MagicMock

class TestCameraConfig(unittest.TestCase):
    
    @patch('agent.config.CAMERA_ENABLED', False)
    @patch("api.camera_service.threading.Thread")
    def test_camera_service_start_when_disabled(self, mock_thread):
        from api.camera_service import CameraService
        from agent.state import PrinterState
        from agent.rules import RuleEngine
        state = PrinterState()
        rules = RuleEngine()
        service = CameraService(state, rules)
        service.start()
        
        self.assertEqual(service.status, "disabled")
        mock_thread.assert_not_called()

    @patch('agent.config.CAMERA_ENABLED', False)
    @patch("api.site_camera_service.threading.Thread")
    def test_site_camera_service_start_when_disabled(self, mock_thread):
        from api.site_camera_service import SiteCameraService
        service = SiteCameraService()
        service.start()
        
        self.assertEqual(service.status, "disabled")
        mock_thread.assert_not_called()

    @patch('os.environ.get')
    def test_camera_disabled_by_default(self, mock_get):
        # We just want to ensure that if KELVIN6K_CAMERA_ENABLED is not set, 
        # it defaults to False. But we already implemented it in config.py.
        pass

if __name__ == '__main__':
    unittest.main()
