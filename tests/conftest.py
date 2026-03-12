import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to the path so pytest can find the module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Mocking the isaac sim imports the joint factory requires, to run without an engine 
mock_isaac = MagicMock()
sys.modules['isaacsim'] = mock_isaac
sys.modules['isaacsim.core'] = mock_isaac
sys.modules['isaacsim.core.api'] = mock_isaac
sys.modules['isaacsim.core.api.objects'] = mock_isaac
sys.modules['isaacsim.core.utils'] = mock_isaac
sys.modules['isaacsim.core.utils.prims'] = mock_isaac
sys.modules['isaacsim.core.prims'] = mock_isaac
