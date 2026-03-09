import sys
import os

# Accept the EULA programmatically through environment variable
os.environ["ISAACSIM_ACCEPT_EULA"] = "Y"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

print("Accepting EULA and Bootstrapping Isaac Sim SimulationApp to fetch extensions...")
try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    print("SimulationApp initialized successfully! All extensions pulled down.\nClosing...")
    simulation_app.close()
    sys.exit(0)
except Exception as e:
    print(f"Error initializing Isaac Sim: {e}")
    sys.exit(1)
