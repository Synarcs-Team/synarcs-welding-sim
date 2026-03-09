import asyncio
import websockets
import requests
import time

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

async def test_websocket(endpoint_name):
    print(f"\n--- Testing {endpoint_name} ---")
    start = time.time()
    try:
        async with websockets.connect(f"{WS_URL}/{endpoint_name}") as ws:
            while True:
                try:
                    msg = await ws.recv()
                    print(f"[{time.time()-start:.2f}s] {msg}")
                    if "[EXIT]" in msg:
                        break
                except websockets.ConnectionClosed:
                    print("Connection closed unexpectedly!")
                    break
    except Exception as e:
        print(f"Failed to connect: {e}")
    print(f"--- {endpoint_name} Finished in {time.time()-start:.2f}s ---\n")

async def main():
    print("Testing /api/configure...")
    payload = {
        "joint_type": "tee", 
        "bw": 0.33, 
        "bl": 0.26, 
        "bt": 0.005, 
        "sh": 0.28, 
        "st": 0.005,
        "tilt": 199,
        "flip": 0,
        "sim_engine": "pybullet"
    }
    r = requests.post(f"{API_URL}/api/configure", json=payload)
    print("Response:", r.json())
    
    await test_websocket("ws/scan")
    await test_websocket("ws/process")
    await test_websocket("ws/weld")
    
if __name__ == "__main__":
    asyncio.run(main())
