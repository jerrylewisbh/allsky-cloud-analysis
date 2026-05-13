import socket
import requests
import time

# --- CONFIGURATION ---
ESP32_IP = "sky-thermal-cam.local"
LISTEN_PORT = 10001
# ---------------------

def get_esp_data():
    try:
        response = requests.get(f"http://{ESP32_IP}/json", timeout=1.5)
        if response.status_code == 200:
            return response.json().get("sensors", {})
    except Exception:
        pass
    return None

def handle_client(client, addr):
    print(f"Connection from {addr}")
    client.settimeout(2.0)
    try:
        while True:
            data = client.recv(1024).decode('utf-8').strip()
            if not data:
                break
            
            print(f"  Command: '{data}'")
            
            if "rx" in data:
                sensors = get_esp_data()
                
                # Fetch data, but provide hardcoded defaults if missing or null
                if sensors:
                    mpsas = sensors.get("sky_brightness_mpsas")
                    temp = sensors.get("temp")
                else:
                    mpsas = None
                    temp = None
                
                # Hardcoded defaults as requested
                if mpsas is None: 
                    mpsas = 18.0
                if temp is None: 
                    temp = 20.0
                
                # Format: r, 21.20m, 0000000034Hz, 0000000000c, 0000000.000s, 018.5C
                # Using %05.2f ensures it looks like "18.00" instead of "18.0" 
                # and pad the temperature to exactly match "018.5C" or "020.0C"
                resp = f"r, {mpsas:05.2f}m, 0000000000Hz, 0000000000c, 0000000.000s, {temp:05.1f}C\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Response: {resp.strip()}")
                
            elif "ix" in data:
                resp = "i,00000002,00000003,00000001,00000022\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Response: {resp.strip()}")
                
            elif "cx" in data:
                resp = "c,00000015.31,00000000.00,00000000.00\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Response: {resp.strip()}")
            
            else:
                pass
                
    except Exception as e:
        print(f"  Connection closed/error: {e}")
    finally:
        client.close()

def run_emulator():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(5)
    print(f"SQM-LE Emulator listening on port {LISTEN_PORT}...")
    print(f"Targeting ESP32 at: http://{ESP32_IP}/json")
    print(f"Defaulting to 18.00 mpsas and 20.0C if sensor is null")

    while True:
        client, addr = server.accept()
        handle_client(client, addr)

if __name__ == "__main__":
    run_emulator()
