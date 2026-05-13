import socket
import requests
import time

# --- CONFIGURATION ---
ESP32_IP = "sky-thermal-cam.local"
LISTEN_PORT = 10001
# ---------------------

def get_esp_data():
    try:
        # Some ESPHome setups use .local, others use IP. 
        # Adding a short timeout to keep the bridge responsive.
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
                # Default to 0.0 if sensors are offline
                mpsas = sensors.get("sky_brightness_mpsas", 0.0) if sensors else 0.0
                temp = sensors.get("temp", 0.0) if sensors else 0.0
                
                if mpsas is None: mpsas = 0.0
                if temp is None: temp = 0.0
                
                # Format: r, 21.20m, 0000000034Hz, 0000000000c, 0000000.000s, 018.5C
                # Note the exact spacing: r, [space] value [m]
                resp = f"r, {mpsas:5.2f}m, 0000000000Hz, 0000000000c, 0000000.000s, {temp:05.1f}C\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Response: {resp.strip()}")
                
            elif "ix" in data:
                # Format: i,protocol,model,feature,firmware
                # No spaces after first comma is common in newer firmwares
                resp = "i,00000002,00000003,00000001,00000022\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Response: {resp.strip()}")
                
            elif "cx" in data:
                # Format: c,offset,train,test
                resp = "c,00000015.31,00000000.00,00000000.00\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Response: {resp.strip()}")
            
            else:
                # Catch-all for unexpected commands
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

    while True:
        client, addr = server.accept()
        handle_client(client, addr)

if __name__ == "__main__":
    run_emulator()
