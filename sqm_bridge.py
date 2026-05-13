import socket
import requests
import time

# --- CONFIGURATION ---
ESP32_IP = "sky-thermal-cam.local"  # Defaulting to mDNS name, change to IP if needed
LISTEN_PORT = 10001
# ---------------------

def get_esp_data():
    try:
        response = requests.get(f"http://{ESP32_IP}/json", timeout=2)
        if response.status_code == 200:
            return response.json().get("sensors", {})
    except Exception as e:
        print(f"Error fetching from ESP32: {e}")
    return None

def run_emulator():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(5)
    print(f"SQM-LE Emulator listening on port {LISTEN_PORT}...")
    print(f"Fetching data from: http://{ESP32_IP}/json")

    while True:
        client, addr = server.accept()
        try:
            data = client.recv(1024).decode('utf-8').strip()
            if not data:
                continue
            
            print(f"Received command: {data} from {addr}")
            
            if "rx" in data:
                sensors = get_esp_data()
                mpsas = sensors.get("sky_brightness_mpsas", 0.0) if sensors else 0.0
                temp = sensors.get("temp", 0.0) if sensors else 0.0
                
                if mpsas is None: mpsas = 0.0
                if temp is None: temp = 0.0
                
                # Format: r, 21.20m, 0000000034Hz, 0000000000c, 0000000.000s, 018.5C
                resp = f"r, {mpsas:05.2f}m, 0000000000Hz, 0000000000c, 0000000.000s, {temp:05.1f}C\r\n"
                client.send(resp.encode('utf-8'))
                
            elif "ix" in data:
                # Returns protocol, model, feature, firmware
                client.send(b"i, 00000002, 00000003, 00000001, 00000001\r\n")
                
            elif "cx" in data:
                # Returns calibration offset
                client.send(b"c, 00000015.31, 00000000.00, 00000000.00\r\n")
                
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client.close()

if __name__ == "__main__":
    run_emulator()
