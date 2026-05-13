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
            # Command could be 'rx', 'ix', or 'cx'
            data = client.recv(1024).decode('utf-8').strip()
            if not data:
                break
            
            print(f"  Command: '{data}'")
            
            if "rx" in data:
                sensors = get_esp_data()
                mpsas = sensors.get("sky_brightness_mpsas") if sensors else None
                temp = sensors.get("temp") if sensors else None
                
                # Hardcoded defaults as requested
                if mpsas is None: mpsas = 18.0
                if temp is None: temp = 20.0
                
                # EXACT UNIhedron SQM-LE Formatting:
                # r, 21.20m,0000000034Hz,0000000000c,0000000.000s, 018.5C
                # Notice: No space after 1st, 2nd, 3rd, 4th commas. Space before Temp.
                resp = f"r, {mpsas:5.2f}m,0000000000Hz,0000000000c,0000000.000s, {temp:05.1f}C\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Sent: {resp.strip()}")
                
            elif "ix" in data:
                # i,protocol,model,feature,firmware
                resp = "i,00000002,00000003,00000001,00000022\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Sent: {resp.strip()}")
                
            elif "cx" in data:
                # c,offset,train,test
                resp = "c,00000015.31,00000000.00,00000000.00\r\n"
                client.send(resp.encode('utf-8'))
                print(f"  Sent: {resp.strip()}")
                
    except Exception as e:
        print(f"  Client disconnected: {e}")
    finally:
        client.close()

def run_emulator():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(5)
    print(f"SQM-LE Emulator listening on port {LISTEN_PORT}...")
    
    while True:
        try:
            client, addr = server.accept()
            handle_client(client, addr)
        except Exception as e:
            print(f"Server loop error: {e}")

if __name__ == "__main__":
    run_emulator()
