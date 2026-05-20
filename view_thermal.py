import urllib.request
import json
import sys
import os
import time

# Configuration
URL = "http://10.0.0.242/json"
WIDTH = 24
HEIGHT = 16

# 36-color continuous heatmap palette (Cold -> Hot)
# Blues -> Cyan -> Green -> Yellow -> Orange -> Red -> Magenta -> White
PALETTE = [
    17, 18, 19, 20, 21, 26, 27, 33, 39, 45, 51,   # Blues to Cyan
    50, 49, 48, 47, 46, 82, 118, 154, 190, 226,   # Cyan to Green to Yellow
    220, 214, 208, 202, 196, 160, 124, 88,        # Yellow to Orange to Red
    89, 90, 126, 162, 198, 201, 231               # Dark Red to Magenta to White
]

def get_color(temp, min_t, max_t):
    """Returns an ANSI escape code mapped to a temperature gradient."""
    span = max_t - min_t
    
    # Set a minimum temperature span so random noise doesn't look like extreme hot/cold
    if span < 5.0:
        center = (max_t + min_t) / 2
        min_t = center - 2.5
        max_t = center + 2.5
        span = 5.0

    # Clamp the temperature to the min/max bounds just in case
    temp = max(min_t, min(max_t, temp))
    
    # Normalize temperature to a 0.0 - 1.0 range
    normalized = (temp - min_t) / span
    
    # Map to palette index
    idx = int(normalized * (len(PALETTE) - 1))
    color_code = PALETTE[idx]
    
    return f"\033[38;5;{color_code}m██\033[0m"

def main():
    print("Starting high-res thermal viewer. Press Ctrl+C to stop.")
    os.system("clear" if os.name == "posix" else "cls")
    
    try:
        while True:
            try:
                response = urllib.request.urlopen(URL, timeout=5)
                data = json.loads(response.read())
                
                frame = data.get("frame", [])
                sensors = data.get("sensors", {})
                ambient = sensors.get("temp", 25.0)
                
                if not frame or len(frame) != WIDTH * HEIGHT:
                    print("Error: Received invalid frame data.")
                    time.sleep(1)
                    continue

                min_t = min(frame)
                max_t = max(frame)

                # Move cursor to top-left (0,0)
                sys.stdout.write("\033[H")
                
                print(f"--- High-Res Thermal View ---")
                print(f"Ambient: {ambient:.1f}°C | Frame Min: {min_t:.1f}°C | Frame Max: {max_t:.1f}°C    ")
                print("-" * (WIDTH * 2))

                # Draw Frame
                for y in range(HEIGHT):
                    line = ""
                    for x in range(WIDTH):
                        temp = frame[y * WIDTH + x]
                        line += get_color(temp, min_t, max_t)
                    print(line)
                
                print("-" * (WIDTH * 2))
                
                # Draw Color Bar Legend
                legend_bar = ""
                for c in PALETTE:
                    legend_bar += f"\033[38;5;{c}m██\033[0m"
                # Scale it to fit the screen width
                print("Color Scale (Cold -> Hot):")
                print(f"{min_t:.1f}°C {legend_bar[:WIDTH*2 - 16]} {max_t:.1f}°C")
                
                sys.stdout.flush()
                
            except urllib.error.URLError as e:
                sys.stdout.write(f"\rConnection error: {e}. Retrying...       \n")
            except Exception as e:
                sys.stdout.write(f"\rError: {e}                                 \n")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        sys.stdout.write("\033[" + str(HEIGHT + 8) + ";0H")
        print("\nExiting...")

if __name__ == "__main__":
    main()
