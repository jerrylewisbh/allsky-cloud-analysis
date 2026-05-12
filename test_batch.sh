#!/bin/bash
# Find 10 valid pairs roughly spaced out
FILES=$(find /Volumes/allsky_images/images/20260510 -name "*.jpg" | grep -v thumbnails | awk 'NR % 200 == 0' | head -n 10)

for ALLSKY in $FILES; do
    # Extract the relative path parts to find the matching thermal
    REL_PATH=$(echo $ALLSKY | sed 's|/Volumes/allsky_images/images/||')
    THERMAL="/Volumes/astro_image_thermal/ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212/exposures/$REL_PATH"
    THERMAL_BMP="${THERMAL%.jpg}.bmp"
    
    if [ -f "$THERMAL_BMP" ]; then
        echo "Processing $ALLSKY..."
        python3 matched_crop.py --allsky "$ALLSKY" --thermal "$THERMAL_BMP" --thermal-fov 55 --rotation 29 --offset-x -0.04 --offset-y 0.10 --distortion -0.04 --flip-h
    else
        echo "No thermal match for $ALLSKY"
    fi
done
