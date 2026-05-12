#!/bin/bash

# Configuration
BACKUP_DIR="/mnt/nas_backups"
DB_CONTAINER="allsky-cloud-analysis-db-1"
DB_USER="allsky"
DB_NAME="cloud_analysis"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="allsky_cloud_backup_$DATE"

# Check if NAS is mounted
if ! mountpoint -q "$BACKUP_DIR"; then
    echo "Error: NAS is not mounted at $BACKUP_DIR"
    exit 1
fi

echo "Starting backup: $BACKUP_NAME"

# 1. Dump the Database
docker exec $DB_CONTAINER pg_dump -U $DB_USER $DB_NAME > "$BACKUP_DIR/$BACKUP_NAME.sql"

# 2. Backup Grafana Provisioning and code
cd /home/jerryfmedeiros/allsky-cloud-analysis
tar -czf "$BACKUP_DIR/$BACKUP_NAME_configs.tar.gz" grafana/ api/ docker-compose.yml

# 3. Cleanup: Keep only last 30 days
find "$BACKUP_DIR" -name "allsky_cloud_backup_*" -mtime +30 -delete

echo "Backup complete!"
