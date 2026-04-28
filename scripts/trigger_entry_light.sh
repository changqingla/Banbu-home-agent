#!/usr/bin/env bash
# Simulate entry_auto_light_v1: door open → person detected (within 30s)

HOST="${BANBU_HOST:-http://localhost:9000}"
URL="$HOST/api/v2/events/batch"

now() { date -u +"%Y-%m-%dT%H:%M:%S.%3NZ"; }

echo "[1/2] Door opened (contact: true -> false)"
curl -sf -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"changed_at\": \"$(now)\",
    \"reported_at\": \"$(now)\",
    \"payload\": [{
      \"device_id\": 4,
      \"sequence\": 1,
      \"values\": {\"contact\": false},
      \"previous_values\": {\"contact\": true}
    }]
  }" | python3 -m json.tool
echo

sleep 2

echo "[2/2] PIR triggered (occupancy: false -> true)"
curl -sf -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"changed_at\": \"$(now)\",
    \"reported_at\": \"$(now)\",
    \"payload\": [{
      \"device_id\": 1,
      \"sequence\": 2,
      \"values\": {\"occupancy\": true},
      \"previous_values\": {\"occupancy\": false}
    }]
  }" | python3 -m json.tool
echo

echo "Done. Check server logs for scene trigger."

# BANBU_HOST=http://192.168.0.188:9000 ./scripts/trigger_entry_light.sh
