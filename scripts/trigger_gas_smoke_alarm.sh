#!/usr/bin/env bash
# Simulate safety_gas_then_smoke_v1: gas detected -> smoke detected (within 10s)

HOST="${BANBU_HOST:-http://localhost:9000}"
URL="$HOST/api/v2/events/batch"

now() { date -u +"%Y-%m-%dT%H:%M:%S.%3NZ"; }

echo "[1/2] Gas detected (gas: false -> true)"
curl -sf -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"changed_at\": \"$(now)\",
    \"reported_at\": \"$(now)\",
    \"payload\": [{
      \"device_id\": 9,
      \"sequence\": 1,
      \"values\": {\"gas\": true},
      \"previous_values\": {\"gas\": false}
    }]
  }" | python3 -m json.tool
echo

sleep 3

echo "[2/2] Smoke detected (smoke: false -> true)"
curl -sf -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"changed_at\": \"$(now)\",
    \"reported_at\": \"$(now)\",
    \"payload\": [{
      \"device_id\": 11,
      \"sequence\": 2,
      \"values\": {\"smoke\": true},
      \"previous_values\": {\"smoke\": false}
    }]
  }" | python3 -m json.tool
echo

echo "Done. Check server logs for scene trigger."

# BANBU_HOST=http://192.168.0.188:9000 ./scripts/trigger_gas_smoke_alarm.sh
