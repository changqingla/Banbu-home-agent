#!/usr/bin/env bash
# Simulate hand_on_cheek_color_temp_light_v1: one high-confidence vision detection.

HOST="${BANBU_HOST:-http://localhost:9000}"
URL="$HOST/api/v2/events/batch"

now() { date -u +"%Y-%m-%dT%H:%M:%S.%3NZ"; }
epoch() { date +%s; }

echo "[1/1] Vision detects hand-on-cheek pose"
curl -sf -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"changed_at\": \"$(now)\",
    \"reported_at\": \"$(now)\",
    \"source\": \"vision\",
    \"payload\": [{
      \"device_id\": \"entry_camera_vision_1\",
      \"sequence\": 1001,
      \"values\": {
        \"scene_id\": \"hand_on_cheek_color_temp_light_v1\",
        \"detected\": true,
        \"confidence\": 0.86,
        \"reason\": \"person is resting their cheek on their hand\",
        \"frame_id\": \"hand_on_cheek_1001\",
        \"frame_at\": $(epoch)
      },
      \"previous_values\": {
        \"scene_id\": null,
        \"detected\": false,
        \"confidence\": 0,
        \"reason\": \"no matching scene\",
        \"frame_id\": \"hand_on_cheek_1000\",
        \"frame_at\": $(epoch)
      }
    }]
  }" | python3 -m json.tool
echo

echo "Done. Check server logs for hand_on_cheek_color_temp_light_v1 trigger."

# BANBU_HOST=http://192.168.0.188:9000 ./scripts/trigger_hand_on_cheek_vision.sh
