import asyncio
import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.api.calls import ACTIVE_DETECTORS, DEMO_PROVIDER
from backend.detection.detector import LiveCallDetector

router = APIRouter(tags=["Real-Time Audio Stream"])

@router.websocket("/ws/audio/{call_id}")
async def websocket_audio_stream(websocket: WebSocket, call_id: str):
    """
    Real-Time Bidirectional Telephony Media Stream.
    Receives incoming raw audio chunks, passes to sliding window detector,
    and streams continuous classification state updates.
    """
    await websocket.accept()

    # Get or create detector
    if call_id not in ACTIVE_DETECTORS:
        ACTIVE_DETECTORS[call_id] = LiveCallDetector(call_id=call_id)
    detector = ACTIVE_DETECTORS[call_id]

    # Task to handle demo streaming if active
    stream_task = None

    async def run_demo_stream_loop(scenario_id: str):
        """Streams demo audio chunks through the detector and sends websocket updates."""
        try:
            async for chunk in DEMO_PROVIDER.stream_audio_chunks(call_id, chunk_duration_sec=0.5):
                if not detector.is_active:
                    break
                updates = await detector.ingest_audio_chunk(chunk)
                for update in updates:
                    await websocket.send_json(update)
        except Exception as e:
            print(f"[!] Demo stream exception: {e}")

    try:
        # Check if demo stream is pre-registered
        if call_id in DEMO_PROVIDER.active_streams:
            scenario = DEMO_PROVIDER.active_streams[call_id].get("scenario", "deepfake_cloned_call")
            stream_task = asyncio.create_task(run_demo_stream_loop(scenario))

        while True:
            # Receive either binary audio buffer or text/JSON control payload
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Convert 16-bit PCM to float32
                pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
                float_data = pcm_data.astype(np.float32) / 32768.0

                updates = await detector.ingest_audio_chunk(float_data)
                for update in updates:
                    await websocket.send_json(update)

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")

                    if action == "start_demo":
                        scenario = payload.get("scenario", "deepfake_cloned_call")
                        await DEMO_PROVIDER.start_call_stream(call_id, options={"scenario": scenario})
                        if stream_task and not stream_task.done():
                            stream_task.cancel()
                        stream_task = asyncio.create_task(run_demo_stream_loop(scenario))

                    elif action == "stop":
                        detector.is_active = False
                        await DEMO_PROVIDER.stop_call_stream(call_id)
                        if stream_task and not stream_task.done():
                            stream_task.cancel()
                        await websocket.send_json({"status": "STOPPED", "call_id": call_id})

                    elif action == "reset":
                        detector.reset()
                        await websocket.send_json({"status": "RESET", "call_id": call_id})

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        print(f"[*] WebSocket disconnected for call {call_id}")
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
    finally:
        if stream_task and not stream_task.done():
            stream_task.cancel()
        if call_id in ACTIVE_DETECTORS:
            ACTIVE_DETECTORS[call_id].is_active = False
