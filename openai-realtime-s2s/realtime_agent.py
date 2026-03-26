import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
import threading
import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    raise RuntimeError("OPENAI_API_KEY bị thiếu.")

API_KEY = os.environ["OPENAI_API_KEY"].strip()
MODEL = "gpt-4o-realtime-preview"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

RATE = 24000
CHUNK = 1024

# VAD: threshold quá cao (0.85) dễ bỏ sót tiếng Việt; silence quá dài làm chờ lâu
VAD_CONFIG = {
    "type": "server_vad",
    "threshold": 0.55,
    "prefix_padding_ms": 350,
    "silence_duration_ms": 550,
}

# Trước đây 1.5s: mic bị thay bằng toàn số 0 → server không nghe được lời bạn ngay sau khi AI nói
ECHO_MUTE_DURATION = 0.35

# Pre-buffer loa: tránh callback đọc nhanh hơn mạng → underrun → tiếng giật/nhảy chữ
SPEAKER_PREROLL_BYTES = int(RATE * 2 * 0.05)

# ✅ Chỉ cho phép barge-in sau khi AI đã phát ra ít nhất N bytes âm thanh thật
BARGE_IN_MIN_BYTES = RATE * 2 * 1  # Tối thiểu 1 giây AI đã phát thì mới cho ngắt


def _must_mask_mic_for_echo(
    *,
    ai_is_speaking: bool,
    mute_mic_until: float,
    loop_time: float,
    speaker_pending_bytes: int,
) -> bool:
    """
    Echo loa→mic khiến server VAD tưởng user nói → barge-in giả, AI tự cắt giữa câu.
    Gửi silence khi AI còn đang phát hoặc buffer loa còn âm (kể cả sau response.audio.done).
    """
    if ai_is_speaking:
        return True
    if loop_time < mute_mic_until:
        return True
    if speaker_pending_bytes > 0:
        return True
    return False


async def run_agent():
    loop = asyncio.get_running_loop()
    print("[DEBUG] run_agent started")

    mic_queue = asyncio.Queue()
    speaker_buffer = bytearray()
    buffer_lock = threading.Lock()

    # ✅ Theo dõi bao nhiêu bytes đã thật sự phát ra loa
    bytes_played = 0
    bytes_played_lock = threading.Lock()

    # Chỉ bắt đầu rút buffer loa sau khi đủ pre-roll (giảm underrun / nhảy chữ)
    speaker_playback_started = False

    def mic_callback(indata, frames, time, status):
        if status:
            print("[DEBUG] mic status:", status)
        data = indata.copy().tobytes()
        loop.call_soon_threadsafe(mic_queue.put_nowait, data)

    def speaker_callback(outdata, frames, time, status):
        if status:
            print("[DEBUG] speaker status:", status)
        nonlocal speaker_buffer, bytes_played, speaker_playback_started
        needed = frames * 2
        with buffer_lock:
            if not speaker_playback_started:
                if len(speaker_buffer) >= SPEAKER_PREROLL_BYTES:
                    speaker_playback_started = True
                    print("[DEBUG] speaker: pre-roll đủ, bắt đầu phát")
            if speaker_playback_started and len(speaker_buffer) >= needed:
                chunk = bytes(speaker_buffer[:needed])
                del speaker_buffer[:needed]
                outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
                with bytes_played_lock:
                    bytes_played += needed
            else:
                outdata.fill(0)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    mic_stream = sd.InputStream(
        samplerate=RATE, channels=1, dtype='int16',
        blocksize=CHUNK, callback=mic_callback
    )
    speaker_stream = sd.OutputStream(
        samplerate=RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK,
        callback=speaker_callback,
        latency="high",
    )

    ai_is_speaking = False
    mute_mic_until = 0.0

    try:
        with mic_stream, speaker_stream:
            print("🎙️  Mic và loa đã sẵn sàng")

            async with websockets.connect(
                WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                print("🟢 Đã kết nối OpenAI Realtime API!")
                print("💬 Hãy bắt đầu nói... (Ctrl+C để thoát)\n")

                await websocket.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "instructions": "Bạn là trợ lý AI nói tiếng Việt. Hãy trả lời ngắn gọn, thân thiện.",
                        "voice": "alloy",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": VAD_CONFIG
                    }
                }))

                async def send_audio():
                    nonlocal mute_mic_until
                    while True:
                        data = await mic_queue.get()
                        now = loop.time()
                        with buffer_lock:
                            pending = len(speaker_buffer)
                        if _must_mask_mic_for_echo(
                            ai_is_speaking=ai_is_speaking,
                            mute_mic_until=mute_mic_until,
                            loop_time=now,
                            speaker_pending_bytes=pending,
                        ):
                            data = b"\x00" * len(data)
                        await websocket.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(data).decode("utf-8")
                        }))

                async def receive_events():
                    nonlocal ai_is_speaking, mute_mic_until, speaker_buffer, bytes_played, speaker_playback_started
                    ai_audio_bytes_count = 0

                    async for message in websocket:
                        event = json.loads(message)
                        event_type = event.get("type", "")

                        if event_type == "response.audio.delta":
                            audio_bytes = base64.b64decode(event["delta"])
                            if len(audio_bytes) == 0:
                                continue
                            if not ai_is_speaking:
                                with buffer_lock:
                                    speaker_playback_started = False
                                print("[DEBUG] response audio: chunk mới → reset pre-roll loa")
                            ai_is_speaking = True
                            ai_audio_bytes_count += len(audio_bytes)
                            with buffer_lock:
                                speaker_buffer.extend(audio_bytes)

                        elif event_type == "input_audio_buffer.speech_started":
                            if ai_is_speaking:
                                # ✅ Chỉ cho barge-in nếu AI đã phát đủ âm thanh ra loa
                                with bytes_played_lock:
                                    played = bytes_played
                                if played >= BARGE_IN_MIN_BYTES:
                                    print("\n🛑 Bạn ngắt lời AI!")
                                    with buffer_lock:
                                        speaker_buffer.clear()
                                        speaker_playback_started = False
                                    with bytes_played_lock:
                                        bytes_played = 0
                                    ai_is_speaking = False
                                    ai_audio_bytes_count = 0
                                else:
                                    # AI chưa kịp phát → bỏ qua barge-in giả
                                    played_s = played / (2 * RATE)
                                    print(f"\n⚠️  Bỏ qua barge-in giả (AI mới phát {played_s:.2f}s)")
                            else:
                                print("\n🎙️  Đang nghe bạn nói...")

                        elif event_type == "response.audio.done":
                            if ai_audio_bytes_count > 0:
                                ai_is_speaking = False
                                mute_mic_until = loop.time() + ECHO_MUTE_DURATION
                                ai_spoke_s = ai_audio_bytes_count / (2 * RATE)
                                print(
                                    f"\n✅ AI nói xong ({ai_spoke_s:.1f}s). "
                                    f"Giảm mic echo ~{ECHO_MUTE_DURATION}s (gửi silence)..."
                                )
                                ai_audio_bytes_count = 0
                                with bytes_played_lock:
                                    bytes_played = 0
                                with buffer_lock:
                                    # Câu rất ngắn (< pre-roll): vẫn phát hết phần còn trong buffer
                                    if len(speaker_buffer) > 0:
                                        speaker_playback_started = True
                                        print("[DEBUG] response.audio.done: flush phần âm còn lại trong buffer loa")

                        elif event_type == "response.done":
                            print("🟡 Đang lắng nghe...")

                        elif event_type == "error":
                            err = event.get("error", {})
                            msg = err.get("message", str(event)) if isinstance(err, dict) else str(event)
                            print(f"\n❌ Lỗi server: {msg}")
                            raise RuntimeError(f"Realtime API error: {msg}")

                    print("[DEBUG] WebSocket receive loop ended (connection closed)")

                send_task = asyncio.create_task(send_audio(), name="send_audio")
                recv_task = asyncio.create_task(receive_events(), name="receive_events")
                tasks = (send_task, recv_task)

                # FIRST_COMPLETED: khi websocket đóng, receive_events không raise →
                # FIRST_EXCEPTION sẽ kẹt vĩnh viễn vì send_audio đang chờ mic_queue.
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        print(f"\n❌ Task '{task.get_name()}' lỗi: {exc}")
                        raise exc

    finally:
        print("\n🔴 Đang dọn dẹp...")
        print("👋 Đã tắt.")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nĐang tắt...")