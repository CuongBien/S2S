import asyncio
import websockets
import json
import base64
import pyaudio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project directory (same folder as this script)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
print(f"[DEBUG] Loaded environment from: {_env_path}")

# --- CẤU HÌNH ---
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Set it in .env next to realtime_agent.py, e.g. OPENAI_API_KEY=sk-..."
    )
API_KEY = os.environ["OPENAI_API_KEY"].strip()
print("[DEBUG] OPENAI_API_KEY is set (length:", len(API_KEY), "chars)")
MODEL = "gpt-4o-realtime-preview"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

# Chuẩn âm thanh của OpenAI Realtime: PCM16, 24kHz, Mono
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

async def run_agent():
    # 1. Khởi tạo PyAudio (Mic và Loa)
    audio = pyaudio.PyAudio()
    
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    speaker_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    # Biến để lưu trữ âm thanh AI đang trả về
    ai_audio_queue = asyncio.Queue()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(WS_URL, extra_headers=headers) as websocket:
        print("🟢 Đã kết nối với OpenAI Realtime API!")
        
        # Gửi cấu hình ban đầu (System Prompt)
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "Bạn là trợ lý AI nói tiếng Việt. Hãy trả lời ngắn gọn, thân thiện.",
                "voice": "alloy" # Giọng nam/nữ của OpenAI
            }
        }))

        # Task 1: Liên tục gửi âm thanh từ Micro của bạn lên Server
        async def send_audio():
            while True:
                # Đọc âm thanh từ mic và mã hóa Base64
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                base64_audio = base64.b64encode(data).decode("utf-8")
                
                # Ném thẳng audio vào buffer của AI (giống như ống nước chảy liên tục)
                await websocket.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }))
                await asyncio.sleep(0) # Nhường luồng

        # Task 2: Liên tục nhận sự kiện và âm thanh từ Server trả về
        async def receive_events():
            async for message in websocket:
                event = json.loads(message)

                # Khi AI đang nói, server sẽ gửi từng cục âm thanh nhỏ
                if event["type"] == "response.audio.delta":
                    audio_bytes = base64.b64decode(event["delta"])
                    await ai_audio_queue.put(audio_bytes)

                # 🔥 ĐÂY LÀ CHÌA KHÓA CỦA BARGE-IN (NGẮT LỜI) 🔥
                # Nếu VAD của Server phát hiện bạn cất tiếng nói trong lúc AI đang nói...
                elif event["type"] == "input_audio_buffer.speech_started":
                    print("\n🛑 BẠN ĐÃ NGẮT LỜI AI! (Barge-in detected)")
                    # 1. Xóa ngay lập tức tất cả âm thanh của AI đang nằm trong hàng đợi chờ phát ra loa
                    while not ai_audio_queue.empty():
                        ai_audio_queue.get_nowait()
                    # (Server cũng sẽ tự động hủy việc sinh câu trả lời cũ để nghe bạn nói)

                elif event["type"] == "response.audio.done":
                    print("\n✅ AI đã nói xong.")

        # Task 3: Phát âm thanh của AI ra loa
        async def play_audio():
            while True:
                audio_chunk = await ai_audio_queue.get()
                speaker_stream.write(audio_chunk)

        # Chạy song song cả 3 việc: Nghe (Mic) - Nhận (WebSocket) - Nói (Loa)
        await asyncio.gather(send_audio(), receive_events(), play_audio())

# Chạy chương trình
if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("Đã tắt trợ lý AI.")