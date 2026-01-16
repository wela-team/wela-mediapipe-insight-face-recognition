"""WebSocket broadcasting module for face recognition events."""

import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Optional
import websockets

from configs.config import PROJECT_ROOT, FACES_DIR


class WebSocketBroadcaster:
    """Broadcasts face recognition events to WebSocket server with cooldown protection."""

    def __init__(
        self,
        uri: str = 'ws://localhost:8765',
        cooldown: int = 5,
        faces_dir: Optional[Path] = None
    ):
        """
        Initialize WebSocket broadcaster for face recognition events.
        
        Args:
            uri: WebSocket server URI (default: 'ws://localhost:8765')
            cooldown: Cooldown period in seconds to prevent duplicate broadcasts (default: 5)
            faces_dir: Directory containing face images (default: from config)
        """
        self.uri = uri
        self.cooldown = cooldown
        self.cache = {}  # Maps name -> last_sent_timestamp
        self.lock = threading.Lock()
        self.websocket = None
        self.connected = False
        self.loop = None
        self.loop_thread = None
        
        # Build mapping from recognized names to picture filenames
        if faces_dir is None:
            faces_dir = FACES_DIR
        self.faces_dir = Path(faces_dir)
        self.name_to_filename = {}
        self._build_filename_mapping()
        
        # Start WebSocket connection in background thread
        self._start_connection()
    
    def _build_filename_mapping(self) -> None:
        """Build mapping from recognized names to picture filenames."""
        if not self.faces_dir.exists():
            print(f"[WEBSOCKET] Faces directory {self.faces_dir} not found, skipping filename mapping")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for file in self.faces_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                name = file.stem  # filename without extension
                self.name_to_filename[name] = file.name
        
        print(f"[WEBSOCKET] Mapped {len(self.name_to_filename)} face images to names")
    
    def _run_async_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_websocket())
    
    async def _connect_websocket(self) -> None:
        """Connect to WebSocket server and maintain connection."""
        while True:
            try:
                websocket = await websockets.connect(self.uri)
                self.websocket = websocket
                self.connected = True
                print(f"[WEBSOCKET] Connected to {self.uri}")
                
                # Keep connection alive and handle messages
                try:
                    async for message in websocket:
                        # Handle incoming messages if needed
                        print(f"[WEBSOCKET] Received: {message}")
                except websockets.exceptions.ConnectionClosed:
                    print("[WEBSOCKET] Connection closed by server")
                    self.connected = False
                    self.websocket = None
                finally:
                    try:
                        await websocket.close()
                    except:
                        pass
            except Exception as e:
                print(f"[WEBSOCKET ERROR] Connection failed: {e}")
                self.connected = False
                self.websocket = None
                # Wait before retrying
                await asyncio.sleep(5)
    
    def _start_connection(self) -> None:
        """Start WebSocket connection in background thread."""
        self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.loop_thread.start()
        # Wait a bit for connection to establish
        time.sleep(1)
    
    async def _send_message_async(self, message: str) -> bool:
        """
        Send message to WebSocket server asynchronously.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.connected or self.websocket is None:
            return False
        
        try:
            # Check if websocket is still open
            if self.websocket.closed:
                self.connected = False
                self.websocket = None
                return False
            
            await self.websocket.send(message)
            return True
        except websockets.exceptions.ConnectionClosed:
            print("[WEBSOCKET] Connection closed during send")
            self.connected = False
            self.websocket = None
            return False
        except Exception as e:
            print(f"[WEBSOCKET ERROR] Failed to send message: {e}")
            self.connected = False
            return False
    
    def broadcast_face(self, name: str) -> bool:
        """
        Broadcast recognized face to WebSocket server if cooldown period has passed.
        
        Args:
            name: Name of the recognized face (must not be "Unknown" or None)
        
        Returns:
            bool: True if broadcast was sent, False otherwise
        """
        if name is None or name == "Unknown":
            return False
        
        current_time = time.time()
        should_send = False
        
        with self.lock:
            last_sent = self.cache.get(name, 0)
            if current_time - last_sent >= self.cooldown:
                should_send = True
                self.cache[name] = current_time
        
        if should_send:
            # Get filename for this recognized face
            filename = self.name_to_filename.get(name, f"{name}.jpg")
            
            # Prepare message as JSON
            message_data = {
                "type": "face_recognized",
                "name": name,
                "filename": filename
            }
            message = json.dumps(message_data)
            
            try:
                # Send message using asyncio
                if self.loop and self.loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_message_async(message),
                        self.loop
                    )
                    success = future.result(timeout=2.0)
                    if success:
                        print(f"[WEBSOCKET] Broadcasted: {filename} (recognized: {name})")
                        return True
                else:
                    print("[WEBSOCKET] Event loop not running")
                    return False
            except Exception as e:
                print(f"[WEBSOCKET ERROR] Failed to send: {e}")
                return False
        
        return False
    
    def close(self) -> None:
        """Close WebSocket connection."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.connected = False
