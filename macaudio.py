import os
import sys
import torch
import threading
import queue
import platform
import AppKit
import Foundation

# Torch and MPS (Metal Performance Shaders) configuration
torch.backends.mps.is_available()
torch.backends.mps.is_built()

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPTNeoXForCausalLM
)

class MacSpeechRecognizer:
    def __init__(self):
        """
        Native macOS speech recognition using AppKit
        """
        self.recognizer = AppKit.NSSpeechRecognizer.alloc().init()
        self.recognizer.setListensInForegroundOnly_(False)
        self.recognizer.setBlocksOtherRecognizers_(True)
        
        # Language set to Japanese
        self.recognizer.setLanguage_("ja-JP")
        
        # Callback setup
        self.recognized_text = None
        self.recognition_complete = threading.Event()
        
        def callback(text):
            self.recognized_text = text
            self.recognition_complete.set()
        
        self.callback = callback

    def listen(self, timeout=5):
        """
        Listen for speech input
        
        Args:
            timeout (int): Maximum listening time
        
        Returns:
            str: Recognized text or None
        """
        # Reset state
        self.recognized_text = None
        self.recognition_complete.clear()
        
        # Start listening
        self.recognizer.startListening()
        
        # Wait for recognition
        self.recognition_complete.wait(timeout=timeout)
        
        # Stop listening
        self.recognizer.stopListening()
        
        return self.recognized_text

class JapaneseChatbot:
    def __init__(self, model_id="rinna/japanese-gpt-neox-3.6b", hf_token=None):
        """
        Initialize chatbot with M1 Pro optimizations and native macOS audio support.
        
        Args:
            model_id (str): Model identifier
            hf_token (str): Hugging Face authentication token
        """
        if not hf_token:
            raise ValueError("Hugging Face 토큰이 필요합니다.")

        # Detect device
        self.device = self._select_optimal_device()
        print(f"🖥️ 사용 중인 장치: {self.device}")

        # Audio setup
        try:
            # Native macOS Speech Recognition
            self.speech_recognizer = MacSpeechRecognizer()
            
            # Text-to-Speech setup for macOS
            self.tts_engine = self._setup_tts()
            
            # Queue for managing audio input/output
            self.audio_queue = queue.Queue()
            
        except Exception as e:
            print(f"오디오 초기화 오류: {e}")
            sys.exit(1)

        print("🖥️ M1 Pro 최적화 모드로 모델 로딩...")

        try:
            # Tokenizer and model loading with MPS/Metal optimization
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True,
                use_fast=False
            )

            # Load model to MPS/Metal device if available
            self.model = GPTNeoXForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == 'mps' else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to optimal device
            self.model = self.model.to(self.device)

            print(f"✅ {model_id} 모델 로딩 완료!")
        
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            sys.exit(1)

    def _select_optimal_device(self):
        """
        Select the optimal device for M1 Pro.
        
        Returns:
            str: Device type ('mps', 'cuda', or 'cpu')
        """
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _setup_tts(self):
        """
        Set up Text-to-Speech for macOS using NSSpeechSynthesizer.
        
        Returns:
            NSSpeechSynthesizer: Configured speech synthesizer
        """
        synthesizer = AppKit.NSSpeechSynthesizer.alloc().init()
        
        # Set Japanese voice if available
        voices = AppKit.NSSpeechSynthesizer.availableVoices()
        japanese_voices = [
            voice for voice in voices 
            if 'ja_JP' in voice or 'Japanese' in voice
        ]
        
        if japanese_voices:
            synthesizer.setVoice_(japanese_voices[0])
        
        return synthesizer

    def listen_for_audio(self):
        """
        Continuous audio listening thread
        """
        while True:
            try:
                print("🎤 듣고 있습니다... (말씀해 주세요)")
                text = self.speech_recognizer.listen()
                
                if text:
                    # Add recognized text to queue for processing
                    self.audio_queue.put(text)
            
            except Exception as e:
                print(f"음성 수신 오류: {e}")
                break

    def process_audio_queue(self):
        """
        Process audio queue and generate responses
        """
        while True:
            try:
                # Wait for text in queue
                text = self.audio_queue.get()
                
                # Check for exit commands
                if text.lower() in ['exit', '종료', '나가기', 'quit']:
                    print("🤖 대화를 종료합니다.")
                    break
                
                print(f"🔊 인식된 텍스트: {text}")
                
                # Generate response
                response = self.generate_response(text)
                print("🤖:", response)
                
                # Speak response using native macOS speech
                self.tts_engine.startSpeakingString_(response)
            
            except Exception as e:
                print(f"오디오 처리 오류: {e}")

    def start_audio_chat(self):
        """
        Start continuous audio chat
        """
        print("🤖 M1 Pro 일본어 챗봇: 안녕하세요!")
        print("💡 대화를 시작하려면 말씀해 주세요.")
        print("💡 '종료' 또는 'exit'로 대화 종료 가능")

        # Create threads for listening and processing
        listener_thread = threading.Thread(target=self.listen_for_audio, daemon=True)
        processor_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        
        # Start threads
        listener_thread.start()
        processor_thread.start()
        
        # Keep main thread running
        try:
            listener_thread.join()
            processor_thread.join()
        except KeyboardInterrupt:
            print("\n챗봇을 종료합니다.")

    def generate_response(self, prompt, max_tokens=256):
        """
        Generate response with M1 Pro device optimization
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        full_prompt = f"### 指示\n{prompt}\n### 回答\n"
        
        try:
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            generation_kwargs = {
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask,
                'max_new_tokens': max_tokens,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.2,
                'pad_token_id': self.tokenizer.pad_token_id
            }
            
            # Generate on the optimal device
            outputs = self.model.generate(**generation_kwargs)
            
            # Move outputs back to CPU for decoding
            response = self.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            response = response.split('\n### 回答')[-1].strip()
            
            return response
        
        except Exception as e:
            print(f"응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

def main():
    # Check if running on macOS
    if platform.system() != 'Darwin':
        print("❌ 이 스크립트는 macOS에서만 최적화되었습니다.")
        sys.exit(1)

    # Get Hugging Face token
    HF_TOKEN = "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"
    
    try:
        # Use Rinna's Japanese GPT-NeoX model
        chatbot = JapaneseChatbot(
            model_id="rinna/japanese-gpt-neox-3.6b", 
            hf_token=HF_TOKEN
        )
        chatbot.start_audio_chat()
    
    except Exception as e:
        print(f"죄송합니다. 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()