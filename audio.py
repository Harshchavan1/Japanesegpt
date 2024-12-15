import os
import sys
import torch
import speech_recognition as sr
import pyttsx3
import threading
import queue
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPTNeoXForCausalLM
)

class JapaneseChatbot:
    def __init__(self, model_id="rinna/japanese-gpt-neox-3.6b", hf_token=None):
        """
        Initialize chatbot with continuous audio support for headphones.
        
        Args:
            model_id (str): Model identifier
            hf_token (str): Hugging Face authentication token
        """
        if not hf_token:
            raise ValueError("Hugging Face 토큰이 필요합니다.")

        # Audio setup
        try:
            # Speech Recognition setup
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Text-to-Speech setup
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS for Japanese
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a Japanese voice
            japanese_voices = [
                voice for voice in voices 
                if 'japanese' in voice.name.lower() or 'japan' in voice.name.lower()
            ]
            
            if japanese_voices:
                self.tts_engine.setProperty('voice', japanese_voices[0].id)
            
            # Queue for managing audio input/output
            self.audio_queue = queue.Queue()
            self.stop_listening = None
            
        except Exception as e:
            print(f"오디오 초기화 오류: {e}")
            sys.exit(1)

        print("🖥️ CPU 최적화 모드로 모델 로딩...")

        try:
            # Tokenizer and model loading
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True,
                use_fast=False
            )

            self.model = GPTNeoXForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

            print(f"✅ {model_id} 모델 로딩 완료!")
        
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            sys.exit(1)

    def listen_for_audio(self):
        """
        Continuous audio listening thread
        """
        while True:
            try:
                with self.microphone as source:
                    print("🎤 듣고 있습니다... (말씀해 주세요)")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source)
                    
                    # Add audio to queue for processing
                    self.audio_queue.put(audio)
            
            except Exception as e:
                print(f"음성 수신 오류: {e}")
                break

    def process_audio_queue(self):
        """
        Process audio queue and generate responses
        """
        while True:
            try:
                # Wait for audio in queue
                audio = self.audio_queue.get()
                
                try:
                    # Recognize speech (Japanese)
                    text = self.recognizer.recognize_google(audio, language='ja-JP')
                    print(f"🔊 인식된 텍스트: {text}")
                    
                    # Check for exit commands
                    if text.lower() in ['exit', '종료', '나가기', 'quit']:
                        print("🤖 대화를 종료합니다.")
                        break
                    
                    # Generate response
                    response = self.generate_response(text)
                    print("🤖:", response)
                    
                    # Speak response
                    self.tts_engine.say(response)
                    self.tts_engine.runAndWait()
                
                except sr.UnknownValueError:
                    print("🤷 음성을 인식할 수 없습니다.")
                except sr.RequestError as e:
                    print(f"음성 인식 서비스 오류: {e}")
            
            except Exception as e:
                print(f"오디오 처리 오류: {e}")

    def start_audio_chat(self):
        """
        Start continuous audio chat
        """
        print("🤖 헤드폰 모드 일본어 챗봇: 안녕하세요!")
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
        Generate response with tokenization handling
        (Same implementation as previous version)
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
            )
            
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
            
            outputs = self.model.generate(**generation_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('\n### 回答')[-1].strip()
            
            return response
        
        except Exception as e:
            print(f"응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

def main():
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