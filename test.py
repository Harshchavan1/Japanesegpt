import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPTNeoXForCausalLM
)

class JapaneseChatbot:
    def __init__(self, model_id="rinna/japanese-gpt-neox-3.6b", hf_token=None):
        """
        Initialize with Rinna's Japanese GPT-NeoX model.
        
        Args:
            model_id (str): Model identifier
            hf_token (str): Hugging Face authentication token
        """
        if not hf_token:
            raise ValueError("Hugging Face 토큰이 필요합니다.")

        print("🖥️ CPU 최적화 모드로 모델 로딩...")

        try:
            # Specific tokenizer loading for Rinna model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True,
                use_fast=False  # Disable fast tokenizer
            )

            # Model loading with specific configurations
            self.model = GPTNeoXForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float32,  # Ensure CPU compatibility
                low_cpu_mem_usage=True
            )

            print(f"✅ {model_id} 모델 로딩 완료!")
            print(f"💻 사용 중인 CPU: {os.popen('wmic cpu get name').read().strip()}")

        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            sys.exit(1)

    def generate_response(self, prompt, max_tokens=256):
        """
        Generate response with improved tokenization handling.
        
        Args:
            prompt (str): User input
            max_tokens (int): Maximum tokens to generate
        
        Returns:
            str: Generated response
        """
        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        full_prompt = f"### 指示\n{prompt}\n### 回答\n"
        
        try:
            # Tokenize input with padding and attention mask
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # Adjust as needed
            )
            
            # Generate with more robust settings
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
            
            # Generate response
            outputs = self.model.generate(**generation_kwargs)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('\n### 回答')[-1].strip()
            
            return response
        
        except Exception as e:
            print(f"응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

    def chat(self):
        """
        Interactive chat loop optimized for CPU
        """
        print("🤖 CPU 최적화 일본어 챗봇: 안녕하세요!")
        print("💡 Rinna GPT-NeoX 모델을 사용하고 있습니다.")
        
        while True:
            try:
                user_input = input("당신: ").strip()
                
                if user_input.lower() in ['exit', '종료', '나가기']:
                    print("🤖 안녕히 가세요!")
                    break
                
                # Generate and print response
                response = self.generate_response(user_input)
                print("🤖:", response)
            
            except KeyboardInterrupt:
                print("\n챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")

def main():
    # Get Hugging Face token
    HF_TOKEN = "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"
    
    if not HF_TOKEN:
        print("Hugging Face 토큰이 필요합니다.")
        HF_TOKEN = input("Hugging Face 토큰을 입력하세요: ").strip()
    
    try:
        # Use Rinna's Japanese GPT-NeoX model
        chatbot = JapaneseChatbot(
            model_id="rinna/japanese-gpt-neox-3.6b", 
            hf_token=HF_TOKEN
        )
        chatbot.chat()
    
    except Exception as e:
        print(f"죄송합니다. 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()