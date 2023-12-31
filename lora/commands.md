
### Phi2 lora training

Download and convert model to npz
```bash
python ../llms/phi2/convert.py --mlx-path phi2-mlx
```

Get files for the HF tokenizer and put into the same folder
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.save_pretrained("phi2-mlx/")
```

Put the following into `phi2-mlx/params.json`
```json
{
    "max_sequence_length": 2048,
    "model_dim": 2560,
    "num_heads": 32,
    "num_layers": 32,
    "rotary_dim": 32
}
```

Then run lora training
```bash
python lora_phi2.py --model phi2-mlx --adapter-file phi2-mlx/adapters.npz --batch-size 4 --lora-layers 16 --train --iters 600
```

Infer with lora
```bash
python lora_phi2.py --model phi2-mlx \
               --adapter-file phi2-mlx/adapters.npz \
               --num-tokens 50 \
               --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

Infer without lora
```bash
python ../llms/phi2/phi2.py  \
            --model phi2-mlx \
            --max-tokens 50 \
            --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

Oops, something's wrong. The lora result looks like shit, even though the test set perplexity and loss is much better.

```bash
python lora_phi2.py --model phi2-mlx --test
# Test loss 3.193, Test ppl 24.365.

python lora_phi2.py --model phi2-mlx --test --adapter-file phi2-mlx/adapters.npz
# Test loss 1.707, Test ppl 5.513.
```

How to test generation:
```bash
python lora_phi2.py --model phi2-mlx --adapter-file phi2-mlx/adapters.npz --num-tokens 50 --prompt "What is the capital of Spain?"
```
```
But not when using 
```bash
python ../llms/phi2/phi2.py --model phi2-mlx --max-tokens 50 --prompt "Q: What is the capital of Spain?"
```

### Mistral 7B v0.2 instruct lora training

Download and convert model to npz
```bash
curl -O https://files.mistral-7b-v0-2.mistral.ai/Mistral-7B-v0.2-Instruct.tar
mkdir mistral
tar -xf Mistral-7B-v0.2-Instruct.tar -C mistral

python convert.py --torch-model mistral --mlx-model mistral-mlx

# cleanup
rm -rf mistral/ Mistral-7B-v0.2-Instruct.tar
```

Train
```bash
python lora.py --model mistral-mlx \
               --adapter-file mistral-mlx/adapters.npz \
               --resume-adapter-file mistral-mlx/adapters.npz \
               --batch-size 2 \
               --lora-layers 8 \
               --train \
               --iters 600
```


Infer with lora
```bash
python lora.py --model mistral-mlx \
               --adapter-file mistral-mlx/adapters.npz \
               --num-tokens 50 \
               --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

Infer without lora
```bash
python ../llms/mistral/mistral.py  \
            --model mistral-mlx \
            --max-tokens 50 \
            --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

Loss/perplexity after training is good:
```bash
 python lora.py --model mistral-mlx --test --adapter-file mistral-mlx/adapters.npz
# Test loss 1.541, Test ppl 4.668.
```
