
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
