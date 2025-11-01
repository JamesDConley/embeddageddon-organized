"""
Minimal example demonstrating FP4 training with Transformer Engine on standard LLaMA.
This is a reproduction case for testing FP4 block scaling functionality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling, Recipe

def convert_linear_to_te_linear(module):
    """Convert all torch.nn.Linear layers to Transformer Engine Linear layers.
    
    Preserves weight tying between embed_tokens and lm_head for LLaMA models.
    """
    # Check for embedding-lm_head weight tying (LLaMA models)
    weights_are_tied = False
    if hasattr(module, 'model') and hasattr(module.model, 'embed_tokens'):
        if hasattr(module, 'lm_head') and isinstance(module.lm_head, torch.nn.Linear):
            embed_weight_id = id(module.model.embed_tokens.weight)
            lm_head_weight_id = id(module.lm_head.weight)
            weights_are_tied = (embed_weight_id == lm_head_weight_id)
    
    # Convert all Linear layers to TE Linear
    def convert_layer(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                te_linear = te.Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype
                )
                te_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    te_linear.bias.data.copy_(child.bias.data)
                setattr(mod, name, te_linear)
                del child
            else:
                convert_layer(child)
    
    convert_layer(module)
    
    # Re-establish weight tying if it existed
    if weights_are_tied:
        print("Restoring weight tying between embed_tokens and lm_head...")
        old_weight = module.lm_head.weight
        del old_weight
        module.lm_head.weight = module.model.embed_tokens.weight
        print(f"Weight tying restored: {module.lm_head.weight.numel():,} shared parameters")
    
    return module


def main():
    print("="*80)
    print("Minimal FP4 Training Example")
    print("="*80)
    
    # Configuration
    model_name = "NousResearch/Llama-3.2-1B"  # Small LLaMA model for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nDevice: {device}")
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"\nOriginal parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Convert to TE Linear layers
    print("\nConverting to Transformer Engine Linear layers...")
    model = convert_linear_to_te_linear(model)
    print(f"After conversion parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup FP4 recipe
    print("\nSetting up NVFP4 Block Scaling recipe...")
    fp4_recipe = NVFP4BlockScaling(disable_stochastic_rounding=True, disable_rht=True, disable_2d_quantization=True)
    
    # Prepare sample input
    print("\nPreparing sample input...")
    sample_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer([sample_text] * 16, return_tensors="pt", padding=True).to(device)
    
    # Forward pass with FP4
    print("\nRunning forward pass with FP4 autocast...")
    model.train()
    
    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
            print(f"✓ Forward pass successful!")
            print(f"  Loss: {loss.item():.4f}")
            
        # Backward pass
        print("\nRunning backward pass...")
        loss.backward()
        print("✓ Backward pass successful!")
        
    except Exception as e:
        print(f"\n✗ ERROR during FP4 execution:")
        print(f"  {type(e).__name__}: {e}")
        raise
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()