import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GenerativePCN
from src.training import train_step
from src.inference import settle_states
from src.energy import predict_lower

from _shared import generate_synthetic_data

def main():
    print("Running PCN Sanity Check...")
    
    X, Y_idx, Y = generate_synthetic_data()
    print(f"Data shape: {X.shape}, Labels shape: {Y.shape}")
    
    pcn = GenerativePCN(layer_sizes=[784, 256, 256, 10])
    key = jax.random.PRNGKey(0)
    params = pcn.init_params(key)
    
    epochs = 200
    eta_w = 0.1
    eta_x = 0.1
    inference_steps = 50
    
    energy_history = []
    
    for epoch in range(epochs):
        params, metrics = train_step(
            params, X, Y, 
            eta_x=eta_x, eta_w=eta_w, 
            inference_steps=inference_steps,
            init_mode="bottom_up"
        )
        
        #  Monotonic Energy Descent check
        inf_energy = metrics["energy_history"]
        
        energy_history.append(metrics["energy"])
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            pseudo_logits = pcn.forward(params, X)
            acc = jnp.mean(jnp.argmax(pseudo_logits, axis=1) == Y_idx)
            print(f"Epoch {epoch} | Total Energy: {metrics['energy']:.4f} | Acc: {acc*100:.2f}%")
            
    # Learning Descent check
    if energy_history[-1] >= energy_history[0]:
        print("ERROR: Global energy did not decrease across training!")
        sys.exit(1)
        
    print("\n--- Training complete. Checking Inference ---")
    
    #  Classification check
    pseudo_logits = pcn.forward(params, X)
    preds = jnp.argmax(pseudo_logits, axis=1)
    acc = jnp.mean(preds == Y_idx)
    print(f"Classification Accuracy on Training Data: {acc * 100:.2f}%")
    
    if acc < 0.9:
        print("Warning: Model failed to adequately overfit the synthetic data.")
        
    #  Generative Check
    print("\n--- Testing Generative Pathway ---")
    from src.utils import get_activation
    # Create states dict with just the labels
    gen_states = {3: jax.nn.one_hot(jnp.arange(10), 10)}
    for i in range(2, -1, -1):
        act_fn, _ = get_activation(params[i].get("activation", "tanh"))
        gen_states[i] = act_fn(gen_states[i+1] @ params[i]["w"].T + params[i]["b"])
        
    generated_X = gen_states[0]
    
    sim = jnp.dot(generated_X[0], X[Y_idx == 0][0]) / (jnp.linalg.norm(generated_X[0]) * jnp.linalg.norm(X[Y_idx == 0][0]))
    print(f"Cosine similarity of generated class 0 with true class 0 sample: {sim:.4f}")
    
    print("\nSanity Check complete.")

if __name__ == "__main__":
    main()
 