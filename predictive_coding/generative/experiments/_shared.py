import jax
import jax.numpy as jnp

def generate_synthetic_data(num_samples=100, num_classes=10, feature_dim=784, seed=42):
    """Generate perfectly separable synthetic data for overfitting test."""
    key = jax.random.PRNGKey(seed)
    
    # Generate 10 orthogonal-ish prototypes
    key_proto, key_noise = jax.random.split(key)
    prototypes = jax.random.normal(key_proto, (num_classes, feature_dim))
    prototypes = jnp.tanh(prototypes)  # bounds [-1, 1]
    
    # Assign classes
    labels = jnp.tile(jnp.arange(num_classes), num_samples // num_classes)
    
    # Add noise
    noise = 0.1 * jax.random.normal(key_noise, (num_samples, feature_dim))
    data = prototypes[labels] + noise
    data = jnp.tanh(data)
    
    # One-hot labels
    labels_onehot = jax.nn.one_hot(labels, num_classes)
    
    return data, labels, labels_onehot
