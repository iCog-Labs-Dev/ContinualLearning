import utils
import jax

print(utils.he_init(jax.random.PRNGKey(0), 5, 3))
