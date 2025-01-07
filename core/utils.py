import hashlib
import random

def generate_random_hash() -> str:

    # Generate a random 16-byte string
    random_bytes = bytearray(random.getrandbits(8) for _ in range(16))
    # Create a hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the random bytes
    hash_object.update(random_bytes)
    # Get the hexadecimal representation of the hash
    random_hash = hash_object.hexdigest()
    return random_hash