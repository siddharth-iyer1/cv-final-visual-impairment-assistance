import numpy as np
import simpleaudio as sa

def generate_sound(depth):
    # Define note frequencies based on depth
    if depth < 20:
        freq = 480  # Higher frequency for nearby objects
    elif depth < 40:
        freq = 360  # Medium frequency for medium-distance objects
    else:
        freq = 240  # Lower frequency for far-away objects

    # Generate sound
    sample_rate = 44100
    T = 0.5  # Note duration in seconds
    t = np.linspace(0, T, int(T * sample_rate), False)
    note = np.sin(freq * t * 2 * np.pi)

    # Normalize audio to desired volume (0.5 is half volume)
    volume = 1
    note *= volume
    note *= 32767 / 1 * np.max(np.abs(note))

    # Convert to 16-bit data
    audio = note.astype(np.int16)

    # Create stereo signal
    stereo_signal = np.zeros([int(sample_rate * T), 2], dtype=np.int16)
    stereo_signal[:, 0] = audio[:]  # Left channel
    stereo_signal[:, 1] = audio[:]  # Right channel

    # Start playback
    play_obj = sa.play_buffer(stereo_signal, 2, 2, sample_rate)

    # Wait for playback to finish before continuing
    play_obj.wait_done()