import numpy as np
import simpleaudio as sa

def generate_sound(depth, side):
    # Define note frequencies based on depth
    depth = (round((depth * 0.0393701), 1))
    if depth < 24:
        freq = 480  # Higher frequency for nearby objects
    elif depth < 60:
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
    if side == 'Right':
        stereo_signal[:, 1] = audio[:]  # Right channel
    elif side == 'Left':
        stereo_signal[:, 0] = audio[:]  # Left channel
    else:
        stereo_signal[:, 1] = audio[:]  # Right channel
        stereo_signal[:, 0] = audio[:]  # Left channel
        
    # Start playback
    play_obj = sa.play_buffer(stereo_signal, 2, 2, sample_rate)

    # Wait for playback to finish before continuing
    play_obj.wait_done()

def detect_side(center_point, width):
    x, _ = center_point
    third_width = width / 3
    if x < third_width:
        return 'Left'
    elif x < 2 * third_width:
        return 'Middle'
    else:
        return 'Right'
