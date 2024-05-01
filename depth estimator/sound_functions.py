import numpy as np
import simpleaudio as sa

# Define the base frequency for the low note
base_freq = 240

# Calculate frequencies for each note
low_note_freq = base_freq
medium_note_freq = base_freq * 2 ** (4 / 12)  # A fourth above the low note
high_note_freq = base_freq * 2 ** (7 / 12)  # A fifth above the low note

# Constants for audio
sample_rate = 44100
T = 1.0
t = np.linspace(0, T, int(T * sample_rate), False)
volume = 0.5

def generate_sine_wave(freq):
    note = np.sin(freq * t * 2 * np.pi)
    note *= volume * 32767 / np.max(np.abs(note))
    return note.astype(np.int16)

def play_sound(note, speaker):
    stereo_signal = np.zeros([int(sample_rate * T), 2], dtype=np.int16)
    if speaker == 'left':
        stereo_signal[:, 0] = note
    elif speaker == 'right':
        stereo_signal[:, 1] = note
    elif speaker == 'both':
        stereo_signal[:, 0] = note
        stereo_signal[:, 1] = note
    play_obj = sa.play_buffer(stereo_signal, 2, 2, sample_rate)
    play_obj.wait_done()

def get_sound(note_type, speaker):
    if note_type == 'low':
        note = low_note
    elif note_type == 'medium':
        note = medium_note
    elif note_type == 'high':
        note = high_note
    play_sound(note, speaker)

low_note = generate_sine_wave(low_note_freq)
medium_note = generate_sine_wave(medium_note_freq)
high_note = generate_sine_wave(high_note_freq)

get_sound('low', 'both')