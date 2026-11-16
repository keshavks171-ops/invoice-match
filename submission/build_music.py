"""build_music.py - the background music of the demo video, synthesized from scratch.

No samples and no third-party audio: every sound is a sine wave shaped with
numpy, so there is nothing to license. Writes a WAV file.

Run from the project folder:  python submission/build_music.py out.wav 118
"""

from __future__ import annotations

import sys
import wave

import numpy as np

RATE = 44100
BEAT = 60 / 92                      # seconds per beat at 92 beats per minute
CHORDS = [                          # four soft chords, eight beats each (frequencies in Hz)
    (130.81, [261.63, 329.63, 392.00, 493.88]),   # C major 7
    (110.00, [220.00, 261.63, 329.63, 392.00]),   # A minor 7
    (87.31, [174.61, 220.00, 261.63, 329.63]),    # F major 7
    (98.00, [196.00, 246.94, 293.66, 392.00]),    # G major (add 4th voice an octave up)
]


def tone(freq: float, seconds: float, attack: float, decay: float) -> np.ndarray:
    """A sine wave that fades in over `attack` seconds and dies away at rate `decay`."""
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = np.minimum(t / attack, 1.0) * np.exp(-decay * t)
    return np.sin(2 * np.pi * freq * t) * envelope


def add(track: np.ndarray, sound: np.ndarray, start: float, gain: float) -> None:
    """Mix `sound` into `track` starting at `start` seconds."""
    first = int(start * RATE)
    last = min(len(track), first + len(sound))
    if first < len(track):
        track[first:last] += gain * sound[: last - first]


def main() -> None:
    out, seconds = sys.argv[1], float(sys.argv[2])
    left, right = np.zeros(int(seconds * RATE)), np.zeros(int(seconds * RATE))
    chord_length = 8 * BEAT
    for index in range(int(seconds / chord_length) + 1):
        bass, notes = CHORDS[index % len(CHORDS)]
        start = index * chord_length
        for track, detune in ((left, 0.997), (right, 1.003)):      # a slightly different pitch per ear
            add(track, tone(bass, chord_length + 1.5, 0.4, 0.25), start, 0.30)
            for note in notes:                                      # the held chord
                add(track, tone(note * detune, chord_length + 2.0, 1.6, 0.18), start, 0.10)
        for step in range(16):                                      # a quiet plucked pattern on top
            note = notes[(0, 2, 1, 3, 2, 1, 3, 2)[step % 8]] * 2
            track = left if step % 2 == 0 else right
            add(track, tone(note, 1.2, 0.01, 4.5), start + step * BEAT / 2, 0.09)
            add(right if track is left else left, tone(note, 1.2, 0.01, 4.5), start + step * BEAT / 2, 0.04)

    stereo = np.stack([left, right], axis=1)
    t = np.arange(len(stereo)) / RATE
    fade = np.minimum(t / 2.0, 1.0) * np.minimum((seconds - t) / 4.0, 1.0)   # fade in 2 s, out 4 s
    stereo = stereo * fade[:, None]
    stereo = stereo / np.abs(stereo).max() * 0.35                            # leave plenty of headroom
    with wave.open(out, "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes((stereo * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main()
