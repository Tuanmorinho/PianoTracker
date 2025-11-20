import mido
from mido import Message, MidiFile, MidiTrack

class MidiWriter:
    def __init__(self, bpm=120):
        self.mid = MidiFile()
        self.track = MidiTrack()
        self.mid.tracks.append(self.track)
        self.bpm = bpm
        self.ticks_per_beat = self.mid.ticks_per_beat # Default 480
        self.last_event_time = 0 # In seconds

    def _seconds_to_ticks(self, seconds):
        # BPM = beats per minute
        # 1 beat = 60 / BPM seconds
        # 1 tick = (60 / BPM) / ticks_per_beat seconds
        # ticks = seconds / seconds_per_tick
        # ticks = seconds * (bpm * ticks_per_beat / 60)
        return int(seconds * (self.bpm * self.ticks_per_beat / 60))

    def add_note_on(self, note, velocity=64, time=0):
        # time is absolute time in seconds
        delta_seconds = time - self.last_event_time
        delta_ticks = self._seconds_to_ticks(delta_seconds)
        self.last_event_time = time
        
        self.track.append(Message('note_on', note=note, velocity=velocity, time=delta_ticks))

    def add_note_off(self, note, velocity=64, time=0):
        # time is absolute time in seconds
        delta_seconds = time - self.last_event_time
        delta_ticks = self._seconds_to_ticks(delta_seconds)
        self.last_event_time = time
        
        self.track.append(Message('note_off', note=note, velocity=velocity, time=delta_ticks))

    def save(self, filename):
        self.mid.save(filename)
