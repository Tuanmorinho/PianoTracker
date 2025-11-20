import cv2
import numpy as np
import time

class VideoProcessor:
    def __init__(self, video_path, roi_points, num_keys=88, start_note=21):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.roi_points = np.array(roi_points, dtype="float32")
        self.num_keys = num_keys
        self.start_note = start_note
        
        # Calculate number of white keys in the range
        self.white_keys = 0
        self.key_map = [] # List of (note_midi, is_black, relative_index)
        
        # Standard piano pattern: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        # Indices in octave: 0, 1,  2, 3,  4, 5, 6,  7, 8,  9, 10, 11
        # White: 0, 2, 4, 5, 7, 9, 11 (C, D, E, F, G, A, B)
        # Black: 1, 3, 6, 8, 10 (C#, D#, F#, G#, A#)
        
        white_key_indices = {0, 2, 4, 5, 7, 9, 11}
        
        current_white_idx = 0
        for i in range(num_keys):
            note = start_note + i
            note_in_octave = note % 12
            is_white = note_in_octave in white_key_indices
            
            if is_white:
                self.key_map.append({'note': note, 'is_black': False, 'idx': current_white_idx})
                current_white_idx += 1
            else:
                # Black key is associated with the white key index it follows/precedes
                # Visually, C# is between C (idx) and D (idx+1).
                # We map it to the boundary.
                self.key_map.append({'note': note, 'is_black': True, 'idx': current_white_idx})
        
        self.num_white_keys = current_white_idx
        
        # Target dimensions
        # We base width on white keys. Say 15px per white key.
        self.white_key_width = 15
        self.target_w = self.num_white_keys * self.white_key_width
        self.target_h = 100
        
        self.target_pts = np.array([
            [0, 0],
            [self.target_w - 1, 0],
            [self.target_w - 1, self.target_h - 1],
            [0, self.target_h - 1]
        ], dtype="float32")
        
        self.M = cv2.getPerspectiveTransform(self.roi_points, self.target_pts)
        
    def get_key_regions(self):
        regions = []
        black_key_width = int(self.white_key_width * 0.6)
        black_key_height = int(self.target_h * 0.6)
        
        for k in self.key_map:
            if not k['is_black']:
                x = k['idx'] * self.white_key_width
                y = 0
                w = self.white_key_width
                h = self.target_h
                regions.append((x, y, w, h, False, k['note']))
            else:
                # Black key centered on the line between k['idx']-1 and k['idx']?
                # k['idx'] is the index of the NEXT white key.
                # So C (idx 0). C# (idx 1). D (idx 1).
                # C# should be at boundary of 0 and 1. i.e. 1 * width.
                center_x = k['idx'] * self.white_key_width
                x = center_x - (black_key_width // 2)
                y = 0
                w = black_key_width
                h = black_key_height
                regions.append((x, y, w, h, True, k['note']))
        return regions

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def warp_keyboard(self, frame):
        """Apply perspective transform to extract keyboard region"""
        warped = cv2.warpPerspective(frame, self.M, (self.target_w, self.target_h))
        return warped

    def get_key_states(self, warped_frame, detector):
        """Detect which keys are pressed based on color detection"""
        mask = detector.detect(warped_frame)
        regions = self.get_key_regions()
        
        # Separate black and white regions for processing
        black_regions = [r for r in regions if r[4]] # r[4] is is_black
        
        key_states = []
        
        # Configurable padding to ignore edges of keys (avoids overlap/bleeding)
        padding_x = 2 
        
        for x, y, w, h, is_black, note in regions:
            # Create a mask for the current key
            key_mask = np.zeros_like(mask)
            
            # Apply padding to the drawn rectangle
            # Ensure width is at least 1 pixel after padding
            pad_w = max(1, w - 2 * padding_x)
            pad_x = x + padding_x
            
            cv2.rectangle(key_mask, (pad_x, y), (pad_x + pad_w, y + h), 255, -1)
            
            if not is_black:
                # If it's a white key, subtract overlapping black keys
                # We also expand the black key subtraction slightly to be safe
                sub_padding = 1
                for bx, by, bw, bh, b_is_black, b_note in black_regions:
                    if bx < x + w and bx + bw > x:
                        # Subtract black key region with slight expansion
                        cv2.rectangle(key_mask, (bx - sub_padding, by), (bx + bw + sub_padding, by + bh), 0, -1)
            
            # Combine with the detection mask
            roi_active = cv2.bitwise_and(mask, mask, mask=key_mask)
            
            # Calculate percentage
            key_area_pixels = cv2.countNonZero(key_mask)
            active_pixels = cv2.countNonZero(roi_active)
            
            if key_area_pixels > 0:
                percentage = active_pixels / key_area_pixels
                # Slightly higher threshold to be stricter
                is_pressed = percentage > 0.15 
            else:
                is_pressed = False
            
            key_states.append(is_pressed)
        
        return key_states

    def release(self):
        self.cap.release()
