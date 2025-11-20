import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import re

def note_to_midi(note_str):
    # e.g. A0 -> 21, C4 -> 60
    note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    match = re.match(r"([A-G]#?)(-?\d+)", note_str.upper())
    if not match:
        return None
    note_name, octave = match.groups()
    midi = (int(octave) + 1) * 12 + note_map[note_name]
    return midi

def midi_to_note(midi):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PianoTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        print("Piano Tracker App v2.0 - Loaded")

        self.title("Piano Tracker & Xuất MIDI")
        self.geometry("1200x800")

        # Layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1) # Push everything up

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Piano Tracker", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.load_btn = ctk.CTkButton(self.sidebar_frame, text="Chọn Video", command=self.load_video)
        self.load_btn.grid(row=1, column=0, padx=20, pady=10)

        self.calibrate_btn = ctk.CTkButton(self.sidebar_frame, text="Chọn Vùng Phím", command=self.start_calibration, state="disabled")
        self.calibrate_btn.grid(row=2, column=0, padx=20, pady=10)

        self.pick_color_btn = ctk.CTkButton(self.sidebar_frame, text="Chọn Màu Phím", command=self.start_color_picking, state="disabled")
        self.pick_color_btn.grid(row=3, column=0, padx=20, pady=10)

        self.set_c_btn = ctk.CTkButton(self.sidebar_frame, text="Chọn Nốt Đô (C4)", command=self.start_set_c, state="disabled")
        self.set_c_btn.grid(row=4, column=0, padx=20, pady=10)

        self.preview_btn = ctk.CTkButton(self.sidebar_frame, text="Kiểm Tra Lưới", command=self.preview_grid_overlay, state="disabled", fg_color="green")
        self.preview_btn.grid(row=5, column=0, padx=20, pady=10)

        self.process_btn = ctk.CTkButton(self.sidebar_frame, text="Bắt Đầu Xử Lý", command=self.start_processing, state="disabled")
        self.process_btn.grid(row=6, column=0, padx=20, pady=10)

        self.export_btn = ctk.CTkButton(self.sidebar_frame, text="Xuất MIDI", command=self.export_midi, state="disabled")
        self.export_btn.grid(row=7, column=0, padx=20, pady=10)
        
        # Key Range Config
        self.config_frame = ctk.CTkFrame(self.sidebar_frame)
        self.config_frame.grid(row=8, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.config_frame, text="Cấu hình phím:").pack(pady=5)
        
        self.start_note_entry = ctk.CTkEntry(self.config_frame, placeholder_text="A0")
        self.start_note_entry.pack(pady=5, padx=5)
        self.start_note_entry.insert(0, "A0")
        
        self.end_note_entry = ctk.CTkEntry(self.config_frame, placeholder_text="C8")
        self.end_note_entry.pack(pady=5, padx=5)
        self.end_note_entry.insert(0, "C8")
        
        self.update_range_btn = ctk.CTkButton(self.config_frame, text="Cập nhật Range", command=self.update_key_range)
        self.update_range_btn.pack(pady=5, padx=5)

        # MIDI Config
        self.midi_config_frame = ctk.CTkFrame(self.sidebar_frame)
        self.midi_config_frame.grid(row=9, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.midi_config_frame, text="Cấu hình MIDI:").pack(pady=5)
        
        self.bpm_entry = ctk.CTkEntry(self.midi_config_frame, placeholder_text="BPM (120)")
        self.bpm_entry.pack(pady=5, padx=5)
        self.bpm_entry.insert(0, "120")
        
        self.time_sig_entry = ctk.CTkEntry(self.midi_config_frame, placeholder_text="Nhịp (4/4)")
        self.time_sig_entry.pack(pady=5, padx=5)
        self.time_sig_entry.insert(0, "4/4")

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Trạng thái: Chờ", wraplength=180)
        self.status_label.grid(row=10, column=0, padx=20, pady=20)

        # Main Content Area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        
        # Container for video to handle centering
        self.video_container = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.video_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.video_label = ctk.CTkLabel(self.video_container, text="Chưa có video nào")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.video_canvas = tk.Canvas(self.video_container, bg="black", highlightthickness=0)
        # Canvas is initially hidden or size 0
        
        self.video_path = None
        self.cap = None
        self.is_processing = False
        self.calibration_mode = False
        self.picking_color_mode = False
        self.setting_c_mode = False
        self.calibration_points = []
        self.target_hsv = None
        
        # Range variables
        self.start_note_midi = 21 # A0
        self.end_note_midi = 108 # C8
        self.num_keys = 88
        
        # Zoom variables
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Dragging variables
        self.dragging_point_index = None
        
        # Bind mouse wheel for zoom
        self.video_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.video_canvas.bind("<ButtonPress-2>", self.start_pan) # Middle mouse to pan
        self.video_canvas.bind("<B2-Motion>", self.do_pan)
        
        # Bind mouse events for interaction (Click, Drag, Release)
        self.video_canvas.bind("<Button-1>", self.on_mouse_click)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def finish_calibration(self):
        self.calibration_mode = False
        print("Hoàn tất chọn vùng:", self.calibration_points)
        self.status_label.configure(text="Đã chọn vùng xong.")
        
        # calibration_points are already in original video coordinates
        self.roi_points = list(self.calibration_points)
        
        # Enable buttons safely
        try:
            self.process_btn.configure(state="normal")
            print("Enabled process_btn")
        except Exception as e:
            print(f"Error enabling process_btn: {e}")

        try:
            self.preview_btn.configure(state="normal")
            print("Enabled preview_btn")
        except Exception as e:
            print(f"Error enabling preview_btn: {e}")
            
        self.display_frame(self.current_frame_cv)

    def update_key_range(self):
        s_note = self.start_note_entry.get()
        e_note = self.end_note_entry.get()
        
        s_midi = note_to_midi(s_note)
        e_midi = note_to_midi(e_note)
        
        if s_midi is not None and e_midi is not None and e_midi > s_midi:
            self.start_note_midi = s_midi
            self.end_note_midi = e_midi
            self.num_keys = e_midi - s_midi + 1
            self.start_note_offset = s_midi # Update offset
            print(f"Updated Range: {s_note} ({s_midi}) to {e_note} ({e_midi}). Total keys: {self.num_keys}")
            self.status_label.configure(text=f"Range: {self.num_keys} phím ({s_note}-{e_note})")
            
            if hasattr(self, 'current_frame_cv'):
                self.display_frame(self.current_frame_cv)
        else:
            self.status_label.configure(text="Lỗi: Tên nốt không hợp lệ!")

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        
        if hasattr(self, 'current_frame_cv'):
            self.display_frame(self.current_frame_cv)

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def do_pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        if hasattr(self, 'current_frame_cv'):
            self.display_frame(self.current_frame_cv)

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.calibrate_btn.configure(state="normal")
                self.pick_color_btn.configure(state="normal")
                self.set_c_btn.configure(state="normal")
                self.status_label.configure(text="Đã tải video. Vui lòng chọn vùng phím.")
            self.cap.release()

    def display_frame(self, frame):
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        self.original_size = (w, h)
        self.current_frame_cv = frame # Store original CV frame for color picking
        
        # Draw Overlay if ROI exists
        if hasattr(self, 'roi_points') and len(self.roi_points) == 4:
            # Draw ROI box
            pts = np.array(self.roi_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame_rgb, [pts], True, (255, 0, 0), 2)
            
            # Draw Corners for dragging
            for p in self.roi_points:
                cv2.circle(frame_rgb, (int(p[0]), int(p[1])), 10, (0, 255, 0), -1) # Green circles
            
            # Draw Key Grid
            # We need to recreate the key regions logic here or use the processor
            # Let's replicate the logic briefly for visualization
            
            # 1. Calculate Regions
            start_note = self.start_note_midi
            num_keys = self.num_keys
            
            white_key_indices = {0, 2, 4, 5, 7, 9, 11}
            key_map = []
            current_white_idx = 0
            for i in range(num_keys):
                note = start_note + i
                if (note % 12) in white_key_indices:
                    key_map.append({'note': note, 'is_black': False, 'idx': current_white_idx})
                    current_white_idx += 1
                else:
                    key_map.append({'note': note, 'is_black': True, 'idx': current_white_idx})
            
            num_white_keys = current_white_idx
            white_key_width = 15 # Arbitrary base unit
            target_w = num_white_keys * white_key_width
            target_h = 100
            
            # Perspective Transform
            src_pts = np.array(self.roi_points, dtype="float32")
            dst_pts = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            M_inv = np.linalg.inv(M)
            
            # Prepare Overlay Layer
            overlay = frame_rgb.copy()
            
            # Helper to draw key
            def draw_key_on_overlay(x, y, w, h, color, text=None, text_color=(255, 0, 0)):
                # Define 4 corners in target space
                pts_target = np.array([
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]]
                ], dtype="float32")
                
                # Transform to source space
                pts_src = cv2.perspectiveTransform(pts_target, M_inv)
                pts_int = pts_src.astype(np.int32)
                
                # Draw filled polygon
                cv2.fillPoly(overlay, [pts_int], color)
                # Draw border
                cv2.polylines(overlay, [pts_int], True, (100, 100, 100), 1)
                
                if text:
                    # Draw text at bottom center
                    # Get bottom center in src space
                    # We can approximate by taking mean of bottom two points
                    b1 = pts_src[3][0]
                    b2 = pts_src[2][0]
                    center = (int((b1[0] + b2[0]) / 2), int((b1[1] + b2[1]) / 2))
                    cv2.putText(overlay, text, (center[0] - 10, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            # Draw White Keys First
            for k in key_map:
                if not k['is_black']:
                    x = k['idx'] * white_key_width
                    draw_key_on_overlay(x, 0, white_key_width, target_h, (200, 200, 200, 100)) # Light Gray
                    
                    # Draw C Note Name
                    if k['note'] % 12 == 0:
                         octave = (k['note'] // 12) - 1
                         # Calculate position again? Or just pass to draw function
                         # Let's just redraw the text
                         pass

            # Draw Black Keys Second
            black_key_width = int(white_key_width * 0.6)
            black_key_height = int(target_h * 0.6)
            
            for k in key_map:
                if k['is_black']:
                    center_x = k['idx'] * white_key_width
                    x = center_x - (black_key_width // 2)
                    draw_key_on_overlay(x, 0, black_key_width, black_key_height, (50, 50, 50, 100)) # Dark Gray

            # Draw Note Names (C only) on top
            for k in key_map:
                if not k['is_black'] and k['note'] % 12 == 0:
                    x = k['idx'] * white_key_width
                    octave = (k['note'] // 12) - 1
                    
                    # Calculate center for text
                    pts_target = np.array([[[x + white_key_width/2, target_h - 10]]], dtype="float32")
                    pt_src = cv2.perspectiveTransform(pts_target, M_inv)[0][0]
                    
                    cv2.putText(overlay, f"C{octave}", (int(pt_src[0]) - 10, int(pt_src[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Blend overlay
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame_rgb, 1 - alpha, 0, frame_rgb)

        # Draw partial calibration points
        if self.calibration_mode and self.calibration_points:
            # Points are now in Original Video Coordinates
            for p in self.calibration_points:
                cv2.circle(frame_rgb, (int(p[0]), int(p[1])), 10, (0, 0, 255), -1) # Red circles

        img = Image.fromarray(frame_rgb)
        
        # Resize for display with Zoom
        self.display_w = int(800 * self.zoom_level)
        self.display_h = int(int(h * (800 / w)) * self.zoom_level)
        
        img = img.resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        
        self.current_image = ImageTk.PhotoImage(img) # Use ImageTk for Canvas
        
        # Hide label, show canvas
        self.video_label.place_forget()
        self.video_canvas.place(relx=0.5, rely=0.5, anchor="center")
        self.video_canvas.config(width=800, height=600) # Fixed canvas size, content scrolls/pans? 
        # Actually, let's just let the canvas be fixed size and move the image inside.
        
        # Center the image based on pan
        center_x = 400 + self.pan_x
        center_y = 300 + self.pan_y
        
        self.video_canvas.delete("all")
        self.video_canvas.create_image(center_x, center_y, anchor="center", image=self.current_image)

    def get_image_coords(self, event):
        center_x = 400 + self.pan_x
        center_y = 300 + self.pan_y
        
        img_top_left_x = center_x - self.display_w // 2
        img_top_left_y = center_y - self.display_h // 2
        
        x = event.x - img_top_left_x
        y = event.y - img_top_left_y
        return x, y

    def on_mouse_click(self, event):
        x, y = self.get_image_coords(event)
        
        # Convert click to Original Coordinates
        scale_x = self.original_size[0] / self.display_w
        scale_y = self.original_size[1] / self.display_h
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        
        # Check if clicking near an existing point to drag (Finished ROI)
        if hasattr(self, 'roi_points') and len(self.roi_points) == 4:
            for i, p_orig in enumerate(self.roi_points):
                # Calculate distance in Display Coords for better UX
                px_disp = p_orig[0] / scale_x
                py_disp = p_orig[1] / scale_y
                
                dist = ((px_disp - x)**2 + (py_disp - y)**2)**0.5
                if dist < 20: # Threshold in pixels
                    self.dragging_point_index = i
                    return

        # If we are in calibration mode (building the points)
        if self.calibration_mode:
            # Check proximity to points being built (stored in Original Coords now)
            for i, p_orig in enumerate(self.calibration_points):
                px_disp = p_orig[0] / scale_x
                py_disp = p_orig[1] / scale_y
                
                dist = ((px_disp - x)**2 + (py_disp - y)**2)**0.5
                if dist < 20:
                    self.dragging_point_index = i
                    return
            
            # If not dragging, add new point (in Original Coords)
            if 0 <= orig_x < self.original_size[0] and 0 <= orig_y < self.original_size[1]:
                self.calibration_points.append((orig_x, orig_y))
                print(f"Điểm đã chọn (Original Coords): {orig_x}, {orig_y}")
                if len(self.calibration_points) == 4:
                    self.finish_calibration()
        
        elif self.picking_color_mode:
            if 0 <= orig_x < self.original_size[0] and 0 <= orig_y < self.original_size[1]:
                # Get color from original frame
                bgr = self.current_frame_cv[orig_y, orig_x]
                hsv_pixel = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                self.target_hsv = hsv_pixel
                print(f"Màu đã chọn: HSV={hsv_pixel}")
                self.status_label.configure(text=f"Đã chọn màu: HSV{hsv_pixel}")
                self.picking_color_mode = False
                self.video_canvas.config(cursor="arrow")
        
        elif self.setting_c_mode:
            if not hasattr(self, 'roi_points'):
                self.status_label.configure(text="Vui lòng chọn vùng phím trước!")
                return

            # We need the perspective transform matrix 'M'
            src_pts = np.array(self.roi_points, dtype="float32")
            
            # Calculate target dimensions based on current config
            # Re-calculate Regions logic to get correct width
            start_note = self.start_note_midi
            num_keys = self.num_keys
            white_key_indices = {0, 2, 4, 5, 7, 9, 11}
            current_white_idx = 0
            for i in range(num_keys):
                note = start_note + i
                if (note % 12) in white_key_indices:
                    current_white_idx += 1
            
            num_white_keys = current_white_idx
            white_key_width = 15
            target_w = num_white_keys * white_key_width
            target_h = 100
            
            dst_pts = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Transform clicked point
            pts = np.array([[[orig_x, orig_y]]], dtype="float32")
            warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
            wx, wy = warped_pt
            
            if 0 <= wx < target_w and 0 <= wy < target_h:
                # Calculate key index
                # We need to map wx back to key index.
                # Since white keys have constant width, we can find the white key index.
                clicked_white_idx = int(wx // white_key_width)
                
                # Now find which MIDI note corresponds to this white index
                # We iterate again
                found_note = None
                curr_w_idx = 0
                for i in range(num_keys):
                    note = start_note + i
                    if (note % 12) in white_key_indices:
                        if curr_w_idx == clicked_white_idx:
                            found_note = note
                            # Check if it's a black key click?
                            # Black keys are centered on boundaries.
                            # This simple logic assumes clicking the white key area.
                            # For setting Middle C, clicking the white key part of C is fine.
                            break
                        curr_w_idx += 1
                
                if found_note is not None:
                    # We want found_note to be Middle C (60)
                    # So new_start_note + i = 60
                    # But 'i' is relative to current start_note.
                    # Actually: found_note (current mapping) -> should be 60.
                    # shift = 60 - found_note
                    # new_start_note = start_note + shift
                    
                    shift = 60 - found_note
                    new_start_note = self.start_note_midi + shift
                    
                    self.start_note_midi = new_start_note
                    self.start_note_offset = new_start_note
                    
                    # Update UI
                    s_note_name = midi_to_note(self.start_note_midi)
                    e_note_name = midi_to_note(self.start_note_midi + self.num_keys - 1)
                    
                    self.start_note_entry.delete(0, "end")
                    self.start_note_entry.insert(0, s_note_name)
                    self.end_note_entry.delete(0, "end")
                    self.end_note_entry.insert(0, e_note_name)
                    
                    print(f"Đã đặt Middle C. Start Note: {self.start_note_midi} ({s_note_name})")
                    self.status_label.configure(text=f"Đã đặt Middle C. Range: {s_note_name}-{e_note_name}")
                    self.setting_c_mode = False
                    self.video_canvas.config(cursor="arrow")
                    self.display_frame(self.current_frame_cv)
            else:
                self.status_label.configure(text="Vui lòng click vào trong vùng bàn phím!")

    def on_mouse_drag(self, event):
        if self.dragging_point_index is not None:
            x, y = self.get_image_coords(event)
            
            # Convert to Original Coords
            scale_x = self.original_size[0] / self.display_w
            scale_y = self.original_size[1] / self.display_h
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            
            # Clamp to video size
            orig_x = max(0, min(orig_x, self.original_size[0] - 1))
            orig_y = max(0, min(orig_y, self.original_size[1] - 1))
            
            if hasattr(self, 'roi_points') and len(self.roi_points) == 4 and not self.calibration_mode:
                self.roi_points[self.dragging_point_index] = (orig_x, orig_y)
                self.display_frame(self.current_frame_cv)
                
            elif self.calibration_mode and self.dragging_point_index < len(self.calibration_points):
                self.calibration_points[self.dragging_point_index] = (orig_x, orig_y)
                self.display_frame(self.current_frame_cv)
        
    def on_mouse_release(self, event):
        self.dragging_point_index = None

    def start_calibration(self):
        self.calibration_mode = True
        self.picking_color_mode = False
        self.setting_c_mode = False
        self.calibration_points = []
        print("Bắt đầu chọn vùng.")
        self.status_label.configure(text="Chọn 4 điểm: Góc Trái-Trên, Phải-Trên, Phải-Dưới, Trái-Dưới")

    def start_color_picking(self):
        self.picking_color_mode = True
        self.calibration_mode = False
        self.setting_c_mode = False
        self.status_label.configure(text="Click vào phím đang sáng để chọn màu.")
        self.video_canvas.config(cursor="cross")

    def start_set_c(self):
        self.setting_c_mode = True
        self.calibration_mode = False
        self.picking_color_mode = False
        self.status_label.configure(text="Click vào phím Đô (Middle C) trên hình.")
        self.video_canvas.config(cursor="hand2")

    def finish_calibration(self):
        self.calibration_mode = False
        print("Hoàn tất chọn vùng:", self.calibration_points)
        self.status_label.configure(text="Đã chọn vùng xong.")
        
        # calibration_points are already in original video coordinates
        self.roi_points = list(self.calibration_points)
        self.preview_btn.configure(state="normal") # Enable preview button
        self.process_btn.configure(state="normal")
        self.display_frame(self.current_frame_cv)

    def preview_grid_overlay(self):
        """Show warped keyboard with grid overlay to verify alignment"""
        if not hasattr(self, 'roi_points') or not hasattr(self, 'current_frame_cv'):
            return
            
        from processor import VideoProcessor
        
        # Initialize temporary processor to get regions and warp
        processor = VideoProcessor(self.video_path, self.roi_points, num_keys=self.num_keys, start_note=self.start_note_midi)
        
        # Warp current frame
        warped = processor.warp_keyboard(self.current_frame_cv)
        
        # Create overlay
        overlay = warped.copy()
        regions = processor.get_key_regions()
        
        # Draw White keys first
        for x, y, w, h, is_black, note in regions:
            if not is_black:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 200, 200), 1) # Border only
                # Fill with low opacity
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)
                
        # Draw Black keys
        for x, y, w, h, is_black, note in regions:
            if is_black:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1) # Green border
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 100, 0), -1) # Dark green fill

        # Blend
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, warped, 1 - alpha, 0, warped)
        
        # Display in canvas
        # Similar to display_processing_preview but static
        frame_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize for display
        display_h = 200
        scale = display_h / warped.shape[0]
        display_w = int(warped.shape[1] * scale)
        
        img = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        
        self.current_image = ImageTk.PhotoImage(img)
        self.video_canvas.config(width=800, height=display_h + 20)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(400, display_h // 2, anchor="center", image=self.current_image)
        
        self.status_label.configure(text="Đang xem trước lưới phím. Nếu lệch, hãy điều chỉnh vùng chọn.")
        processor.release() # Release cap inside processor (though we didn't use it for reading)

    def start_processing(self):
        if self.is_processing:
            # Stop logic
            self.is_processing = False
            self.status_label.configure(text="Đang dừng...")
            self.process_btn.configure(state="disabled") # Disable until loop finishes
            return

        print("Bắt đầu xử lý...")
        if not hasattr(self, 'roi_points'):
            print("Vui lòng chọn vùng phím trước!")
            self.status_label.configure(text="Lỗi: Vui lòng chọn vùng phím trước!")
            return
            
        self.is_processing = True
        self.process_btn.configure(text="Dừng Xử Lý")
        self.calibrate_btn.configure(state="disabled")
        self.load_btn.configure(state="disabled")
        self.set_c_btn.configure(state="disabled")
        self.status_label.configure(text="Đang xử lý...")
        
        # Start processing thread
        threading.Thread(target=self.processing_loop, daemon=True).start()

    def processing_loop(self):
        from processor import VideoProcessor
        from detector import KeyDetector
        from midi_writer import MidiWriter
        
        # Get BPM and Time Signature
        try:
            bpm = int(self.bpm_entry.get())
        except ValueError:
            bpm = 120
            
        try:
            ts_str = self.time_sig_entry.get()
            num, den = map(int, ts_str.split('/'))
            time_sig = (num, den)
        except ValueError:
            time_sig = (4, 4)
            
        processor = VideoProcessor(self.video_path, self.roi_points, num_keys=self.num_keys, start_note=self.start_note_midi)
        detector = KeyDetector(target_hsv=self.target_hsv)
        midi_writer = MidiWriter(bpm=bpm, time_signature=time_sig)
        
        # MIDI Note mapping
        start_note = self.start_note_midi
        
        # Debouncing counters
        keys_on_frames = [0] * self.num_keys
        keys_off_frames = [0] * self.num_keys
        
        # Actual playing state (debounced)
        playing_states = [False] * self.num_keys
        
        frame_count = 0
        fps = processor.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30 # Fallback
        print(f"Video FPS: {fps}")
        
        while self.is_processing:
            frame = processor.get_frame()
            if frame is None:
                break
            
            warped = processor.warp_keyboard(frame)
            states = processor.get_key_states(warped, detector)
            
            current_time = frame_count / fps
            
            for i, is_pressed_raw in enumerate(states):
                note = start_note + i
                
                if is_pressed_raw:
                    keys_on_frames[i] += 1
                    keys_off_frames[i] = 0
                else:
                    keys_off_frames[i] += 1
                    keys_on_frames[i] = 0
                
                # Debounce Logic
                # Note On: Needs 2 consecutive frames of detection
                if not playing_states[i] and keys_on_frames[i] >= 2:
                    playing_states[i] = True
                    midi_writer.add_note_on(note, velocity=100, time=current_time)
                    print(f"Note On: {note} at {current_time:.2f}s")
                
                # Note Off: Needs 3 consecutive frames of NO detection (sustain)
                elif playing_states[i] and keys_off_frames[i] >= 3:
                    playing_states[i] = False
                    midi_writer.add_note_off(note, velocity=0, time=current_time)
        
            frame_count += 1
            
            # Update GUI (optional, might slow down)
            if frame_count % 2 == 0: # Update more frequently
                # Pass copy of frame and states
                # We pass 'states' (raw detection) to preview so user sees what computer sees
                self.after(0, lambda f=warped.copy(), s=states, p=processor: self.display_processing_preview(f, s, p))
    
        processor.release()
        self.midi_writer = midi_writer # Store for export
        self.after(0, self.finish_processing)

    def display_processing_preview(self, frame, states, processor):
        # Display the warped keyboard view with overlay
        frame_vis = frame.copy()
        
        # Get regions from processor to ensure alignment
        regions = processor.get_key_regions()
        
        # Draw White keys first (background)
        for i, (x, y, w, h, is_black, note) in enumerate(regions):
            if not is_black:
                is_pressed = states[i] if i < len(states) else False
                
                # Color: Green if pressed, White/Gray if not
                if is_pressed:
                    color = (0, 255, 0) 
                else:
                    color = (200, 200, 200)
                
                # Draw filled rect
                cv2.rectangle(frame_vis, (x, y), (x + w, y + h), color, -1)
                # Draw border
                cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (50, 50, 50), 1)
                
                # Draw Note Name for C
                if note % 12 == 0:
                    octave = (note // 12) - 1
                    cv2.putText(frame_vis, f"C{octave}", (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Draw Black keys second (foreground)
        for i, (x, y, w, h, is_black, note) in enumerate(regions):
            if is_black:
                is_pressed = states[i] if i < len(states) else False
                
                if is_pressed:
                    color = (0, 200, 0)
                else:
                    color = (30, 30, 30)
                
                cv2.rectangle(frame_vis, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        # Convert to RGB for Tkinter
        frame_rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize for better visibility in GUI if needed, but keep aspect ratio
        # Target height is 100 in processor. Let's scale up slightly.
        display_h = 150
        scale = display_h / frame.shape[0]
        display_w = int(frame.shape[1] * scale)
        
        img = img.resize((display_w, display_h), Image.Resampling.NEAREST)
        
        self.current_image = ImageTk.PhotoImage(img)
        self.video_canvas.config(width=800, height=display_h + 20)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(400, display_h // 2, anchor="center", image=self.current_image)

    def finish_processing(self):
        self.is_processing = False
        self.status_label.configure(text="Xử lý hoàn tất. Sẵn sàng xuất MIDI.")
        self.process_btn.configure(state="normal", text="Bắt Đầu Xử Lý")
        self.calibrate_btn.configure(state="normal")
        self.load_btn.configure(state="normal")
        self.set_c_btn.configure(state="normal")
        self.export_btn.configure(state="normal")

    def export_midi(self):
        if hasattr(self, 'midi_writer'):
            file_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI Files", "*.mid")])
            if file_path:
                self.midi_writer.save(file_path)
                self.status_label.configure(text=f"Đã lưu vào {file_path}")
                print(f"Đã lưu MIDI vào {file_path}")
        else:
            print("Không có dữ liệu MIDI để xuất.")

if __name__ == "__main__":
    app = PianoTrackerApp()
    app.mainloop()
