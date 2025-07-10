# Standard library imports for the GUI
import sys
import threading
import os
import tkinter as tk
from tkinter import filedialog

# Third-party libraries for graphics and computation
import pygame
import numpy as np

# Local application imports
from cpu import CPU
from assembler import Assembler, AssemblyError

class SimulatorGUI:
    def __init__( self, cpu ):
        print( "[GUI Init] Initializing Pygame..."); pygame.init(); print("[GUI Init] Pygame Initialized." )
        self.cpu = cpu
        self.cpu.gui = self
        self.cpu.update_callback = self.update_gui_callback
        self.cpu_thread = None
        self.screen_width = 100; self.screen_height = 100; self.pixel_size = 4
        self.info_panel_width = 320
        self.total_width = self.screen_width * self.pixel_size + self.info_panel_width
        self.total_height = self.screen_height * self.pixel_size
        print( f"[GUI Init] Setting display mode ({self.total_width}x{self.total_height})..." );
        try: self.screen = pygame.display.set_mode( ( self.total_width, self.total_height ) ); pygame.display.set_caption( "ST.A.R.Box CPU Simulator" )
        except Exception as e: print( f"[GUI Init] ERROR setting display mode: {e}" ); pygame.quit(); sys.exit()
        print( "[GUI Init] Display mode set." )
        print( "[GUI Init] Loading font..." );
        try: self.font = pygame.font.SysFont( "monospace", 14 ); self.font_small = pygame.font.SysFont( "monospace", 12 ); print( "[GUI Init] Monospace font loaded." )
        except Exception: print( "[GUI Init] Monospace font not found." ); self.font = pygame.font.Font( None, 20 ); self.font_small = pygame.font.Font( None, 16 )
        self.vram_surface = pygame.Surface((self.screen_width, self.screen_height))
        self.info_panel_surface = pygame.Surface((self.info_panel_width, self.total_height))
        self.colors = [ ( 0,0,0 ), ( 128,0,0 ), ( 0,128,0 ), ( 128,128,0 ), ( 0,0,128 ), ( 128,0,128 ), ( 0,128,128 ), ( 192,192,192 ), ( 64,64,64 ), ( 255,0,0 ), ( 0,255,0 ), ( 255,255,0 ), ( 0,0,255 ), ( 255,0,255 ), ( 0,255,255 ), ( 255,255,255 ) ]
        self.palette_array = np.array(self.colors, dtype=np.uint8)
        self.rebuild_memory_view()
        self.running_gui = True
        self.status_message = "Load a program to begin."
        self.code_loaded = False
        self.info_dirty = True
        self.last_known_pc = -1
        self.last_known_sp = -1
        self.last_known_flags = {}
        self.last_cpu_running_state = False
        self.button_color = (80, 80, 100)
        self.button_hover_color = (110, 110, 140)
        self.button_text_color = (255, 255, 255)
        info_x = 10; button_y = 40; button_width = 90; button_height = 30; button_spacing = 10
        self.load_button_rect = pygame.Rect(info_x, button_y, button_width, button_height)
        self.run_button_rect = pygame.Rect(info_x + button_width + button_spacing, button_y, button_width, button_height)
        self.stop_button_rect = pygame.Rect(info_x + (button_width + button_spacing) * 2, button_y, button_width, button_height)
        self.tk_root = tk.Tk(); self.tk_root.withdraw()
        self.is_assembling = False
        self.ASSEMBLY_COMPLETE_EVENT = pygame.USEREVENT + 1
        self.ASSEMBLY_ERROR_EVENT = pygame.USEREVENT + 2
        print( "[GUI Init] Initialization complete." )
    def rebuild_memory_view(self):
        print("[GUI] Rebuilding memory view...")
        screen_start = self.cpu.screen_address
        self.screen_memory_view = np.frombuffer(self.cpu.memory, dtype=np.uint8, count=(self.screen_width * self.screen_height), offset=screen_start).reshape((self.screen_height, self.screen_width))
    def update_gui_callback( self ): 
        self.status_message = f"CPU Halted. Final PC: 0x{self.cpu.registers.get( 'PC', 0 ):04X}"; self.info_dirty = True; print( "[CPU Callback] CPU thread finished." )
    def run_gui_loop( self ):
        print( "[GUI Loop] Starting GUI loop..." ); clock = pygame.time.Clock()
        while self.running_gui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print( "[GUI Event] QUIT received." ); self.stop_simulator(); self.running_gui = False; break
                elif event.type == self.ASSEMBLY_COMPLETE_EVENT:
                    self._on_assembly_complete(event.file_path, event.code)
                elif event.type == self.ASSEMBLY_ERROR_EVENT:
                    self._on_assembly_error(event.message)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        info_panel_x_start = self.screen_width * self.pixel_size
                        mouse_pos_relative = (event.pos[0] - info_panel_x_start, event.pos[1])
                        if self.load_button_rect.collidepoint(mouse_pos_relative): self.load_program_from_dialog()
                        elif self.run_button_rect.collidepoint(mouse_pos_relative): self.run_cpu()
                        elif self.stop_button_rect.collidepoint(mouse_pos_relative): self.stop_simulator()
                if event.type == pygame.KEYDOWN:
                    key_code = 0
                    if event.unicode and 32 <= ord(event.unicode) <= 126: key_code = ord(event.unicode)
                    elif event.key == pygame.K_RETURN: key_code = 13
                    elif event.key == pygame.K_BACKSPACE: key_code = 8
                    if key_code and not self.cpu.interrupt_requests['KEYBOARD']:
                        self.cpu.memory[self.cpu.keyboard_data_address] = key_code
                        self.cpu.interrupt_requests['KEYBOARD'] = True; self.info_dirty = True
            if not self.running_gui: break
            if (self.cpu.registers['PC'] != self.last_known_pc or self.cpu.registers['SP'] != self.last_known_sp or self.cpu.flags != self.last_known_flags or self.cpu.running != self.last_cpu_running_state):
                self.info_dirty = True
            self.update_gui_display(); clock.tick( 60 )
        print( "[GUI Loop] Exiting GUI loop." ); pygame.quit()
    def _assemble_worker(self, file_path, code_text):
        try:
            assembler = Assembler()
            assembled_code = assembler.assemble(code_text)
            if assembled_code is None:
                # Handle case where assemble() returns None without an exception
                raise AssemblyError("Assembler returned no code.")

            # Success: Post an event to the main thread with the results
            completion_event = pygame.event.Event(self.ASSEMBLY_COMPLETE_EVENT, {
                "file_path": file_path,
                "code": assembled_code
            })
            pygame.event.post(completion_event)
        except Exception as e:
            # Failure: Post an error event to the main thread
            error_event = pygame.event.Event(self.ASSEMBLY_ERROR_EVENT, {
                "message": f"Assembly failed: {e}"
            })
            pygame.event.post(error_event)

    # Handler for when assembly succeeds
    def _on_assembly_complete(self, file_path, assembled_code):
        if self.cpu_thread and self.cpu_thread.is_alive():
            self.stop_simulator()

        # Reset CPU and load the newly assembled code
        self.cpu.__init__(self.cpu.memory_size, self.cpu.stack_size, self, self.update_gui_callback)
        self.rebuild_memory_view()
        self.vram_surface.fill((0, 0, 0))
        
        if self.cpu.load_program(assembled_code, 0):
            self.cpu.registers["PC"] = 0
            self.status_message = f"Loaded {os.path.basename(file_path)}. Ready."
            self.code_loaded = True
        else:
            self.status_message = "Error: Program too large for memory."
            self.code_loaded = False
        
        self.is_assembling = False
        self.info_dirty = True

    # Handler for when assembly fails
    def _on_assembly_error(self, message):
        self.status_message = message
        self.code_loaded = False
        self.is_assembling = False
        self.info_dirty = True
        
    def load_program_from_dialog(self):
        if self.is_assembling:
            self.status_message = "Assembly already in progress."
            self.info_dirty = True
            return

        file_path = filedialog.askopenfilename(title="Open Assembly File", filetypes=(("Assembly Files", "*.asm"), ("All files", "*.*")))
        
        # Bring window to focus (code from previous step)
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.SetForegroundWindow(pygame.display.get_wm_info()['window'])
        else:
            pygame.display.toggle_fullscreen()
            pygame.display.toggle_fullscreen()

        if not file_path:
            return # User cancelled

        try:
            with open(file_path, "r") as f:
                code_text = f.read()
            
            # Update status and start the background thread for assembly
            self.is_assembling = True
            self.info_dirty = True
            self.status_message = f"Assembling {os.path.basename(file_path)}..."
            threading.Thread(target=self._assemble_worker, args=(file_path, code_text), daemon=True).start()

        except Exception as e:
            self.status_message = f"Error reading file: {e}"
            self.info_dirty = True
            
    def run_cpu(self):
        if self.cpu_thread and self.cpu_thread.is_alive():
            self.status_message = "Simulator is already running."
            self.info_dirty = True
            return
        if not self.code_loaded:
            self.status_message = "No program loaded."
            self.info_dirty = True
            return

        self.cpu.registers = {"PC": 0, "SP": self.cpu.memory_size}
        for i in range(10): self.cpu.registers[f"R{i}"] = 0
        self.cpu.flags = {'Z': 0, 'C': 0, 'N': 0, 'V': 0, 'A': 0, 'I': 0}
        
        self.status_message = f"Running from 0x{self.cpu.registers['PC']:04X}..."
        self.info_dirty = True
        self.cpu_thread = threading.Thread(target=self.cpu.run, daemon=True)
        self.cpu_thread.start()
    def stop_simulator( self ):
        if self.cpu_thread and self.cpu_thread.is_alive():
            self.cpu.stop_event.set(); self.cpu_thread.join( timeout=1.0 ); self.status_message = "Simulator stopped by user."; self.info_dirty = True
        self.cpu_thread = None
    def update_gui_display( self ):
        rgb_array = self.palette_array[self.screen_memory_view]; pygame.surfarray.blit_array(self.vram_surface, np.swapaxes(rgb_array, 0, 1))
        scaled_surface = pygame.transform.scale(self.vram_surface, (self.screen_width * self.pixel_size, self.total_height)); self.screen.blit(scaled_surface, (0, 0))
        if self.info_dirty:
            self.info_panel_surface.fill((50, 50, 50)); info_x = 10; y = 10
            status_surf=self.font.render( self.status_message, True, ( 255,255,0 ) ); self.info_panel_surface.blit( status_surf,( info_x, y ) ); y += 65
            reg_title=self.font.render( "Registers:", True, ( 200,200,200 ) ); self.info_panel_surface.blit( reg_title,( info_x, y ) ); y += 18
            regs = list(self.cpu.registers.items()); num_regs_col1 = (len(regs) + 1) // 2
            for i, (reg, val) in enumerate(regs):
                col_x = info_x if i < num_regs_col1 else info_x + 140; row_y = y + ((i % num_regs_col1) * 16)
                txt = f"{reg:<3}: 0x{val:04X}" if reg in ["PC", "SP"] else f"{reg:<3}: 0x{val:02X} ({val})"; surf = self.font_small.render(txt, True, (255, 255, 255)); self.info_panel_surface.blit(surf, (col_x, row_y))
            y += num_regs_col1 * 16 + 5
            flags_str = f"FLAGS: Z={self.cpu.flags['Z']} N={self.cpu.flags['N']} C={self.cpu.flags['C']} V={self.cpu.flags['V']} A={self.cpu.flags['A']} I={self.cpu.flags['I']}"
            flags_surf = self.font_small.render( flags_str, True, ( 255,255,255 ) ); self.info_panel_surface.blit( flags_surf, ( info_x, y ) ); y += 20
            stack_title = self.font.render( "Stack:", True, ( 200,200,200 ) ); self.info_panel_surface.blit( stack_title, ( info_x, y ) ); y += 18
            stack_lines = self.cpu.get_stack_as_string( max_entries = 4 ).splitlines()[1:]
            for line in stack_lines: surf = self.font_small.render( line, True, ( 255,255,255 ) ); self.info_panel_surface.blit( surf, ( info_x, y ) ); y += 16
            y += 5; mem_title = self.font.render( "Memory (Around PC):", True, ( 200,200,200 ) ); self.info_panel_surface.blit( mem_title, ( info_x, y ) ); y += 18
            pc = self.cpu.registers.get( "PC", 0 ); mem_start = max( 0, pc - 8 ); mem_bytes = 48
            mem_lines = self.cpu.get_memory_as_string( mem_start, mem_bytes, bytes_per_line = 8 ).splitlines()[1:]
            for line in mem_lines:
                color=( 255,255,255 ); is_pc_line = False
                try:
                    line_addr = int( line.split( ':', 1 )[ 0 ], 16 );
                    if line_addr <= pc < line_addr + 8: color=( 0,255,0 ); is_pc_line = True
                except: pass
                pc_marker = " <--" if is_pc_line else ""
                surf = self.font_small.render( line + pc_marker, True, color ); self.info_panel_surface.blit( surf, ( info_x, y ) ); y += 16
            self.last_known_pc = self.cpu.registers['PC']; self.last_known_sp = self.cpu.registers['SP']; self.last_known_flags = self.cpu.flags.copy(); self.last_cpu_running_state = self.cpu.running; self.info_dirty = False
        info_panel_x_start = self.screen_width * self.pixel_size
        self.screen.blit(self.info_panel_surface, (info_panel_x_start, 0))
        mouse_pos = pygame.mouse.get_pos(); mouse_pos_relative = (mouse_pos[0] - info_panel_x_start, mouse_pos[1])
        load_color = self.button_hover_color if self.load_button_rect.collidepoint(mouse_pos_relative) else self.button_color
        pygame.draw.rect(self.screen, load_color, self.load_button_rect.move(info_panel_x_start, 0), border_radius=5)
        load_text = self.font_small.render("Load ASM", True, self.button_text_color); self.screen.blit(load_text, (self.load_button_rect.x + 15 + info_panel_x_start, self.load_button_rect.y + 8))
        run_color = self.button_hover_color if self.run_button_rect.collidepoint(mouse_pos_relative) else self.button_color
        pygame.draw.rect(self.screen, run_color, self.run_button_rect.move(info_panel_x_start, 0), border_radius=5)
        run_text = self.font_small.render("Run", True, self.button_text_color); self.screen.blit(run_text, (self.run_button_rect.x + 30 + info_panel_x_start, self.run_button_rect.y + 8))
        stop_color = self.button_hover_color if self.stop_button_rect.collidepoint(mouse_pos_relative) else self.button_color
        pygame.draw.rect(self.screen, stop_color, self.stop_button_rect.move(info_panel_x_start, 0), border_radius=5)
        stop_text = self.font_small.render("Stop", True, self.button_text_color); self.screen.blit(stop_text, (self.stop_button_rect.x + 28 + info_panel_x_start, self.stop_button_rect.y + 8))
        pygame.display.flip()

if __name__ == "__main__":
    print( "[Main] Script starting..." )
    memory_size = 65535; stack_size = 512
    print( "[Main] Creating CPU..." )
    cpu = CPU( memory_size, stack_size, gui=None, update_callback=None )
    print( f"[MEMORY] RAM Max: {cpu.memory_size}" ); print( f"[MEMORY] RAM Used: {cpu.mem_used}" ); print( f"[MEMORY] RAM Free: {cpu.mem_free}" )
    print( f"[MEMORY] Font Data: 0x{cpu.font_addr:04x}" ); print( f"[MEMORY] Kbd Status: 0x{cpu.keyboard_status_address:04x}" ); print( f"[MEMORY] Kbd Data: 0x{cpu.keyboard_data_address:04x}" )
    print( f"[MEMORY] Video Address: 0x{cpu.screen_address:04x}" ); print( f"[MEMORY] Stack: 0x{cpu.stack_limit:04x}-0x{cpu.stack_base:04x}" )
    print( "[Main] Creating SimulatorGUI..." )
    gui = SimulatorGUI( cpu )
    print( "[Main] Starting GUI event loop..." )
    gui.run_gui_loop()
    print( "[Main] GUI loop finished." )