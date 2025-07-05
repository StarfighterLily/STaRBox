import pygame
import numpy as np
import sys
import threading
import time
import os

class Assembler:
    def __init__( self ):
        pass

    def assemble( self, assembly_code ):
        lines = assembly_code.strip().splitlines()
        code = bytearray()

        # --- Opcode and Register Definitions ---
        opcodes = {
            "NOP": 0x00, "INC": 0x01, "DEC": 0x02, "MOV": 0x03,
            "ADD": 0x04, "SUB": 0x05, "JMP": 0x06, # Addr16
            "STORE": 0x07, # Addr16, Reg
            "HLT": 0xFF,
            "MOVI": 0x08,  # Reg, Imm8
            "ADDI": 0x09,  # Reg, Imm8
            "SUBI": 0x0A,  # Reg, Imm8
            "LOAD": 0x0B,  # Reg, Addr16
            "LOADI": 0x0C, # Reg, [RegAddr] (Zero-page only interpretation)
            "STOREI": 0x0D, # [RegAddr], Reg (Zero-page only interpretation)
            "CMP": 0x0E,   # Reg, Reg
            "CMPI": 0x0F,  # Reg, Imm8
            "PUSH": 0x20,  # Reg
            "POP": 0x21,   # Reg
            "CALL": 0x22,  # Addr16
            "RET": 0x23,   # Implied
            "JZ": 0x30,    # Addr16
            "JNZ": 0x31,   # Addr16
            "JC": 0x32,    # Addr16
            "JNC": 0x33,   # Addr16
            "JN": 0x34,    # Addr16
            "JNN": 0x35,   # Addr16
            "STOREIND": 0x15, # Rm (Stores Rm to address in R0:R1) - Using 0x15 now
            "LOADIND": 0x16,  # Rm (Loads from address in R0:R1 into Rm)
        }
        registers = {
            "R0": 0x10, "R1": 0x11, "R2": 0x12, "R3": 0x13, "R4": 0x14,
        }

        # --- Helper function to define constants ---
        def define_constants( lines_list ):
            constants = {}
            remaining_lines = []
            for line in lines_list:
                clean_line = line.split('#', 1)[0].strip()
                if not clean_line: continue
                parts = clean_line.split()
                if len(parts) >= 3 and parts[1].upper() == 'EQU':
                    const_name = parts[0].upper()
                    try:
                        const_val = int(parts[2], 0)
                        constants[const_name] = const_val
                    except ValueError:
                        print(f"[Assembler Error] Invalid value for EQU constant '{parts[0]}'")
                        return None, None
                else:
                    remaining_lines.append(line)
            return constants, remaining_lines

        # --- Pre-Pass: Handle EQU ---
        print("[Assembler Pre-Pass] Looking for EQU constants...")
        constants, lines = define_constants(lines)
        if constants is None: return None
        print(f"[Assembler Pre-Pass] Found constants: {constants}")

        def get_instruction_size( op_name, operands_list ):
            op_name = op_name.upper()
            op_code = opcodes.get( op_name )
            if op_code is None: return 0 # Error handled later

            size = 1 # Opcode byte

            if op_name in [ "INC", "DEC", "PUSH", "POP", "STOREIND", "LOADIND" ]: # Op + Reg
                size += 1
            elif op_name in [ "MOVI", "ADDI", "SUBI", "CMPI" ]: # Op + Reg + Imm8
                size += 2
            elif op_name in [ "MOV", "ADD", "SUB", "CMP", "LOADI", "STOREI" ]: # Op + Reg + Reg
                size += 2
            elif op_name in [ "JMP", "CALL", "JZ", "JNZ", "JC", "JNC", "JN", "JNN" ]: # Op + Addr16
                size += 2
            elif op_name in [ "LOAD", "STORE" ]: # Op + Reg + Addr16 (LOAD) or Op + Addr16 + Reg (STORE)
                size += 3
            elif op_name in [ "HLT", "RET", "NOP" ]: # Op only
                size += 0
            else:
                print( f"[WARN] Add size calculation for known opcode: {op_name}" )
                return 1 # Default if missed

            return size

        def split_operands( operand_string ):
            if not operand_string: return []
            return [ op.strip() for op in operand_string.split( ',' ) ]

        print( "[Assembler Pass 1] Starting Label Scan & Intermediate Build..." )
        labels = {}
        code_intermediate = []
        current_address = 0
        lines_with_num = list( enumerate( lines, 1 ) )

        for line_num, line in lines_with_num:
            original_line = line
            line = line.split( '#', 1 )[ 0 ].strip()
            if not line: continue
            parts = line.split()
            if not parts: continue
            instruction_part_maybe_label = parts[ 0 ]
            defined_label = None

            if instruction_part_maybe_label.endswith( ':' ):
                label = instruction_part_maybe_label[ :-1 ]
                if not label or not label.isidentifier():
                     print( f"[Assembler Error] Invalid label name '{label}' on line {line_num}" )
                     return None
                label_upper = label.upper()
                if label_upper in labels:
                    print( f"[Assembler Error] Duplicate label definition '{label}' on line {line_num}" )
                    return None
                labels[ label_upper ] = current_address
                defined_label = label_upper
                if len( parts ) == 1: continue
                else: parts = parts[ 1: ]
                if not parts: continue

            if parts:
                instruction_op = parts[ 0 ].upper()
                operand_str = ' '.join( parts[ 1: ] )
                operands = split_operands( operand_str )

                if instruction_op not in opcodes:
                     print( f"[Assembler Error] Unrecognized instruction '{instruction_op}' on line {line_num}" )
                     return None

                instr_size = get_instruction_size( instruction_op, operands )
                if instr_size == 0:
                    print( f"[Assembler Error] Could not determine size for '{instruction_op}' on line {line_num}." )
                    return None

                code_intermediate.append( {
                    'line': line_num, 'address': current_address, 'label': defined_label,
                    'op': instruction_op, 'operands': operands, 'size': instr_size
                } )
                current_address += instr_size

        print( f"[Assembler Pass 1] Scan & Intermediate Build Complete. Labels: {labels}" )
        print( f"[Assembler Pass 1] Calculated code size: {current_address} bytes." )

        print( "[Assembler Pass 2] Generating code..." )
        code = bytearray()

        def parse_register( operand, line_num, allow_indirect=False ):
             operand_upper = operand.upper(); is_indirect = False; reg_name = operand_upper
             if allow_indirect and operand_upper.startswith( '[' ) and operand_upper.endswith( ']' ):
                 is_indirect = True; reg_name = operand_upper[ 1:-1 ].strip()
             reg_code = registers.get( reg_name )
             if reg_code is None: print( f"Assembly Error (Line {line_num}): Invalid register '{operand}'" ); return None, is_indirect
             return reg_code, is_indirect

        def parse_value(operand, line_num, labels_dict, consts_dict, bits=8):
            operand_upper = operand.upper()
            if operand_upper in consts_dict:
                val = consts_dict[operand_upper]
            elif operand_upper in labels_dict:
                val = labels_dict[operand_upper]
            else:
                try:
                    val = int(operand, 0)
                except ValueError:
                    print(f"Assembly Error (Line {line_num}): Invalid immediate/label/constant '{operand}'")
                    return None
            
            max_val = (1 << bits) - 1
            if 0 <= val <= max_val:
                return val
            else:
                print(f"Assembly Error (Line {line_num}): Value '{operand}' ({val}) out of {bits}-bit range")
                return None

        for item in code_intermediate:
            line_num = item[ 'line' ]; op = item[ 'op' ]; operands = item[ 'operands' ]
            op_code = opcodes[ op ]; expected_size = item[ 'size' ]; start_len = len( code )
            code.append( op_code )

            try:
                if op in [ "INC", "DEC", "PUSH", "POP", "STOREIND", "LOADIND" ]:
                    if len( operands ) != 1: raise ValueError( "expects 1 register operand" )
                    reg_code, _ = parse_register( operands[ 0 ], line_num )
                    if reg_code is None: raise ValueError( "Invalid register" )
                    code.append( reg_code )
                elif op in [ "MOVI", "ADDI", "SUBI", "CMPI" ]:
                    if len( operands ) != 2: raise ValueError( "expects 2 operands (Reg, Value)" )
                    reg_code, _ = parse_register( operands[ 0 ], line_num )
                    val = parse_value( operands[ 1 ], line_num, labels, constants, 8 )
                    if reg_code is None or val is None: raise ValueError( "Invalid operand(s)" )
                    code.append( reg_code ); code.append( val )
                elif op in [ "MOV", "ADD", "SUB", "CMP" ]:
                    if len( operands ) != 2: raise ValueError( "expects 2 register operands" )
                    reg1_code, _ = parse_register( operands[ 0 ], line_num )
                    reg2_code, _ = parse_register( operands[ 1 ], line_num )
                    if reg1_code is None or reg2_code is None: raise ValueError( "Invalid register(s)" )
                    code.append( reg1_code ); code.append( reg2_code )
                elif op in [ "LOADI", "STOREI" ]:
                    if len( operands ) != 2: raise ValueError( "expects 2 operands (Reg, [Reg] or [Reg], Reg)" )
                    if op == "LOADI":
                        reg_data_code, _ = parse_register( operands[ 0 ], line_num )
                        reg_addr_code, is_indirect = parse_register( operands[ 1 ], line_num, allow_indirect=True )
                        if reg_data_code is None or reg_addr_code is None or not is_indirect: raise ValueError( "Invalid operands for LOADI. Use format: LOADI RegData, [RegAddr]" )
                        code.append( reg_data_code ); code.append( reg_addr_code )
                    else:
                        reg_addr_code, is_indirect = parse_register( operands[ 0 ], line_num, allow_indirect=True )
                        reg_data_code, _ = parse_register( operands[ 1 ], line_num )
                        if reg_addr_code is None or not is_indirect or reg_data_code is None: raise ValueError( "Invalid operands for STOREI. Use format: STOREI [RegAddr], RegData" )
                        code.append( reg_addr_code ); code.append( reg_data_code )
                elif op in [ "JMP", "CALL", "JZ", "JNZ", "JC", "JNC", "JN", "JNN" ]:
                    if len( operands ) != 1: raise ValueError( "expects 1 address/label operand" )
                    address = parse_value( operands[ 0 ], line_num, labels, constants, 16 )
                    if address is None: raise ValueError( "Invalid address/label" )
                    code.append( ( address >> 8 ) & 0xFF ); code.append( address & 0xFF )
                elif op == "LOAD":
                    if len( operands ) != 2: raise ValueError( "expects 2 operands (Register, Address/Label)" )
                    reg_code, _ = parse_register( operands[ 0 ], line_num )
                    address = parse_value( operands[ 1 ], line_num, labels, constants, 16 )
                    if reg_code is None or address is None: raise ValueError( "Invalid operand(s)" )
                    code.append( reg_code ); code.append( ( address >> 8 ) & 0xFF ); code.append( address & 0xFF )
                elif op == "STORE":
                    if len( operands ) != 2: raise ValueError( "expects 2 operands (Address/Label, Register)" )
                    address = parse_value( operands[ 0 ], line_num, labels, constants, 16 )
                    reg_code, _ = parse_register( operands[ 1 ], line_num )
                    if reg_code is None or address is None: raise ValueError( "Invalid operand(s)" )
                    code.append( ( address >> 8 ) & 0xFF ); code.append( address & 0xFF ); code.append( reg_code )
                elif op in [ "HLT", "RET", "NOP" ]:
                    if len( operands ) != 0: raise ValueError( "expects 0 operands" )
                else:
                     raise ValueError( f"Unhandled known opcode '{op}' in Pass 2 operand stage" )

                generated_size = len( code ) - start_len
                if generated_size != expected_size:
                     print( f"Internal Warning (Line {line_num}): Size mismatch for {op}. Expected {expected_size}, generated {generated_size}." )
            except ValueError as e:
                 print( f"Assembly Error (Line {line_num}): {e} - Instruction: '{op} {', '.join( operands )}'" )
                 return None

        print(f"[Assembler Pass 2] Completed. Final code size: {len( code )} bytes.")
        return code

class CPU:
    def __init__( self, memory_size, stack_size, gui=None, update_callback=None ):
        self.memory_size = memory_size
        self.stack_size = stack_size
        self.memory = bytearray( self.memory_size )
        self.registers = { "PC": 0, "SP": self.memory_size, "R0": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0 }
        self.flags = { 'Z': 0, 'C': 0, 'N': 0 }
        self.screen_width = 100
        self.screen_height = 100

        # --- MEMORY MAP ---
        self.stack_base = self.memory_size # 65535, 0xFFFF (assuming default settings)
        self.stack_limit = self.memory_size - self.stack_size # 65023, 0xFFDF
        self.screen_address = self.stack_limit - ( 100 * 100 ) # 55023, 0xD6EF
        self.keyboard_data_address = self.screen_address - 1 # 55022, 0xD6EE
        self.keyboard_status_address = self.screen_address - 2 # 55021, 0xD6ED
        # --- END MEMORY MAP ---

        self.running = False
        self.stop_event = threading.Event()
        self.gui = gui
        self.update_callback = update_callback
        self.instructions_per_check = 10000

        self.register_names = {
             0x10: "R0", 0x11: "R1", 0x12: "R2", 0x13: "R3", 0x14: "R4"
        }
        self.register_codes = { v: k for k, v in self.register_names.items() }

    def _set_flags( self, result, carry_val=None ):
        result &= 0xFF; self.flags[ 'Z' ] = 1 if result == 0 else 0
        self.flags[ 'N' ] = 1 if ( result & 0x80 ) else 0
        if carry_val is not None: self.flags[ 'C' ] = 1 if carry_val else 0

    def _check_stack_push( self, bytes_to_push=1 ):
        if self.registers[ 'SP' ] - bytes_to_push < self.stack_limit:
            print( f"Stack Overflow Error. Halting." ); self.running = False; return False
        return True

    def _check_stack_pop( self, bytes_to_pop=1 ):
        if self.registers[ 'SP' ] + bytes_to_pop > self.stack_base:
             print( f"Stack Underflow Error. Halting." ); self.running = False; return False
        if self.registers[ 'SP' ] < self.stack_limit or self.registers[ 'SP' ] >= self.stack_base:
             print( f"Stack Invalid SP Error. Halting." ); self.running = False; return False
        return True

    def load_program( self, program_code, start_address ):
        if start_address + len( program_code ) > len( self.memory ): return False
        for i, byte in enumerate( program_code ): self.memory[ start_address + i ] = byte & 255
        return True

    def execute_instruction( self ):
        if self.stop_event.is_set(): self.running = False; return
        pc = self.registers[ "PC" ]
        if not ( 0 <= pc < self.memory_size ):
            print( f"PC Error. Halting." ); self.running = False; return

        instruction = self.memory[ pc ]
        pc_increment = 1

        def fetch_addr16( offset=1 ):
            if pc + offset + 1 < self.memory_size:
                hi = self.memory[ pc + offset ]; lo = self.memory[ pc + offset + 1 ]; return ( hi << 8 ) | lo
            print( f"Fetch Addr16 bounds error. Halting." ); self.running = False; return None

        def fetch_reg_code( offset=1 ):
            if pc + offset < self.memory_size:
                 code = self.memory[ pc + offset ]
                 if code in self.register_names: return code
                 else: print( f"Invalid register code 0x{code:02X}. Halting." ); self.running = False; return None
            print( f"Fetch RegCode bounds error. Halting." ); self.running = False; return None

        def fetch_imm8( offset=1 ):
             if pc + offset < self.memory_size: return self.memory[ pc + offset ]
             print( f"Fetch Imm8 bounds error. Halting." ); self.running = False; return None

        try:
            if instruction == 0x00: # NOP
                pass
            elif instruction == 0x01: # INC Reg
                reg_code = fetch_reg_code( 1 )
                if reg_code is not None: reg_name = self.register_names[ reg_code ]; val = self.registers[ reg_name ]; result = ( val + 1 ) & 0xFF; self._set_flags( result ); self.registers[ reg_name ] = result; pc_increment = 2
            elif instruction == 0x02: # DEC Reg
                reg_code = fetch_reg_code( 1 )
                if reg_code is not None: reg_name = self.register_names[ reg_code ]; val = self.registers[ reg_name ]; result = ( val - 1 ) & 0xFF; self._set_flags( result ); self.registers[ reg_name ] = result; pc_increment = 2
            elif instruction == 0x03: # MOV Reg1, Reg2
                reg1 = fetch_reg_code( 1 ); reg2 = fetch_reg_code( 2 )
                if reg1 is not None and reg2 is not None: result = self.registers[ self.register_names[ reg2 ] ]; self.registers[ self.register_names[ reg1 ] ] = result; self._set_flags( result ); pc_increment = 3
            elif instruction == 0x04: # ADD Reg1, Reg2
                reg1 = fetch_reg_code( 1 ); reg2 = fetch_reg_code( 2 )
                if reg1 is not None and reg2 is not None: val1 = self.registers[ self.register_names[ reg1 ] ]; val2 = self.registers[ self.register_names[ reg2 ] ]; res16 = val1 + val2; res8 = res16 & 0xFF; carry = 1 if res16 > 0xFF else 0; self._set_flags( res8, carry ); self.registers[ self.register_names[ reg1 ] ] = res8; pc_increment = 3
            elif instruction == 0x05: # SUB Reg1, Reg2
                reg1 = fetch_reg_code( 1 ); reg2 = fetch_reg_code( 2 )
                if reg1 is not None and reg2 is not None: val1 = self.registers[ self.register_names[ reg1 ] ]; val2 = self.registers[ self.register_names[ reg2 ] ]; res8 = ( val1 - val2 ) & 0xFF; carry = 1 if val1 >= val2 else 0; self._set_flags( res8, carry ); self.registers[ self.register_names[ reg1 ] ] = res8; pc_increment = 3
            elif instruction == 0x06: # JMP Addr16
                addr = fetch_addr16( 1 )
                if addr is not None: self.registers[ "PC" ] = addr; pc_increment = 0
            elif instruction == 0x07: # STORE Addr16, Reg
                address = fetch_addr16( 1 ); reg_code = fetch_reg_code( 3 )
                if address is not None and reg_code is not None:
                    if 0 <= address < self.memory_size:
                        val = self.registers[ self.register_names[ reg_code ] ]; self.memory[ address ] = val
                        if self.screen_address <= address < self.screen_address + ( self.screen_width * self.screen_height ):
                            if self.gui: self.gui.update_pixel( ( address - self.screen_address ) % self.screen_width, ( address - self.screen_address ) // self.screen_width, val )
                        pc_increment = 4
                    else: print( f"STORE bounds error. Halting." ); self.running = False; pc_increment = 0
            elif instruction == 0x08: # MOVI Reg, Imm8
                reg = fetch_reg_code( 1 ); imm = fetch_imm8( 2 )
                if reg is not None and imm is not None: self.registers[ self.register_names[ reg ] ] = imm; self._set_flags( imm ); pc_increment = 3
            elif instruction == 0x09: # ADDI Reg, Imm8
                reg = fetch_reg_code( 1 ); imm = fetch_imm8( 2 )
                if reg is not None and imm is not None: val1 = self.registers[ self.register_names[ reg ] ]; res16 = val1 + imm; res8 = res16 & 0xFF; carry = 1 if res16 > 0xFF else 0; self._set_flags( res8, carry ); self.registers[ self.register_names[ reg ] ] = res8; pc_increment = 3
            elif instruction == 0x0A: # SUBI Reg, Imm8
                reg = fetch_reg_code( 1 ); imm = fetch_imm8( 2 )
                if reg is not None and imm is not None: val1 = self.registers[ self.register_names[ reg ] ]; res8 = ( val1 - imm ) & 0xFF; carry = 1 if val1 >= imm else 0; self._set_flags( res8, carry ); self.registers[ self.register_names[ reg ] ] = res8; pc_increment = 3
            elif instruction == 0x0B: # LOAD Reg, Addr16
                reg = fetch_reg_code( 1 ); addr = fetch_addr16( 2 )
                if reg is not None and addr is not None:
                    if 0 <= addr < self.memory_size:
                        val = self.memory[ addr ]
                        self.registers[ self.register_names[ reg ] ] = val
                        self._set_flags( val )
                        pc_increment = 4
                    else:
                        print( f"LOAD bounds error. Halting." )
                        self.running = False
                        pc_increment = 0
            elif instruction == 0x0C: # LOADI RegData, [RegAddr]
                reg_d = fetch_reg_code( 1 ); reg_a = fetch_reg_code( 2 )
                if reg_d is not None and reg_a is not None: addr = self.registers[ self.register_names[ reg_a ] ]; val = self.memory[ addr ]; self.registers[ self.register_names[ reg_d ] ] = val; self._set_flags( val ); pc_increment = 3
            elif instruction == 0x0D: # STOREI [RegAddr], RegData
                reg_a = fetch_reg_code( 1 ); reg_d = fetch_reg_code( 2 )
                if reg_a is not None and reg_d is not None: addr = self.registers[ self.register_names[ reg_a ] ]; val = self.registers[ self.register_names[ reg_d ] ]; self.memory[ addr ] = val; pc_increment = 3
            elif instruction == 0x0E: # CMP Reg1, Reg2
                reg1 = fetch_reg_code( 1 ); reg2 = fetch_reg_code( 2 )
                if reg1 is not None and reg2 is not None: val1 = self.registers[ self.register_names[ reg1 ] ]; val2 = self.registers[ self.register_names[ reg2 ] ]; res8 = ( val1 - val2 ) & 0xFF; carry = 1 if val1 >= val2 else 0; self._set_flags( res8, carry ); pc_increment = 3
            elif instruction == 0x0F: # CMPI Reg, Imm8
                reg = fetch_reg_code( 1 ); imm = fetch_imm8( 2 )
                if reg is not None and imm is not None: val1 = self.registers[ self.register_names[ reg ] ]; res8 = ( val1 - imm ) & 0xFF; carry = 1 if val1 >= imm else 0; self._set_flags( res8, carry ); pc_increment = 3
            elif instruction == 0x15: # STOREIND Rm
                reg_code = fetch_reg_code( 1 )
                if reg_code is not None:
                    addr_hi = self.registers[ 'R0' ]; addr_lo = self.registers[ 'R1' ]; address = ( addr_hi << 8 ) | addr_lo
                    if 0 <= address < self.memory_size:
                        data = self.registers[ self.register_names[ reg_code ] ]; self.memory[ address ] = data
                        if self.screen_address <= address < self.screen_address + ( self.screen_width * self.screen_height ):
                            if self.gui: self.gui.update_pixel( ( address - self.screen_address ) % self.screen_width, ( address - self.screen_address ) // self.screen_width, data )
                        pc_increment = 2
                    else: print( f"STOREIND bounds error. Halting." ); self.running = False; pc_increment = 0
            elif instruction == 0x16: # LOADIND Rm
                reg_code = fetch_reg_code( 1 )
                if reg_code is not None:
                    addr_hi = self.registers[ 'R0' ]; addr_lo = self.registers[ 'R1' ]; address = ( addr_hi << 8 ) | addr_lo
                    if 0 <= address < self.memory_size:
                        data = self.memory[ address ]; self.registers[ self.register_names[ reg_code ] ] = data; self._set_flags( data )
                        pc_increment = 2
                    else: print( f"LOADIND bounds error. Halting." ); self.running = False; pc_increment = 0
            elif instruction == 0x20: # PUSH Reg
                reg = fetch_reg_code( 1 )
                if reg is not None and self._check_stack_push( 1 ): val = self.registers[ self.register_names[ reg ] ]; self.registers[ 'SP' ] -= 1; self.memory[ self.registers[ 'SP' ] ] = val; pc_increment = 2
            elif instruction == 0x21: # POP Reg
                reg = fetch_reg_code( 1 )
                if reg is not None and self._check_stack_pop( 1 ): val = self.memory[ self.registers[ 'SP' ] ]; self.registers[ 'SP' ] += 1; self.registers[ self.register_names[ reg ] ] = val; self._set_flags( val ); pc_increment = 2
            elif instruction == 0x22: # CALL Addr16
                addr = fetch_addr16( 1 )
                if addr is not None and self._check_stack_push( 2 ): ret_addr = ( pc + 3 ) & 0xFFFF; self.registers[ 'SP' ] -= 1; self.memory[ self.registers[ 'SP' ] ] = ( ret_addr >> 8 ) & 0xFF; self.registers[ 'SP' ] -= 1; self.memory[ self.registers[ 'SP' ] ] = ret_addr & 0xFF; self.registers[ 'PC' ] = addr; pc_increment = 0
            elif instruction == 0x23: # RET
                if self._check_stack_pop( 2 ): addr_lo = self.memory[ self.registers[ 'SP' ] ]; self.registers[ 'SP' ] += 1; addr_hi = self.memory[ self.registers[ 'SP' ] ]; self.registers[ 'SP' ] += 1; ret_addr = ( addr_hi << 8 ) | addr_lo; self.registers[ 'PC' ] = ret_addr; pc_increment = 0
            elif 0x30 <= instruction <= 0x35: # Conditional Jumps
                cond = False
                if instruction == 0x30: cond = ( self.flags[ 'Z' ] == 1 ) # JZ
                elif instruction == 0x31: cond = ( self.flags[ 'Z' ] == 0 ) # JNZ
                elif instruction == 0x32: cond = (self.flags[ 'C' ] == 1) # JC
                elif instruction == 0x33: cond = (self.flags[ 'C' ] == 0) # JNC
                elif instruction == 0x34: cond = (self.flags[ 'N' ] == 1) # JN
                elif instruction == 0x35: cond = (self.flags[ 'N' ] == 0) # JNN
                addr = fetch_addr16( 1 )
                if addr is not None:
                    if cond: self.registers[ 'PC' ] = addr; pc_increment = 0
                    else: pc_increment = 3
            elif instruction == 0xFF: # HALT
                self.running = False; pc_increment = 0
            else:
                print( f"Unknown instruction 0x{instruction:02X}. Halting." ); self.running = False; pc_increment = 0

        except Exception as e:
            print( f"CRITICAL CPU ERROR during execution of 0x{instruction:02X} at PC=0x{pc:04X}: {e}" )
            import traceback; traceback.print_exc()
            self.running = False; pc_increment = 0

        if self.running and pc_increment > 0:
            self.registers["PC"] = ( pc + pc_increment ) & 0xFFFF
        
    def run( self ):
        self.running = True; self.stop_event.clear()
        instructions_executed_total = 0; instructions_since_last_check = 0
        start_pc = self.registers[ 'PC' ]
        print( f"CPU starting execution from PC=0x{start_pc:04X}" )
        while self.running:
            self.execute_instruction()
            if not self.running: break
            instructions_executed_total += 1; instructions_since_last_check += 1
            if instructions_since_last_check >= self.instructions_per_check:
                if self.stop_event.is_set(): self.running = False
                instructions_since_last_check = 0
        final_pc = self.registers[ 'PC' ]
        status = "Stopped by event." if self.stop_event.is_set() else "Halted normally or due to error."
        print( f"CPU execution finished. {status} Final PC: 0x{final_pc:04X}. Total instructions executed: {instructions_executed_total}" )
        if self.update_callback: self.update_callback()

    def get_stack_as_string( self, max_entries=8 ):
        stack_str = f"--- Stack Top (SP=0x{self.registers[ 'SP' ]:04X}) ---\n"; count = 0
        start_addr = self.registers[ 'SP' ]; end_addr = min( self.stack_base, start_addr + max_entries )
        if start_addr >= self.stack_base: stack_str += "(SP at/above base)\n"
        for addr in range( start_addr, end_addr ):
             if addr < self.stack_limit: continue
             val = self.memory[ addr ]; marker = "<-- SP" if addr == start_addr else ""
             stack_str += f"[0x{addr:04X}]: 0x{val:02X} {marker}\n"; count += 1
        if count == 0 and start_addr < self.stack_base : stack_str += "(Empty or SP below range shown)\n"
        return stack_str

    def get_memory_as_string( self, start_address, num_bytes, bytes_per_line=8 ):
        start = max( 0, start_address ); end = min( start + num_bytes, len( self.memory ) )
        mem_str = f"--- Mem Dump: 0x{start:04X}-0x{end-1:04X} ---\n"; line = ""
        for i in range( start, end ):
            if i % bytes_per_line == 0:
                if line: mem_str += line + "\n"
                line = f"0x{i:04X}: "
            line += f"{self.memory[i]:02X} "
        if line: mem_str += line.rstrip()
        return mem_str

class SimulatorGUI:
    def __init__( self, cpu ):
        print( "[GUI Init] Initializing Pygame..."); pygame.init(); print("[GUI Init] Pygame Initialized." )
        self.cpu = cpu
        self.cpu.gui = self
        self.cpu.update_callback = self.update_gui_callback
        self.cpu_thread = None
        self.screen_width = 100; self.screen_height = 100; self.pixel_size = 4
        self.info_panel_width = 270
        self.total_width = self.screen_width * self.pixel_size + self.info_panel_width
        self.total_height = self.screen_height * self.pixel_size
        print( f"[GUI Init] Setting display mode ({self.total_width}x{self.total_height})..." );
        try: self.screen = pygame.display.set_mode( ( self.total_width, self.total_height ) ); pygame.display.set_caption( "ST.A.R.Box CPU Simulator" )
        except Exception as e: print( f"[GUI Init] ERROR setting display mode: {e}" ); pygame.quit(); sys.exit()
        print( "[GUI Init] Display mode set." )
        print( "[GUI Init] Loading font..." );
        try: self.font = pygame.font.SysFont( "monospace", 14 ); self.font_small = pygame.font.SysFont( "monospace", 12 ); print( "[GUI Init] Monospace font loaded." )
        except Exception: print( "[GUI Init] Monospace font not found." ); self.font = pygame.font.Font( None, 20 ); self.font_small = pygame.font.Font( None, 16 )
        self.screen_surface = pygame.Surface( ( self.screen_width * self.pixel_size, self.screen_height * self.pixel_size ) )
        self.screen_surface.fill( ( 0,0,0 ) )
        self.colors = [ ( 0,0,0 ), ( 128,0,0 ), ( 0,128,0 ), ( 128,128,0 ), ( 0,0,128 ), ( 128,0,128 ), ( 0,128,128 ), ( 192,192,192 ), ( 64,64,64 ), ( 255,0,0 ), ( 0,255,0 ), ( 255,255,0 ), ( 0,0,255 ), ( 255,0,255 ), ( 0,255,255 ), ( 255,255,255 ) ]
        self.running_gui = True; self.status_message = "Idle. Press 'home' to load/run."
        self.needs_display_update = True
        print( "[GUI Init] Initialization complete." )

    def update_pixel( self, col, row, color_index ):
        if 0 <= col < self.screen_width and 0 <= row < self.screen_height:
            px, py = col * self.pixel_size, row * self.pixel_size
            color = self.colors[ color_index ] if 0 <= color_index < len( self.colors ) else ( 255, 0, 0 )
            pygame.draw.rect( self.screen_surface, color, ( px, py, self.pixel_size, self.pixel_size ) )
            self.needs_display_update = True

    def update_gui_callback( self ): self.status_message = f"CPU Halted. Final PC: 0x{self.cpu.registers.get( 'PC', 0 ):04X}"; self.needs_display_update = True; print( "[CPU Callback] CPU thread finished." )

    def run_gui_loop( self ):
        print( "[GUI Loop] Starting GUI loop..." ); clock = pygame.time.Clock()
        while self.running_gui:
            for event in pygame.event.get():
                 if event.type == pygame.QUIT: print( "[GUI Event] QUIT received." ); self.stop_simulator(); self.running_gui = False; break
                 if event.type == pygame.KEYDOWN:
                     print( f"[GUI Event] KEYDOWN: {event.key} ({pygame.key.name( event.key )})" )
                     if event.key == pygame.K_HOME: print( "[GUI Action] 'home' key pressed." ); self.run_simulator_from_input()
                     elif event.key == pygame.K_DELETE: print( "[GUI Action] 'delete' key pressed." ); self.stop_simulator()
                     else:
                        # --- KEYBOARD INPUT HANDLING ---
                        # Check if the CPU is ready for a new key
                        if self.cpu.memory[self.cpu.keyboard_status_address] == 0:
                            # Check for valid, printable ASCII characters
                            if event.unicode and 32 <= ord(event.unicode) <= 126 or event.key == pygame.K_RETURN:
                                key_code = ord(event.unicode) if event.unicode else 13 # Use 13 for Enter
                                self.cpu.memory[self.cpu.keyboard_data_address] = key_code
                                self.cpu.memory[self.cpu.keyboard_status_address] = 1 # Set status to "key ready"
                                print(f"[GUI Keyboard] Sent key '{chr(key_code)}' (0x{key_code:02X}) to CPU.")
                                self.needs_display_update = True # Update memory view
                            elif event.key == pygame.K_BACKSPACE:
                                self.cpu.memory[self.cpu.keyboard_data_address] = 8 # ASCII for backspace
                                self.cpu.memory[self.cpu.keyboard_status_address] = 1
                                print(f"[GUI Keyboard] Sent key 'BACKSPACE' (0x08) to CPU.")
                                self.needs_display_update = True
                        else:
                            print("[GUI Keyboard] CPU not ready for input (Status=1). Keypress ignored.")
                        # --- END KEYBOARD INPUT HANDLING ---

            if not self.running_gui: break

            if self.needs_display_update or self.cpu.running:
                 self.update_gui_display()
                 self.needs_display_update = False

            clock.tick( 60 )

        print( "[GUI Loop] Exiting GUI loop." ); pygame.quit(); print( "[GUI Loop] Pygame quit." )

    def run_simulator_from_input( self ):
        print( "[Sim Control] Attempting to run simulator from input..." ); file_path = "program.asm"
        print( f"[Sim Control] Looking for assembly file: {os.path.abspath( file_path )}" )
        try:
             with open( file_path, "r" ) as f: code_text = f.read()
             print( f"[Sim Control] Loaded code from {file_path}" )
        except Exception as e: print( f"[Sim Control] ERROR reading {file_path}: {e}" ); self.status_message = f"Error reading {file_path}"; self.needs_display_update=True; return

        print( "[Sim Control] Assembling code..." ); assembler = Assembler(); assembled_code = assembler.assemble( code_text )
        if assembled_code is None: print( "[Sim Control] Assembly failed." ); self.status_message = "Assembly Error."; self.needs_display_update=True; return
        print( f"[Sim Control] Assembly successful. Len: {len( assembled_code )} bytes." )

        if self.cpu_thread and self.cpu_thread.is_alive(): print( "[Sim Control] Stopping existing CPU thread..." ); self.stop_simulator()

        print( "[Sim Control] Resetting CPU..." ); start_address = 0
        self.cpu.__init__( self.cpu.memory_size, self.cpu.stack_size, self.cpu.gui, self.cpu.update_callback )
        # Also clear the screen buffer on reset
        self.screen_surface.fill((0,0,0))
        print( "[Sim Control] CPU state and screen reset." )

        print( f"[Sim Control] Loading program @ 0x{start_address:04X}..." );
        if self.cpu.load_program( assembled_code, start_address ):
            self.cpu.registers[ "PC" ] = start_address; print( "[Sim Control] Program loaded. Starting CPU thread..." )
            self.status_message = f"Running from 0x{start_address:04X}..."; self.needs_display_update=True
            self.cpu_thread = threading.Thread( target=self.cpu.run, daemon=True ); self.cpu_thread.start()
            print( "[Sim Control] CPU thread started." )
        else: print( "[Sim Control] Error loading program." ); self.status_message = "Error loading program."; self.needs_display_update=True

    def stop_simulator( self ):
        print( "[Sim Control] Stop simulator requested." )
        if self.cpu_thread and self.cpu_thread.is_alive():
            print( "[Sim Control] Signaling CPU thread stop..." ); self.cpu.stop_event.set()
            self.cpu_thread.join( timeout=1.0 )
            if self.cpu_thread.is_alive(): print( "[Sim Control] WARNING: CPU join timeout!" )
            else: print( "[Sim Control] CPU thread joined." )
        else: print( "[Sim Control] CPU thread not running." )
        self.status_message = "CPU Stopped."; self.needs_display_update=True
        self.cpu_thread = None

    def update_gui_display( self ):
        self.screen.blit( self.screen_surface, ( 0, 0 ) )
        info_panel_rect = pygame.Rect( self.screen_width * self.pixel_size, 0, self.info_panel_width, self.total_height )
        pygame.draw.rect( self.screen, ( 50, 50, 50 ), info_panel_rect )
        info_x = self.screen_width * self.pixel_size + 10; y = 10
        status_surf=self.font.render( self.status_message, True, ( 255,255,0 ) ); self.screen.blit( status_surf,( info_x, y ) ); y += 25
        reg_title=self.font.render( "Registers:", True, ( 200,200,200 ) ); self.screen.blit( reg_title,( info_x, y ) ); y += 18
        for reg, val in self.cpu.registers.items():
            txt = f"{reg:<3}: 0x{val:04X}" if reg in [ "PC", "SP" ] else f"{reg:<3}: 0x{val:02X} ({val})"
            surf = self.font_small.render(txt, True, ( 255,255,255 ) ); self.screen.blit( surf,( info_x, y ) ); y += 16
        flags_str = f"FLAGS: Z={self.cpu.flags[ 'Z' ]} N={self.cpu.flags[ 'N' ]} C={self.cpu.flags[ 'C' ]}"; flags_surf = self.font_small.render( flags_str, True, ( 255,255,255 ) ); self.screen.blit( flags_surf, ( info_x, y ) ); y += 20
        stack_title = self.font.render( "Stack:", True, ( 200,200,200 ) ); self.screen.blit( stack_title, ( info_x, y ) ); y += 18
        stack_lines = self.cpu.get_stack_as_string( max_entries = 6 ).splitlines(); stack_lines = stack_lines[ 1: ]
        for line in stack_lines:
             surf = self.font_small.render( line, True, ( 255,255,255 ) ); self.screen.blit( surf, ( info_x, y ) ); y += 16
             if y > self.total_height - 150: break
        y += 5; mem_title = self.font.render( "Memory (Around PC):", True, ( 200,200,200 ) ); self.screen.blit( mem_title, ( info_x, y ) ); y += 18
        pc = self.cpu.registers.get( "PC", 0 ); mem_start = max( 0, pc - 8 ); mem_bytes = 48
        mem_lines = self.cpu.get_memory_as_string( mem_start, mem_bytes, bytes_per_line = 8 ).splitlines(); mem_lines = mem_lines[ 1: ]
        for line in mem_lines:
            color=( 255,255,255 ); is_pc_line = False
            try:
                 line_addr = int( line.split( ':', 1 )[ 0 ], 16 );
                 if line_addr <= pc < line_addr + 8: color=( 0,255,0 ); is_pc_line = True
            except: pass
            pc_marker = " <-- PC" if is_pc_line else ""
            surf = self.font_small.render( line + pc_marker, True, color ); self.screen.blit( surf, ( info_x, y ) ); y += 16
            if y > self.total_height - 20: break

        pygame.display.flip()

if __name__ == "__main__":
    print( "[Main] Script starting..." )
    memory_size = 65536
    stack_size = 512
    print( "[Main] Creating CPU..." )
    cpu = CPU( memory_size, stack_size, gui=None, update_callback=None )
    print( "[Main] Creating SimulatorGUI..." )
    gui = SimulatorGUI( cpu )
    print( "[Main] Starting GUI event loop..." )
    gui.run_gui_loop()
    print( "[Main] GUI loop finished." )
    print( "[Main] Script finished." )