import pygame
import numpy as np
import sys
import threading
import time
import os
import re

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
            "LOADI": 0x0C, # Reg, [RegAddr]
            "STOREI": 0x0D, # [RegAddr], Reg
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
            "STOREIND": 0x15, # [R0:R1], Rm
            "LOADIND": 0x16,  # Rm, [R0:R1]
            "AND": 0x40, "OR": 0x41, "XOR": 0x42, "NOT": 0x43,
            "ANDI": 0x44, "ORI": 0x45, "XORI": 0x46,
            "SHL": 0x47, "SHR": 0x48,
            # --- NEW ADVANCED OPCODES ---
            "MUL": 0x50,      # R_dest, R_src
            "MOV16": 0x51,    # R_pair, Imm16
            "ADD16": 0x52,    # R_pair_dest, R_pair_src
            "PUSHA": 0x53,    # Implied
            "POPA": 0x54,     # Implied
            "LOADIX": 0x55,   # R_dest, [R_pair + Imm8]
            "STOREIX": 0x56,  # [R_pair + Imm8], R_src
            "LOADSP": 0x57,   # R_dest, [SP + Imm8]
        }
        registers = {
            "R0": 0x10, "R1": 0x11, "R2": 0x12, "R3": 0x13, "R4": 0x14,
        }
        # Register pairs are identified by their high-order register (R0 or R2)
        register_pairs = { "R0": 0xA0, "R2": 0xA2 }


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

        print("[Assembler Pre-Pass] Looking for EQU constants...")
        constants, lines = define_constants(lines)
        if constants is None: return None
        print(f"[Assembler Pre-Pass] Found constants: {constants}")

        def get_instruction_size( op_name, operands_list ):
            op_name = op_name.upper()
            if op_name not in opcodes: return 0

            size = 1 # Opcode byte
            if op_name in ["INC", "DEC", "PUSH", "POP", "STOREIND", "LOADIND", "NOT"]: size += 1
            elif op_name in ["MOVI", "ADDI", "SUBI", "CMPI", "ANDI", "ORI", "XORI", "SHL", "SHR", "LOADSP"]: size += 2
            elif op_name in ["MOV", "ADD", "SUB", "CMP", "AND", "OR", "XOR", "MUL", "ADD16"]: size += 2
            elif op_name in ["JMP", "CALL", "JZ", "JNZ", "JC", "JNC", "JN", "JNN"]: size += 2
            elif op_name in ["LOAD", "STORE"]: size += 3
            elif op_name in ["HLT", "RET", "NOP", "PUSHA", "POPA"]: size += 0
            elif op_name in ["MOV16", "LOADIX", "STOREIX"]: size += 3
            else:
                print(f"[WARN] Add size calculation for known opcode: {op_name}")
                return 1
            return size

        def split_operands( operand_string ):
            if not operand_string: return []
            # This regex handles simple operands and indexed addressing like [R0 + 10]
            return [op.strip() for op in re.split(r",\s*(?![^\[]*\])", operand_string)]

        print( "[Assembler Pass 1] Starting Label Scan & Intermediate Build..." )
        labels = {}; code_intermediate = []; current_address = 0
        lines_with_num = list( enumerate( lines, 1 ) )

        for line_num, line in lines_with_num:
            line = line.split('#', 1)[0].strip()
            if not line: continue
            parts = line.split()
            if not parts: continue
            
            defined_label = None
            if parts[0].endswith(':'):
                label = parts[0][:-1].upper()
                if not label.isidentifier():
                    print(f"[Assembler Error] Invalid label name '{label}' on line {line_num}"); return None
                if label in labels:
                    print(f"[Assembler Error] Duplicate label definition '{label}' on line {line_num}"); return None
                labels[label] = current_address
                defined_label = label
                parts = parts[1:]
                if not parts: continue

            instruction_op = parts[0].upper()
            operand_str = ' '.join(parts[1:])
            operands = split_operands(operand_str)

            if instruction_op not in opcodes:
                print(f"[Assembler Error] Unrecognized instruction '{instruction_op}' on line {line_num}"); return None

            instr_size = get_instruction_size(instruction_op, operands)
            code_intermediate.append({
                'line': line_num, 'address': current_address, 'label': defined_label,
                'op': instruction_op, 'operands': operands, 'size': instr_size
            })
            current_address += instr_size

        print(f"[Assembler Pass 1] Complete. Labels: {labels}. Code size: {current_address} bytes.")
        print("[Assembler Pass 2] Generating code...")
        code = bytearray()

        def parse_register(operand, line_num):
            reg_code = registers.get(operand.upper())
            if reg_code is None: print(f"Assembly Error (Line {line_num}): Invalid register '{operand}'"); return None
            return reg_code
        
        def parse_register_pair(operand, line_num):
            pair_code = register_pairs.get(operand.upper())
            if pair_code is None: print(f"Assembly Error (Line {line_num}): Invalid register pair '{operand}' (must be R0 or R2)"); return None
            return pair_code

        def parse_value(operand, line_num, bits=8):
            operand_upper = operand.upper()
            val = constants.get(operand_upper, labels.get(operand_upper))
            if val is None:
                try: val = int(operand, 0)
                except ValueError: print(f"Assembly Error (Line {line_num}): Invalid immediate/label/constant '{operand}'"); return None
            
            max_val = (1 << bits) - 1
            if not (0 <= val <= max_val): print(f"Assembly Error (Line {line_num}): Value '{operand}' ({val}) out of {bits}-bit range"); return None
            return val

        def parse_indexed_operand(operand, line_num):
            match = re.match(r"\[\s*(R0|R2|SP)\s*\+\s*([^\]]+)\]", operand.upper())
            if not match: print(f"Assembly Error (Line {line_num}): Invalid indexed operand format '{operand}'"); return None, None, None
            
            reg_name, val_str = match.groups()
            reg_code = register_pairs.get(reg_name) if reg_name != "SP" else 0xFF # Special code for SP
            val = parse_value(val_str, line_num, 8)
            if val is None: return None, None, None
            return reg_code, val, reg_name

        for item in code_intermediate:
            line_num, op, operands = item['line'], item['op'], item['operands']
            op_code = opcodes[op]
            code.append(op_code)

            try:
                if op in ["INC", "DEC", "PUSH", "POP", "NOT"]:
                    reg = parse_register(operands[0], line_num); code.append(reg)
                elif op in ["MOVI", "ADDI", "SUBI", "CMPI", "ANDI", "ORI", "XORI", "SHL", "SHR"]:
                    reg = parse_register(operands[0], line_num); val = parse_value(operands[1], line_num, 8)
                    code.extend([reg, val])
                elif op in ["MOV", "ADD", "SUB", "CMP", "AND", "OR", "XOR", "MUL"]:
                    reg1 = parse_register(operands[0], line_num); reg2 = parse_register(operands[1], line_num)
                    code.extend([reg1, reg2])
                elif op in ["JMP", "CALL", "JZ", "JNZ", "JC", "JNC", "JN", "JNN"]:
                    addr = parse_value(operands[0], line_num, 16)
                    code.extend([(addr >> 8) & 0xFF, addr & 0xFF])
                elif op == "LOAD":
                    reg = parse_register(operands[0], line_num); addr = parse_value(operands[1], line_num, 16)
                    code.extend([reg, (addr >> 8) & 0xFF, addr & 0xFF])
                elif op == "STORE":
                    addr = parse_value(operands[0], line_num, 16); reg = parse_register(operands[1], line_num)
                    code.extend([(addr >> 8) & 0xFF, addr & 0xFF, reg])
                elif op == "LOADIND":
                    reg_dest = parse_register(operands[0], line_num); code.append(reg_dest)
                elif op == "STOREIND":
                    reg_src = parse_register(operands[0], line_num); code.append(reg_src)
                elif op == "MOV16":
                    reg_pair = parse_register_pair(operands[0], line_num); val = parse_value(operands[1], line_num, 16)
                    code.extend([reg_pair, (val >> 8) & 0xFF, val & 0xFF])
                elif op == "ADD16":
                    reg_pair1 = parse_register_pair(operands[0], line_num); reg_pair2 = parse_register_pair(operands[1], line_num)
                    code.extend([reg_pair1, reg_pair2])
                elif op == "LOADIX":
                    reg_dest = parse_register(operands[0], line_num)
                    reg_pair, offset, _ = parse_indexed_operand(operands[1], line_num)
                    code.extend([reg_dest, reg_pair, offset])
                elif op == "STOREIX":
                    reg_pair, offset, _ = parse_indexed_operand(operands[0], line_num)
                    reg_src = parse_register(operands[1], line_num)
                    code.extend([reg_pair, offset, reg_src])
                elif op == "LOADSP":
                    reg_dest = parse_register(operands[0], line_num)
                    _, offset, _ = parse_indexed_operand(operands[1], line_num)
                    code.extend([reg_dest, offset])

            except (TypeError, ValueError) as e:
                print(f"Assembly Error (Line {line_num}): Malformed operands for '{op}'. Details: {e}"); return None

        print(f"[Assembler Pass 2] Completed. Final code size: {len(code)} bytes.")
        return code

class CPU:
    def __init__( self, memory_size, stack_size, gui=None, update_callback=None ):
        self.memory_size = memory_size; self.stack_size = stack_size
        self.memory = bytearray(self.memory_size)
        self.registers = {"PC": 0, "SP": self.memory_size, "R0": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0}
        self.flags = {'Z': 0, 'C': 0, 'N': 0}
        self.screen_width = 100; self.screen_height = 100

        self.stack_base = self.memory_size
        self.stack_limit = self.memory_size - self.stack_size
        self.screen_address = self.stack_limit - (self.screen_width * self.screen_height)
        self.keyboard_data_address = self.screen_address - 1
        self.keyboard_status_address = self.screen_address - 2
        self.font_addr = self.keyboard_status_address - 760

        self.running = False; self.stop_event = threading.Event()
        self.gui = gui; self.update_callback = update_callback
        self.instructions_per_check = 100000

        self.register_names = {0x10: "R0", 0x11: "R1", 0x12: "R2", 0x13: "R3", 0x14: "R4"}
        self.register_pair_names = {0xA0: ("R0", "R1"), 0xA2: ("R2", "R3")}

    def _set_flags(self, result, is_16bit=False):
        mask = 0xFFFF if is_16bit else 0xFF
        sign_bit = 0x8000 if is_16bit else 0x80
        self.flags['Z'] = 1 if (result & mask) == 0 else 0
        self.flags['N'] = 1 if (result & sign_bit) else 0

    def _set_carry(self, val): self.flags['C'] = 1 if val else 0

    def _check_stack_op(self, size, is_push):
        new_sp = self.registers['SP'] - size if is_push else self.registers['SP'] + size
        if not (self.stack_limit <= new_sp <= self.stack_base):
            print(f"Stack {'Overflow' if is_push else 'Underflow'} Error. Halting."); self.running = False; return False
        return True

    def load_program(self, program_code, start_address):
        if start_address + len(program_code) > len(self.memory): return False
        self.memory[start_address:start_address + len(program_code)] = program_code
        return True

    def execute_instruction( self ):
        if self.stop_event.is_set(): self.running = False; return
        pc = self.registers["PC"]
        if not (0 <= pc < self.memory_size):
            print(f"PC Error. Halting."); self.running = False; return

        instruction = self.memory[pc]
        pc_increment = 1

        def fetch(offset=1, num_bytes=1):
            if pc + offset + num_bytes > self.memory_size:
                print(f"Fetch bounds error at PC=0x{pc:04X}. Halting.")
                self.running = False; return None
            if num_bytes == 1: return self.memory[pc + offset]
            return self.memory[pc + offset : pc + offset + num_bytes]

        def fetch_reg_name(offset=1):
            code = fetch(offset)
            if code is None: return None
            name = self.register_names.get(code)
            if name is None: print(f"Invalid register code 0x{code:02X} at PC=0x{pc:04X}. Halting."); self.running = False
            return name

        def fetch_pair_names(offset=1):
            code = fetch(offset)
            if code is None: return None
            names = self.register_pair_names.get(code)
            if names is None: print(f"Invalid register pair code 0x{code:02X} at PC=0x{pc:04X}. Halting."); self.running = False
            return names

        def fetch_addr16(offset=1):
            data = fetch(offset, 2)
            if data is None: return None
            return (data[0] << 8) | data[1]

        try:
            if instruction == 0x00: pc_increment = 1 # NOP
            elif instruction == 0x01: reg = fetch_reg_name(); val = self.registers[reg]; res = (val + 1) & 0xFF; self._set_flags(res); self.registers[reg] = res; pc_increment = 2 # INC
            elif instruction == 0x02: reg = fetch_reg_name(); val = self.registers[reg]; res = (val - 1) & 0xFF; self._set_flags(res); self.registers[reg] = res; pc_increment = 2 # DEC
            elif instruction == 0x03: r1, r2 = fetch_reg_name(1), fetch_reg_name(2); res = self.registers[r2]; self.registers[r1] = res; self._set_flags(res); pc_increment = 3 # MOV
            elif instruction == 0x04: r1,r2=fetch_reg_name(1),fetch_reg_name(2); v1=self.registers[r1];v2=self.registers[r2]; res16=v1+v2; self._set_carry(res16>0xFF); res8=res16&0xFF; self._set_flags(res8); self.registers[r1]=res8; pc_increment=3 # ADD
            elif instruction == 0x05: r1,r2=fetch_reg_name(1),fetch_reg_name(2); v1=self.registers[r1];v2=self.registers[r2]; self._set_carry(v1>=v2); res8=(v1-v2)&0xFF; self._set_flags(res8); self.registers[r1]=res8; pc_increment=3 # SUB
            elif instruction == 0x06: addr = fetch_addr16(); self.registers["PC"] = addr; pc_increment = 0 # JMP
            elif instruction == 0x07: addr,reg=fetch_addr16(1),fetch_reg_name(3); self.memory[addr] = self.registers[reg]; pc_increment = 4 # STORE
            elif instruction == 0x08: reg,val=fetch_reg_name(1),fetch(2); self.registers[reg]=val; self._set_flags(val); pc_increment = 3 # MOVI
            elif instruction == 0x09: reg,v2=fetch_reg_name(1),fetch(2); v1=self.registers[reg]; res16=v1+v2; self._set_carry(res16>0xFF); res8=res16&0xFF; self._set_flags(res8); self.registers[reg]=res8; pc_increment = 3 # ADDI
            elif instruction == 0x0A: reg,v2=fetch_reg_name(1),fetch(2); v1=self.registers[reg]; self._set_carry(v1>=v2); res8=(v1-v2)&0xFF; self._set_flags(res8); self.registers[reg]=res8; pc_increment = 3 # SUBI
            elif instruction == 0x0B: reg,addr=fetch_reg_name(1),fetch_addr16(2); val=self.memory[addr]; self.registers[reg]=val; self._set_flags(val); pc_increment = 4 # LOAD
            elif instruction == 0x0E: r1,r2=fetch_reg_name(1),fetch_reg_name(2); v1=self.registers[r1];v2=self.registers[r2]; self._set_carry(v1>=v2); res8=(v1-v2)&0xFF; self._set_flags(res8); pc_increment = 3 # CMP
            elif instruction == 0x0F: reg,v2=fetch_reg_name(1),fetch(2); v1=self.registers[reg]; self._set_carry(v1>=v2); res8=(v1-v2)&0xFF; self._set_flags(res8); pc_increment = 3 # CMPI
            elif instruction == 0x15: reg=fetch_reg_name(); addr=(self.registers["R0"]<<8)|self.registers["R1"]; self.memory[addr]=self.registers[reg]; pc_increment = 2 # STOREIND
            elif instruction == 0x16: reg=fetch_reg_name(); addr=(self.registers["R0"]<<8)|self.registers["R1"]; val=self.memory[addr]; self.registers[reg]=val; self._set_flags(val); pc_increment = 2 # LOADIND
            elif instruction == 0x20: reg=fetch_reg_name(); self._check_stack_op(1,True); self.registers["SP"]-=1; self.memory[self.registers["SP"]]=self.registers[reg]; pc_increment = 2 # PUSH
            elif instruction == 0x21: reg=fetch_reg_name(); self._check_stack_op(1,False); val=self.memory[self.registers["SP"]]; self.registers[reg]=val; self.registers["SP"]+=1; self._set_flags(val); pc_increment = 2 # POP
            elif instruction == 0x22: addr=fetch_addr16(); self._check_stack_op(2,True); ret=(pc+3)&0xFFFF; self.registers["SP"]-=2; self.memory[self.registers["SP"]]=(ret>>8)&0xFF; self.memory[self.registers["SP"]+1]=ret&0xFF; self.registers["PC"]=addr; pc_increment=0 # CALL
            elif instruction == 0x23: self._check_stack_op(2,False); hi=self.memory[self.registers["SP"]]; lo=self.memory[self.registers["SP"]+1]; self.registers["SP"]+=2; self.registers["PC"]=(hi<<8)|lo; pc_increment=0 # RET
            
            elif 0x30 <= instruction <= 0x35:
                cond=False; addr=fetch_addr16()
                if   instruction == 0x30: cond=(self.flags['Z']==1) # JZ
                elif instruction == 0x31: cond=(self.flags['Z']==0) # JNZ
                elif instruction == 0x32: cond=(self.flags['C']==1) # JC
                elif instruction == 0x33: cond=(self.flags['C']==0) # JNC
                elif instruction == 0x34: cond=(self.flags['N']==1) # JN
                elif instruction == 0x35: cond=(self.flags['N']==0) # JNN
                if cond: self.registers["PC"]=addr; pc_increment=0
                else: pc_increment=3
            
            elif instruction == 0x40: r1,r2=fetch_reg_name(1),fetch_reg_name(2); res=self.registers[r1]&self.registers[r2]; self._set_flags(res); self.registers[r1]=res; pc_increment=3 # AND
            elif instruction == 0x41: r1,r2=fetch_reg_name(1),fetch_reg_name(2); res=self.registers[r1]|self.registers[r2]; self._set_flags(res); self.registers[r1]=res; pc_increment=3 # OR
            elif instruction == 0x42: r1,r2=fetch_reg_name(1),fetch_reg_name(2); res=self.registers[r1]^self.registers[r2]; self._set_flags(res); self.registers[r1]=res; pc_increment=3 # XOR
            elif instruction == 0x43: reg=fetch_reg_name(); res=(~self.registers[reg])&0xFF; self._set_flags(res); self.registers[reg]=res; pc_increment=2 # NOT
            elif instruction == 0x44: reg,v2=fetch_reg_name(1),fetch(2); res=self.registers[reg]&v2; self._set_flags(res); self.registers[reg]=res; pc_increment=3 # ANDI
            elif instruction == 0x45: reg,v2=fetch_reg_name(1),fetch(2); res=self.registers[reg]|v2; self._set_flags(res); self.registers[reg]=res; pc_increment=3 # ORI
            elif instruction == 0x46: reg,v2=fetch_reg_name(1),fetch(2); res=self.registers[reg]^v2; self._set_flags(res); self.registers[reg]=res; pc_increment=3 # XORI
            
            elif instruction == 0x47: # SHL
                reg, shifts = fetch_reg_name(1), fetch(2)
                if reg and shifts is not None:
                    val = self.registers[reg]; carry = 0
                    for _ in range(shifts):
                        carry = 1 if (val & 0x80) else 0
                        val = (val << 1) & 0xFF
                    self._set_carry(carry); self.registers[reg]=val; self._set_flags(val); pc_increment=3
            elif instruction == 0x48: # SHR
                reg, shifts = fetch_reg_name(1), fetch(2)
                if reg and shifts is not None:
                    val = self.registers[reg]; carry = 0
                    for _ in range(shifts):
                        carry = 1 if (val & 0x01) else 0
                        val >>= 1
                    self._set_carry(carry); self.registers[reg]=val; self._set_flags(val); pc_increment=3
            
            elif instruction == 0x50: r1,r2=fetch_reg_name(1),fetch_reg_name(2); v1=self.registers[r1];v2=self.registers[r2]; res16=v1*v2; self._set_carry(res16>0xFF); res8=res16&0xFF; self._set_flags(res8); self.registers[r1]=res8; pc_increment=3 # MUL
            elif instruction == 0x51: (r_hi,r_lo),val=fetch_pair_names(1),fetch_addr16(2); self.registers[r_hi]=(val>>8)&0xFF; self.registers[r_lo]=val&0xFF; pc_increment=4 # MOV16
            elif instruction == 0x52: (d_hi,d_lo),(s_hi,s_lo)=fetch_pair_names(1),fetch_pair_names(2); v1=(self.registers[d_hi]<<8)|self.registers[d_lo]; v2=(self.registers[s_hi]<<8)|self.registers[s_lo]; res32=v1+v2; self._set_carry(res32>0xFFFF); res16=res32&0xFFFF; self._set_flags(res16,True); self.registers[d_hi]=(res16>>8)&0xFF; self.registers[d_lo]=res16&0xFF; pc_increment=3 # ADD16
            elif instruction == 0x53: self._check_stack_op(5,True); self.registers["SP"]-=5; regs=["R0","R1","R2","R3","R4"]; [self.memory.__setitem__(self.registers["SP"]+i, self.registers[r]) for i,r in enumerate(regs)]; pc_increment=1 # PUSHA
            elif instruction == 0x54: self._check_stack_op(5,False); regs=["R0","R1","R2","R3","R4"]; [self.registers.__setitem__(r, self.memory[self.registers["SP"]+i]) for i,r in enumerate(regs)]; self.registers["SP"]+=5; pc_increment=1 # POPA
            elif instruction == 0x55: r_dest,(p_hi,p_lo),offset=fetch_reg_name(1),fetch_pair_names(2),fetch(3); base=(self.registers[p_hi]<<8)|self.registers[p_lo]; val=self.memory[base+offset]; self.registers[r_dest]=val; self._set_flags(val); pc_increment=4 # LOADIX
            elif instruction == 0x56: (p_hi,p_lo),offset,r_src=fetch_pair_names(1),fetch(2),fetch_reg_name(3); base=(self.registers[p_hi]<<8)|self.registers[p_lo]; self.memory[base+offset]=self.registers[r_src]; pc_increment=4 # STOREIX
            elif instruction == 0x57: r_dest,offset=fetch_reg_name(1),fetch(2); val=self.memory[self.registers["SP"]+offset]; self.registers[r_dest]=val; self._set_flags(val); pc_increment=3 # LOADSP
            
            elif instruction == 0xFF: self.running = False; pc_increment = 0 # HALT
            else: print(f"Unknown instruction 0x{instruction:02X}. Halting."); self.running = False; pc_increment = 0

        except (TypeError, IndexError, KeyError) as e:
            print(f"CRITICAL CPU ERROR at PC=0x{pc:04X} (Op: 0x{instruction:02X}): {e}");
            import traceback; traceback.print_exc(); self.running = False; pc_increment = 0

        if self.running and pc_increment > 0: self.registers["PC"] = (pc + pc_increment) & 0xFFFF
        
    def run( self ):
        self.running = True; self.stop_event.clear()
        start_time = time.time()
        while self.running:
            self.execute_instruction()
            if self.stop_event.is_set(): self.running = False
        print(f"CPU execution finished. Final PC: 0x{self.registers['PC']:04X}.")
        if self.update_callback: self.update_callback()
    
    # The rest of the CPU and GUI classes remain the same...
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
                        if self.cpu.memory[self.cpu.keyboard_status_address] == 0:
                            if event.unicode and 32 <= ord(event.unicode) <= 126 or event.key == pygame.K_RETURN:
                                key_code = ord(event.unicode) if event.unicode else 13
                                self.cpu.memory[self.cpu.keyboard_data_address] = key_code
                                self.cpu.memory[self.cpu.keyboard_status_address] = 1
                                self.needs_display_update = True
                            elif event.key == pygame.K_BACKSPACE:
                                self.cpu.memory[self.cpu.keyboard_data_address] = 8
                                self.cpu.memory[self.cpu.keyboard_status_address] = 1
                                self.needs_display_update = True

            if not self.running_gui: break
            if self.needs_display_update or (self.cpu_thread and self.cpu_thread.is_alive()):
                 self.update_gui_display(); self.needs_display_update = False
            clock.tick( 60 )
        print( "[GUI Loop] Exiting GUI loop." ); pygame.quit()

    def run_simulator_from_input( self ):
        file_path = "program.asm"
        try:
             with open( file_path, "r" ) as f: code_text = f.read()
        except Exception as e: self.status_message = f"Error reading {file_path}"; self.needs_display_update=True; return

        assembler = Assembler(); assembled_code = assembler.assemble( code_text )
        if assembled_code is None: self.status_message = "Assembly Error."; self.needs_display_update=True; return

        if self.cpu_thread and self.cpu_thread.is_alive(): self.stop_simulator()

        self.cpu.__init__( self.cpu.memory_size, self.cpu.stack_size, self.cpu.gui, self.cpu.update_callback )
        self.screen_surface.fill( ( 0,0,0 ) )
        
        font = [ 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x18,0x18,0x00,0x18,0x00,0x00,0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x5A,0x24,0x7E,0x24,0x5A,0x18,0x00,0x44,0x4A,0x52,0x4A,0x48,0x00,0x00,0x00,0x6C,0x54,0x28,0x1A,0x36,0x00,0x00,0x00,0x18,0x30,0x00,0x00,0x00,0x00,0x00,0x00,0x0C,0x18,0x30,0x60,0x60,0x00,0x00,0x00,0x60,0x30,0x18,0x0C,0x0C,0x00,0x00,0x00,0x00,0x36,0xDB,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x3E,0x00,0x3E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x30,0x00,0x00,0x00,0x00,0x00,0x3E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x04,0x08,0x10,0x20,0x40,0x00,0x00,0x00,0x3C,0x42,0x42,0x42,0x42,0x42,0x3C,0x00,0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00,0x3C,0x42,0x02,0x04,0x08,0x10,0x7E,0x00,0x3C,0x42,0x02,0x1C,0x02,0x42,0x3C,0x00,0x04,0x0C,0x14,0x24,0x7E,0x04,0x04,0x00,0x7E,0x40,0x40,0x7C,0x02,0x02,0x3C,0x00,0x3C,0x40,0x40,0x7C,0x42,0x42,0x3C,0x00,0x40,0x40,0x20,0x10,0x08,0x08,0x08,0x00,0x3C,0x42,0x42,0x3C,0x42,0x42,0x3C,0x00,0x3C,0x42,0x42,0x3E,0x02,0x02,0x3C,0x00,0x00,0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x00,0x18,0x18,0x30,0x04,0x08,0x10,0x20,0x10,0x08,0x04,0x00,0x00,0x3E,0x00,0x3E,0x00,0x3E,0x00,0x00,0x20,0x10,0x08,0x04,0x08,0x10,0x20,0x00,0x3C,0x42,0x02,0x0C,0x18,0x00,0x18,0x00,0x3C,0x42,0x4A,0x4A,0x4A,0x40,0x3C,0x00,0x18,0x3C,0x66,0x66,0x7E,0x66,0x66,0x00,0x7C,0x66,0x66,0x7C,0x66,0x66,0x7C,0x00,0x3C,0x66,0x40,0x40,0x40,0x66,0x3C,0x00,0x7C,0x66,0x66,0x66,0x66,0x66,0x7C,0x00,0x7E,0x40,0x40,0x7C,0x40,0x40,0x7E,0x00,0x7E,0x40,0x40,0x7C,0x40,0x40,0x40,0x00,0x3C,0x66,0x40,0x40,0x4E,0x66,0x3C,0x00,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x02,0x02,0x02,0x02,0x62,0x66,0x3C,0x00,0x66,0x6C,0x78,0x70,0x6C,0x66,0x66,0x00,0x40,0x40,0x40,0x40,0x40,0x40,0x7E,0x00,0x66,0x66,0x7E,0x7E,0x76,0x66,0x66,0x00,0x66,0x66,0x76,0x7E,0x6E,0x66,0x66,0x00,0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x7C,0x66,0x66,0x7C,0x40,0x40,0x40,0x00,0x3C,0x66,0x66,0x66,0x6E,0x6C,0x3E,0x00,0x7C,0x66,0x66,0x7C,0x6C,0x66,0x66,0x00,0x3C,0x60,0x3C,0x06,0x3C,0x00,0x00,0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00,0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00,0x66,0x66,0x66,0x3C,0x66,0x66,0x66,0x00,0x66,0x3C,0x18,0x3C,0x66,0x00,0x00,0x00,0x66,0x3C,0x18,0x18,0x18,0x00,0x00,0x00,0x7E,0x02,0x04,0x08,0x7E,0x00,0x00,0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x40,0x20,0x10,0x08,0x04,0x00,0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x3C,0x66,0x00,0x00,0x00,0x00,0x00,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x30,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x3C,0x06,0x3E,0x66,0x66,0x3E,0x00,0x40,0x40,0x7C,0x66,0x66,0x66,0x7C,0x00,0x00,0x00,0x3C,0x60,0x60,0x60,0x3C,0x00,0x06,0x06,0x3E,0x66,0x66,0x66,0x3E,0x00,0x00,0x3C,0x66,0x7E,0x60,0x60,0x3C,0x00,0x1C,0x30,0x78,0x30,0x30,0x30,0x30,0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x66,0x3E,0x40,0x40,0x7C,0x66,0x66,0x66,0x66,0x00,0x00,0x38,0x18,0x18,0x18,0x18,0x3C,0x00,0x04,0x04,0x04,0x04,0x64,0x64,0x38,0x00,0x40,0x40,0x60,0x70,0x6C,0x66,0x66,0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x7C,0x66,0x76,0x66,0x66,0x00,0x00,0x00,0x7C,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00,0x00,0x7C,0x66,0x66,0x7C,0x40,0x40,0x40,0x00,0x3E,0x66,0x66,0x3E,0x06,0x06,0x06,0x00,0x00,0x7C,0x66,0x40,0x40,0x40,0x00,0x00,0x00,0x3E,0x06,0x3C,0x60,0x3E,0x00,0x38,0x18,0x18,0x78,0x18,0x18,0x18,0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x3E,0x00,0x00,0x00,0x66,0x66,0x3C,0x18,0x18,0x00,0x00,0x00,0x66,0x66,0x76,0x7E,0x66,0x00,0x00,0x00,0x66,0x3C,0x18,0x3C,0x66,0x00,0x00,0x00,0x66,0x3C,0x18,0x3C,0x18,0x00,0x00,0x00,0x7E,0x04,0x08,0x10,0x7E,0x00,0x0C,0x18,0x18,0x7E,0x18,0x18,0x0C,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x30,0x18,0x18,0x7E,0x18,0x18,0x30,0x00,0x00,0x76,0xDB,0x00,0x00,0x00,0x00,0x00 ]
        self.cpu.load_program( font, self.cpu.font_addr )
        
        start_address = 0
        if self.cpu.load_program( assembled_code, start_address ):
            self.cpu.registers[ "PC" ] = start_address
            self.status_message = f"Running from 0x{start_address:04X}..."; self.needs_display_update=True
            self.cpu_thread = threading.Thread( target=self.cpu.run, daemon=True ); self.cpu_thread.start()
        else: self.status_message = "Error loading program."; self.needs_display_update=True

    def stop_simulator( self ):
        if self.cpu_thread and self.cpu_thread.is_alive():
            self.cpu.stop_event.set()
            self.cpu_thread.join( timeout=1.0 )
        self.cpu_thread = None

    def update_gui_display( self ):
        # Scan the CPU's video memory and update the screen surface
        screen_start_addr = self.cpu.screen_address
        # Clear the surface with black only if needed, or handle per pixel
        # self.screen_surface.fill((0, 0, 0))
    
        for row in range(self.screen_height):
            for col in range(self.screen_width):
                # Calculate the memory address for the current pixel
                mem_addr = screen_start_addr + (row * self.screen_width) + col
                color_index = self.cpu.memory[mem_addr]
            
                # Draw the pixel onto the surface
                px, py = col * self.pixel_size, row * self.pixel_size
                color = self.colors[color_index]
                pygame.draw.rect(self.screen_surface, color, (px, py, self.pixel_size, self.pixel_size))

        # Now blit the updated surface to the screen
        self.screen.blit( self.screen_surface, ( 0, 0 ) )

        # --- The rest of the function (drawing the info panel) remains the same ---
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
