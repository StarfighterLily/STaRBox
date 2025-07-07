import pygame
import numpy as np
import sys
import threading
import time
import os
import re

class Assembler:
    def __init__(self):
        self.opcodes = {
            # Mnemonic: [ (Opcode, OperandSignature, Size), ... ]
            "NOP":      [(0x00, [], 1)],
            "HLT":      [(0xFF, [], 1)],
            "RET":      [(0x23, [], 1)],
            "PUSHA":    [(0x53, [], 1)],
            "POPA":     [(0x54, [], 1)],
            "INC":      [(0x01, ["r"], 2)],
            "DEC":      [(0x02, ["r"], 2)],
            "NOT":      [(0x43, ["r"], 2)],
            "PUSH":     [(0x20, ["r"], 2)],
            "POP":      [(0x21, ["r"], 2)],
            "STOREIND": [(0x15, ["r"], 2)],
            "LOADIND":  [(0x16, ["r"], 2)],
            "JMP":      [(0x06, ["v16"], 3)],
            "CALL":     [(0x22, ["v16"], 3)],
            "JZ":       [(0x30, ["v16"], 3)],
            "JNZ":      [(0x31, ["v16"], 3)],
            "JC":       [(0x32, ["v16"], 3)],
            "JNC":      [(0x33, ["v16"], 3)],
            "JN":       [(0x34, ["v16"], 3)],
            "JNN":      [(0x35, ["v16"], 3)],
            "LOAD":     [(0x0B, ["r", "v16"], 4)],
            "STORE":    [(0x07, ["v16", "r"], 4)],
            "LOADSP":   [(0x57, ["r", "idx_sp"], 3)],
            "LOADIX":   [(0x55, ["r", "idx"], 4)],
            "STOREIX":  [(0x56, ["idx", "r"], 4)],
            "MOV": [
                (0x03, ["r", "r"], 3),
                (0x08, ["r", "v8"], 3),
                (0x51, ["rp", "v16"], 4)
            ],
            "ADD": [
                (0x04, ["r", "r"], 3),
                (0x09, ["r", "v8"], 3),
                (0x52, ["rp", "rp"], 3),
                (0x58, ["rp", "v16"], 4)
            ],
            "SUB": [
                (0x05, ["r", "r"], 3),
                (0x0A, ["r", "v8"], 3),
                (0x59, ["rp", "rp"], 3),
                (0x5A, ["rp", "v16"], 4)
            ],
            "CMP": [
                (0x0E, ["r", "r"], 3),
                (0x0F, ["r", "v8"], 3),
                (0x5B, ["rp", "rp"], 3),
                (0x5C, ["rp", "v16"], 4)
            ],
            "AND": [
                (0x40, ["r", "r"], 3),
                (0x44, ["r", "v8"], 3)
            ],
            "OR": [
                (0x41, ["r", "r"], 3),
                (0x45, ["r", "v8"], 3)
            ],
            "XOR": [
                (0x42, ["r", "r"], 3),
                (0x46, ["r", "v8"], 3)
            ],
            "MUL": [
                (0x50, ["r", "r"], 3)
            ],
            "SHL": [
                (0x47, ["r", "v8"], 3)
            ],
            "SHR": [
                (0x48, ["r", "v8"], 3)
            ],
        }
        self.registers = { "R0": 0x10, "R1": 0x11, "R2": 0x12, "R3": 0x13, "R4": 0x14 }
        self.register_pairs = { "R0": 0xA0, "R2": 0xA2 }

    def assemble(self, assembly_code):
        lines = assembly_code.strip().splitlines()
        
        # --- (Helper Functions for Parsing) ---
        def parse_operand_type(operand_str, constants_dict):
            operand_str = operand_str.upper()
            if operand_str in self.registers: return "r"
            if operand_str in self.register_pairs: return "rp"
            if re.match(r"\[\s*SP\s*\+\s*[^\]]+\]", operand_str): return "idx_sp"
            if re.match(r"\[\s*(R0|R2)\s*\+\s*[^\]]+\]", operand_str): return "idx"
            
            # For type determination, we just need a rough idea.
            # We will do full expression evaluation in pass 2.
            # For now, just check if it looks like a number. If not, assume it's a label (v16).
            try:
                 # Check if it's a simple number first
                 val = int(operand_str, 0)
                 return "v8" if 0 <= val <= 0xFF else "v16"
            except ValueError:
                 # If it contains operators or is a non-numeric string, it could be an expression or label
                 if any(op in operand_str for op in ['+', '-', '*', '/']):
                     # It's an expression. We can't know its size for sure without
                     # evaluating, but v16 is a safe bet for signature matching.
                     return "v16"
                 # Could be a constant or label. Check constant dict.
                 val = constants_dict.get(operand_str)
                 if val is not None:
                     return "v8" if 0 <= val <= 0xFF else "v16"
                 # Assume it's a label, which is a 16-bit address
                 return "v16"


        def define_constants(lines_list):
            constants, remaining_lines = {}, []
            for line in lines_list:
                clean_line = line.split('#', 1)[0].strip()
                if not clean_line: continue
                parts = clean_line.split()
                if len(parts) >= 3 and parts[1].upper() == 'EQU':
                    try:
                        constants[parts[0].upper()] = int(parts[2], 0)
                    except ValueError:
                        print(f"Error: Invalid EQU value for '{parts[0]}'")
                        return None, None
                else:
                    remaining_lines.append(line)
            return constants, remaining_lines

        def split_operands(operand_string):
            if not operand_string: return []
            return [op.strip() for op in re.split(r",\s*(?![^\[]*\])", operand_string)]

        # --- (Pass 1: Label and Constant Definition) ---
        print("[Assembler] Pass 1: Defining constants and labels...")
        constants, lines = define_constants(lines)
        if constants is None: return None
        
        labels, code_intermediate, current_address = {}, [], 0
        for line_num, line in enumerate(lines, 1):
            line = line.split('#', 1)[0].strip()
            if not line: continue
            
            label_def = None
            if ':' in line:
                label, instruction_part = line.split(':', 1)
                label = label.strip().upper()
                if label in labels:
                    print(f"Error (Line {line_num}): Duplicate label '{label}'")
                    return None
                labels[label] = current_address
                label_def = label
                line = instruction_part.strip()

            if not line:
                code_intermediate.append({'line': line_num, 'address': current_address, 'label': label_def, 'op': 'NOP_LABEL', 'operands': [], 'size': 0})
                continue
            
            parts = line.split(maxsplit=1)
            op = parts[0].upper()
            operands = split_operands(parts[1]) if len(parts) > 1 else []

            if op not in self.opcodes:
                print(f"Error (Line {line_num}): Unknown instruction '{op}'")
                return None
            
            operand_sig = [parse_operand_type(op_str, constants) for op_str in operands]

            # --- CONTEXT-SENSITIVE FIX ---
            if "v16" in operand_sig:
                for i, op_str in enumerate(operands):
                    if operand_sig[i] == 'r' and op_str.upper() in self.register_pairs:
                        operand_sig[i] = 'rp'
            
            matched_variant = None
            for opcode, sig, size in self.opcodes[op]:
                if len(sig) == len(operand_sig):
                    is_match = all((s1 == s2) or (s1 == 'v16' and s2 == 'v8') for s1, s2 in zip(sig, operand_sig))
                    if is_match:
                        matched_variant = {'op': op, 'opcode': opcode, 'sig': sig, 'size': size, 'operands': operands, 'line': line_num}
                        break
            
            if not matched_variant:
                print(f"Error (Line {line_num}): Invalid operands for '{op}': {operands} (Interpreted as: {operand_sig})")
                return None
            
            code_intermediate.append({**matched_variant, 'address': current_address, 'label': label_def})
            current_address += matched_variant['size']
        print(f"[Assembler] Pass 1 Complete. Labels: {labels}")

        # --- (Pass 2: Code Generation) ---
        print("[Assembler] Pass 2: Generating bytecode...")
        code = bytearray()
        
        # --- NEW: Expression Evaluation Logic ---
        def evaluate_expression(expr_str, line_num):
            """Evaluates a simple arithmetic expression involving numbers, constants, and labels."""
            
            def resolve_term(term):
                term = term.strip().upper()
                if not term: return None
                if term in constants: return constants[term]
                if term in labels: return labels[term]
                try: return int(term, 0)
                except ValueError: return None

            # Split by operators, keeping the operators
            tokens = re.split(r'([+\-*/])', expr_str)
            tokens = [t.strip() for t in tokens if t.strip()]
            
            if not tokens:
                print(f"Error (Line {line_num}): Invalid or empty expression '{expr_str}'")
                return None
                
            # Handle unary minus/plus at the beginning
            if tokens[0] in ['+', '-'] and len(tokens) > 1:
                 tokens = ['0'] + tokens
            
            # Resolve the first term
            current_val = resolve_term(tokens[0])
            if current_val is None:
                print(f"Error (Line {line_num}): Cannot resolve term '{tokens[0]}' in expression '{expr_str}'")
                return None
            
            # Evaluate left-to-right
            i = 1
            while i < len(tokens):
                op = tokens[i]
                if i + 1 >= len(tokens):
                    print(f"Error (Line {line_num}): Incomplete expression '{expr_str}'")
                    return None
                
                val = resolve_term(tokens[i+1])
                if val is None:
                    print(f"Error (Line {line_num}): Cannot resolve term '{tokens[i+1]}' in expression '{expr_str}'")
                    return None
                
                if op == '+': current_val += val
                elif op == '-': current_val -= val
                elif op == '*': current_val *= val
                elif op == '/':
                    if val == 0:
                        print(f"Error (Line {line_num}): Division by zero in '{expr_str}'")
                        return None
                    current_val //= val  # Integer division
                
                i += 2 # Move to the next operator
                
            return current_val

        # --- MODIFIED: parse_value now uses the expression evaluator ---
        def parse_value(operand, line_num, bits=8):
            val = evaluate_expression(operand, line_num)
            if val is None: # Error already printed by evaluator
                return None
            
            max_val = (1 << bits) - 1
            # For signed context, we might need different checks, but for now, treat as unsigned.
            min_val = 0 
            if not (min_val <= val <= max_val):
                print(f"Error (Line {line_num}): Result of '{operand}' ({val}) is out of {bits}-bit range.")
                return None
            return val

        def parse_register(op, ln): return self.registers.get(op.upper())
        def parse_register_pair(op, ln): return self.register_pairs.get(op.upper())
        
        # --- MODIFIED: parse_indexed_operand now uses the expression evaluator for its offset ---
        def parse_indexed_operand(operand, line_num):
            match = re.match(r"\[\s*(R0|R2|SP)\s*\+\s*([^\]]+)\]", operand.upper())
            if not match: return None, None
            reg_name, val_str = match.groups()
            reg_code = self.register_pairs.get(reg_name) if reg_name != "SP" else 0xFF
            # The value part (val_str) is an expression to be evaluated
            offset = parse_value(val_str, line_num, 8)
            return reg_code, offset

        for instr in code_intermediate:
            if instr['op'] == 'NOP_LABEL': continue
            
            opcode, sig, operands, ln = instr['opcode'], instr['sig'], instr['operands'], instr['line']
            code.append(opcode)
            
            try:
                for i, op_sig in enumerate(sig):
                    op_str = operands[i]
                    if op_sig == "r": code.append(parse_register(op_str, ln))
                    elif op_sig == "rp": code.append(parse_register_pair(op_str, ln))
                    elif op_sig == "v8": 
                        val = parse_value(op_str, ln, 8)
                        if val is None: return None
                        code.append(val)
                    elif op_sig == "v16":
                        val = parse_value(op_str, ln, 16)
                        if val is None: return None
                        code.extend([(val >> 8) & 0xFF, val & 0xFF])
                    elif op_sig == "idx":
                        reg_pair_code, offset = parse_indexed_operand(op_str, ln)
                        if offset is None: return None
                        code.extend([reg_pair_code, offset])
                    elif op_sig == "idx_sp":
                        _, offset = parse_indexed_operand(op_str, ln)
                        if offset is None: return None
                        code.append(offset)

            except (TypeError, ValueError):
                 print(f"Assembly Error (Line {ln}): Malformed operands for '{instr['op']}'. Halted."); return None
        
        print(f"[Assembler] Pass 2 Complete. Final code size: {len(code)} bytes.")
        return code

class CPU:
    def __init__(self, memory_size, stack_size, gui=None, update_callback=None):
        # ... (Constructor remains the same from the previous version)
        self.memory_size = memory_size; self.stack_size = stack_size
        self.memory = bytearray(self.memory_size)
        self.registers = {"PC": 0, "SP": self.memory_size, "R0": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0}
        self.flags = {'Z': 0, 'C': 0, 'N': 0}
        self.screen_width = 100; self.screen_height = 100
        self.stack_base = self.memory_size
        print( f"[CPU REPORT] Stack Base Memory: 0x{self.stack_base:04x}" )
        self.stack_limit = self.memory_size - self.stack_size
        print( f"[CPU REPORT] Stack Limit Memory: 0x{self.stack_limit:04x}" )
        self.screen_address = self.stack_limit - (self.screen_width * self.screen_height)
        print( f"[CPU REPORT] Video Memory: 0x{self.screen_address:04x}" )
        self.keyboard_data_address = self.screen_address - 1
        print( f"[CPU REPORT] KBD Data Memory: 0x{self.keyboard_data_address:04x}" )
        self.keyboard_status_address = self.screen_address - 2
        print( f"[CPU REPORT] KBD Status Memory: 0x{self.keyboard_status_address:04x}" )
        self.font_addr = self.keyboard_status_address - 760
        print( f"[CPU REPORT] Font Bitmap Memory: 0x{self.font_addr:04x}" )
        self.running = False; self.stop_event = threading.Event()
        self.gui = gui; self.update_callback = update_callback
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

    def execute_instruction(self):
        if self.stop_event.is_set(): self.running = False; return
        pc = self.registers["PC"]
        if not (0 <= pc < self.memory_size):
            print(f"PC Error. Halting."); self.running = False; return

        instruction = self.memory[pc]
        pc_increment = 1
        
        def fetch(offset=1, num_bytes=1):
            if pc + offset + num_bytes > self.memory_size: self.running = False; return None
            return self.memory[pc + offset] if num_bytes == 1 else self.memory[pc + offset : pc + offset + num_bytes]

        def fetch_reg_name(offset=1):
            name = self.register_names.get(fetch(offset))
            if name is None: self.running = False
            return name

        def fetch_pair_names(offset=1):
            names = self.register_pair_names.get(fetch(offset))
            if names is None: self.running = False
            return names
        
        def fetch_addr16(offset=1):
            data = fetch(offset, 2)
            return (data[0] << 8) | data[1] if data else None

        try:
            if instruction in [0x00, 0xFF, 0x23, 0x53, 0x54]:
                if instruction == 0xFF: self.running = False; pc_increment = 0 # HLT
                elif instruction == 0x23:
                    self._check_stack_op(2,False); hi=self.memory[self.registers["SP"]]; lo=self.memory[self.registers["SP"]+1]; self.registers["SP"]+=2; self.registers["PC"]=(hi<<8)|lo; pc_increment=0
                elif instruction == 0x53:
                    self._check_stack_op(5,True); self.registers["SP"]-=5; regs=["R0","R1","R2","R3","R4"]; [self.memory.__setitem__(self.registers["SP"]+i, self.registers[r]) for i,r in enumerate(regs)]
                elif instruction == 0x54:
                    self._check_stack_op(5,False); regs=["R0","R1","R2","R3","R4"]; [self.registers.__setitem__(r, self.memory[self.registers["SP"]+i]) for i,r in enumerate(regs)]; self.registers["SP"]+=5
                pc_increment = 1 if instruction != 0x23 and instruction != 0xFF else 0
            
            elif instruction in [0x01, 0x02, 0x43, 0x20, 0x21, 0x15, 0x16]:
                reg = fetch_reg_name(); pc_increment = 2
                if   instruction == 0x01: res = (self.registers[reg] + 1) & 0xFF; self._set_flags(res); self.registers[reg] = res
                elif instruction == 0x02: res = (self.registers[reg] - 1) & 0xFF; self._set_flags(res); self.registers[reg] = res
                elif instruction == 0x43: res = (~self.registers[reg])&0xFF; self._set_flags(res); self.registers[reg] = res
                elif instruction == 0x20: self._check_stack_op(1,True); self.registers["SP"]-=1; self.memory[self.registers["SP"]]=self.registers[reg]
                elif instruction == 0x21: self._check_stack_op(1,False); val=self.memory[self.registers["SP"]]; self.registers[reg]=val; self.registers["SP"]+=1; self._set_flags(val)
                elif instruction == 0x15: addr=(self.registers["R0"]<<8)|self.registers["R1"]; self.memory[addr]=self.registers[reg]
                elif instruction == 0x16: addr=(self.registers["R0"]<<8)|self.registers["R1"]; val=self.memory[addr]; self.registers[reg]=val; self._set_flags(val)
            
            elif instruction in [0x06, 0x22, *range(0x30, 0x36)]:
                addr = fetch_addr16(); pc_increment = 3
                if instruction == 0x06: self.registers["PC"] = addr; pc_increment = 0
                elif instruction == 0x22:
                    self._check_stack_op(2,True); ret=(pc+3)&0xFFFF; self.registers["SP"]-=2; self.memory[self.registers["SP"]]=(ret>>8)&0xFF; self.memory[self.registers["SP"]+1]=ret&0xFF; self.registers["PC"]=addr; pc_increment=0
                else:
                    cond = (instruction == 0x30 and self.flags['Z']==1) or \
                           (instruction == 0x31 and self.flags['Z']==0) or \
                           (instruction == 0x32 and self.flags['C']==1) or \
                           (instruction == 0x33 and self.flags['C']==0) or \
                           (instruction == 0x34 and self.flags['N']==1) or \
                           (instruction == 0x35 and self.flags['N']==0)
                    if cond: self.registers["PC"] = addr; pc_increment = 0
            
            elif instruction in [0x03, 0x04, 0x05, 0x0E, 0x40, 0x41, 0x42, 0x50]: # R, R
                r1, r2 = fetch_reg_name(1), fetch_reg_name(2); v1, v2 = self.registers[r1], self.registers[r2]; pc_increment = 3
                if   instruction == 0x03: res = v2; self.registers[r1] = res; self._set_flags(res)
                elif instruction == 0x04: res16=v1+v2; self._set_carry(res16>0xFF); res=res16&0xFF; self.registers[r1]=res; self._set_flags(res)
                elif instruction == 0x05: self._set_carry(v1>=v2); res=(v1-v2)&0xFF; self.registers[r1]=res; self._set_flags(res)
                elif instruction == 0x0E: self._set_carry(v1>=v2); res=(v1-v2)&0xFF; self._set_flags(res)
                elif instruction == 0x40: res=v1&v2; self.registers[r1]=res; self._set_flags(res)
                elif instruction == 0x41: res=v1|v2; self.registers[r1]=res; self._set_flags(res)
                elif instruction == 0x42: res=v1^v2; self.registers[r1]=res; self._set_flags(res)
                elif instruction == 0x50: res16=v1*v2; self._set_carry(res16>0xFF); res=res16&0xFF; self.registers[r1]=res; self._set_flags(res)

            elif instruction in [0x08, 0x09, 0x0A, 0x0F, 0x44, 0x45, 0x46]: # R, Imm8
                reg, v2 = fetch_reg_name(1), fetch(2); v1 = self.registers[reg]; pc_increment = 3
                if   instruction == 0x08: res = v2; self.registers[reg] = res; self._set_flags(res)
                elif instruction == 0x09: res16=v1+v2; self._set_carry(res16>0xFF); res=res16&0xFF; self.registers[reg]=res; self._set_flags(res)
                elif instruction == 0x0A: self._set_carry(v1>=v2); res=(v1-v2)&0xFF; self.registers[reg]=res; self._set_flags(res)
                elif instruction == 0x0F: self._set_carry(v1>=v2); res=(v1-v2)&0xFF; self._set_flags(res)
                elif instruction == 0x44: res=v1&v2; self.registers[reg]=res; self._set_flags(res)
                elif instruction == 0x45: res=v1|v2; self.registers[reg]=res; self._set_flags(res)
                elif instruction == 0x46: res=v1^v2; self.registers[reg]=res; self._set_flags(res)

            elif instruction == 0x47:
                reg, shifts = fetch_reg_name(1), fetch(2)
                v1 = self.registers[reg]
                carry = 0
                for _ in range(shifts):
                    carry = 1 if (v1 & 0x80) else 0
                    v1 = (v1 << 1) & 0xFF
                self._set_carry(carry)
                self.registers[reg] = v1
                self._set_flags(v1)
                pc_increment = 3

            elif instruction == 0x48:
                reg, shifts = fetch_reg_name(1), fetch(2)
                v1 = self.registers[reg]
                carry = 0
                for _ in range(shifts):
                    carry = 1 if (v1 & 0x01) else 0
                    v1 >>= 1
                self._set_carry(carry)
                self.registers[reg] = v1
                self._set_flags(v1)
                pc_increment = 3

            elif instruction in [0x51, 0x58, 0x5A, 0x5C]: # RP, Imm16
                (r_hi, r_lo), val16 = fetch_pair_names(1), fetch_addr16(2); pc_increment = 4
                if   instruction == 0x51: self.registers[r_hi]=(val16>>8)&0xFF; self.registers[r_lo]=val16&0xFF
                else:
                    v1 = (self.registers[r_hi] << 8) | self.registers[r_lo]
                    if   instruction == 0x58: res32=v1+val16; self._set_carry(res32>0xFFFF); res=res32&0xFFFF; self._set_flags(res,True); self.registers[r_hi]=(res>>8)&0xFF; self.registers[r_lo]=res&0xFF
                    elif instruction == 0x5A: self._set_carry(v1>=val16); res=(v1-val16)&0xFFFF; self._set_flags(res,True); self.registers[r_hi]=(res>>8)&0xFF; self.registers[r_lo]=res&0xFF
                    elif instruction == 0x5C: self._set_carry(v1>=val16); res=(v1-val16)&0xFFFF; self._set_flags(res,True)
            
            elif instruction in [0x52, 0x59, 0x5B]: # RP, RP
                (d_hi, d_lo), (s_hi, s_lo) = fetch_pair_names(1), fetch_pair_names(2); pc_increment = 3
                v1 = (self.registers[d_hi]<<8)|self.registers[d_lo]; v2 = (self.registers[s_hi]<<8)|self.registers[s_lo]
                if   instruction == 0x52: res32=v1+v2; self._set_carry(res32>0xFFFF); res=res32&0xFFFF; self._set_flags(res,True); self.registers[d_hi]=(res>>8)&0xFF; self.registers[d_lo]=res&0xFF
                elif instruction == 0x59: self._set_carry(v1>=v2); res=(v1-v2)&0xFFFF; self._set_flags(res,True); self.registers[d_hi]=(res>>8)&0xFF; self.registers[d_lo]=res&0xFF
                elif instruction == 0x5B: self._set_carry(v1>=v2); res=(v1-v2)&0xFFFF; self._set_flags(res,True)
            
            elif instruction in [0x0B, 0x07, 0x55, 0x56, 0x57]:
                if   instruction == 0x0B: reg,addr=fetch_reg_name(1),fetch_addr16(2); val=self.memory[addr]; self.registers[reg]=val; self._set_flags(val); pc_increment = 4
                elif instruction == 0x07: addr,reg=fetch_addr16(1),fetch_reg_name(3); self.memory[addr] = self.registers[reg]; pc_increment = 4
                elif instruction == 0x55: r_dest,(p_hi,p_lo),offset=fetch_reg_name(1),fetch_pair_names(2),fetch(3); base=(self.registers[p_hi]<<8)|self.registers[p_lo]; val=self.memory[base+offset]; self.registers[r_dest]=val; self._set_flags(val); pc_increment=4
                elif instruction == 0x56: (p_hi,p_lo),offset,r_src=fetch_pair_names(1),fetch(2),fetch_reg_name(3); base=(self.registers[p_hi]<<8)|self.registers[p_lo]; self.memory[base+offset]=self.registers[r_src]; pc_increment=4
                elif instruction == 0x57: r_dest,offset=fetch_reg_name(1),fetch(2); val=self.memory[self.registers["SP"]+offset]; self.registers[r_dest]=val; self._set_flags(val); pc_increment=3
            
            else:
                print(f"Unknown instruction 0x{instruction:02X}. Halting."); self.running = False; pc_increment = 0

        except (TypeError, IndexError, KeyError) as e:
            print(f"CRITICAL CPU ERROR at PC=0x{pc:04X} (Op: 0x{instruction:02X}): {e}");
            import traceback; traceback.print_exc(); self.running = False; pc_increment = 0

        if self.running and pc_increment > 0:
            self.registers["PC"] = (pc + pc_increment) & 0xFFFF
    
    def run(self):
        self.running = True; self.stop_event.clear()
        start_time = time.time()
        while self.running:
            self.execute_instruction()
            if self.stop_event.is_set(): self.running = False
        print(f"CPU execution finished. Final PC: 0x{self.registers['PC']:04X}.")
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
        screen_start_addr = self.cpu.screen_address
    
        for row in range(self.screen_height):
            for col in range(self.screen_width):
                mem_addr = screen_start_addr + (row * self.screen_width) + col
                color_index = self.cpu.memory[mem_addr]
            
                px, py = col * self.pixel_size, row * self.pixel_size
                color = self.colors[color_index]
                pygame.draw.rect(self.screen_surface, color, (px, py, self.pixel_size, self.pixel_size))

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
            pc_marker = " <--" if is_pc_line else ""
            surf = self.font_small.render( line + pc_marker, True, color ); self.screen.blit( surf, ( info_x, y ) ); y += 16
            if y > self.total_height - 20: break

        pygame.display.flip()

if __name__ == "__main__":
    print( "[Main] Script starting..." )
    memory_size = 65535
    stack_size = 512
    print( "[Main] Creating CPU..." )
    cpu = CPU( memory_size, stack_size, gui=None, update_callback=None )
    print( "[Main] Creating SimulatorGUI..." )
    gui = SimulatorGUI( cpu )
    print( "[Main] Starting GUI event loop..." )
    gui.run_gui_loop()
    print( "[Main] GUI loop finished." )