import pygame
import numpy as np
import sys
import threading
import time
import os
import re
import tkinter as tk
from tkinter import filedialog
from abc import ABC, abstractmethod

# --- FONT DATA ---
font_data = [
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x18,0x18,0x00,0x18,0x00,0x00,
    0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x5A,0x24,0x7E,0x24,0x5A,0x18,0x00,
    0x44,0x4A,0x52,0x4A,0x48,0x00,0x00,0x00,0x6C,0x54,0x28,0x1A,0x36,0x00,0x00,0x00,
    0x18,0x30,0x00,0x00,0x00,0x00,0x00,0x00,0x0C,0x18,0x30,0x60,0x60,0x00,0x00,0x00,
    0x60,0x30,0x18,0x0C,0x0C,0x00,0x00,0x00,0x00,0x36,0xDB,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x3E,0x00,0x3E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x30,0x00,
    0x00,0x00,0x00,0x00,0x3E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,
    0x00,0x00,0x00,0x04,0x08,0x10,0x20,0x40,0x00,0x00,0x00,0x3C,0x42,0x42,0x42,0x42,
    0x42,0x3C,0x00,0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00,0x3C,0x42,0x02,0x04,0x08,
    0x10,0x7E,0x00,0x3C,0x42,0x02,0x1C,0x02,0x42,0x3C,0x00,0x04,0x0C,0x14,0x24,0x7E,
    0x04,0x04,0x00,0x7E,0x40,0x40,0x7C,0x02,0x02,0x3C,0x00,0x3C,0x40,0x40,0x7C,0x42,
    0x42,0x3C,0x00,0x40,0x40,0x20,0x10,0x08,0x08,0x08,0x00,0x3C,0x42,0x42,0x3C,0x42,
    0x42,0x3C,0x00,0x3C,0x42,0x42,0x3E,0x02,0x02,0x3C,0x00,0x00,0x00,0x18,0x18,0x00,
    0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x00,0x18,0x18,0x30,0x04,0x08,0x10,0x20,0x10,
    0x08,0x04,0x00,0x00,0x3E,0x00,0x3E,0x00,0x3E,0x00,0x00,0x20,0x10,0x08,0x04,0x08,
    0x10,0x20,0x00,0x3C,0x42,0x02,0x0C,0x18,0x00,0x18,0x00,0x3C,0x42,0x4A,0x4A,0x4A,
    0x40,0x3C,0x00,0x18,0x3C,0x66,0x66,0x7E,0x66,0x66,0x00,0x7C,0x66,0x66,0x7C,0x66,
    0x66,0x7C,0x00,0x3C,0x66,0x40,0x40,0x40,0x66,0x3C,0x00,0x7C,0x66,0x66,0x66,0x66,
    0x66,0x7C,0x00,0x7E,0x40,0x40,0x7C,0x40,0x40,0x7E,0x00,0x7E,0x40,0x40,0x7C,0x40,
    0x40,0x40,0x00,0x3C,0x66,0x40,0x40,0x4E,0x66,0x3C,0x00,0x66,0x66,0x66,0x7E,0x66,
    0x66,0x66,0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x02,0x02,0x02,0x02,0x62,
    0x66,0x3C,0x00,0x66,0x6C,0x78,0x70,0x6C,0x66,0x66,0x00,0x40,0x40,0x40,0x40,0x40,
    0x40,0x7E,0x00,0x66,0x66,0x7E,0x7E,0x76,0x66,0x66,0x00,0x66,0x66,0x76,0x7E,0x6E,
    0x66,0x66,0x00,0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x7C,0x66,0x66,0x7C,0x40,
    0x40,0x40,0x00,0x3C,0x66,0x66,0x66,0x6E,0x6C,0x3E,0x00,0x7C,0x66,0x66,0x7C,0x6C,
    0x66,0x66,0x00,0x3C,0x60,0x3C,0x06,0x3C,0x00,0x00,0x00,0x7E,0x18,0x18,0x18,0x18,
    0x18,0x18,0x00,0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00,0x66,0x66,0x3C,0x18,0x3C,
    0x66,0x66,0x00,0x66,0x66,0x66,0x3C,0x66,0x66,0x66,0x00,0x66,0x3C,0x18,0x3C,0x66,
    0x00,0x00,0x00,0x66,0x3C,0x18,0x18,0x18,0x00,0x00,0x00,0x7E,0x02,0x04,0x08,0x7E,
    0x00,0x00,0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x40,0x20,0x10,0x08,0x04,
    0x00,0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x3C,0x66,0x00,0x00,
    0x00,0x00,0x00,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x30,0x18,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x3C,0x06,0x3E,0x66,0x66,0x3E,0x00,0x40,0x40,0x7C,0x66,0x66,
    0x66,0x7C,0x00,0x00,0x00,0x3C,0x60,0x60,0x60,0x3C,0x00,0x06,0x06,0x3E,0x66,0x66,
    0x66,0x3E,0x00,0x00,0x3C,0x66,0x7E,0x60,0x60,0x3C,0x00,0x1C,0x30,0x78,0x30,0x30,
    0x30,0x30,0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x66,0x3E,0x40,0x40,0x7C,0x66,0x66,
    0x66,0x66,0x00,0x00,0x38,0x18,0x18,0x18,0x18,0x3C,0x00,0x04,0x04,0x04,0x04,0x64,
    0x64,0x38,0x00,0x40,0x40,0x60,0x70,0x6C,0x66,0x66,0x00,0x38,0x18,0x18,0x18,0x18,
    0x18,0x18,0x00,0x00,0x00,0x7C,0x66,0x76,0x66,0x66,0x00,0x00,0x00,0x7C,0x66,0x66,
    0x66,0x66,0x00,0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00,0x00,0x7C,0x66,0x66,0x7C,
    0x40,0x40,0x40,0x00,0x3E,0x66,0x66,0x3E,0x06,0x06,0x06,0x00,0x00,0x7C,0x66,0x40,
    0x40,0x40,0x00,0x00,0x00,0x3E,0x06,0x3C,0x60,0x3E,0x00,0x38,0x18,0x18,0x78,0x18,
    0x18,0x18,0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x3E,0x00,0x00,0x00,0x66,0x66,0x3C,
    0x18,0x18,0x00,0x00,0x00,0x66,0x66,0x76,0x7E,0x66,0x00,0x00,0x00,0x66,0x3C,0x18,
    0x3C,0x66,0x00,0x00,0x00,0x66,0x3C,0x18,0x3C,0x18,0x00,0x00,0x00,0x7E,0x04,0x08,
    0x10,0x7E,0x00,0x0C,0x18,0x18,0x7E,0x18,0x18,0x0C,0x00,0x18,0x18,0x18,0x18,0x18,
    0x18,0x18,0x00,0x30,0x18,0x18,0x7E,0x18,0x18,0x30,0x00,0x00,0x76,0xDB,0x00,0x00,
    0x00,0x00,0x00]

# --- INSTRUCTION SET DEFINITION ---
class AssemblyError(Exception):
    pass

class Instruction(ABC):
    """Abstract Base Class for all instructions."""
    mnemonic = "INVALID"
    opcode = 0xFF
    size = 0
    
    @abstractmethod
    def assemble(self, asm, operands, all_symbols):
        """Try to assemble the given operands. Return bytecode list or raise AssemblyError."""
        pass

    @abstractmethod
    def execute(self, cpu):
        """Execute the instruction. Return the number of bytes to increment PC."""
        pass

# --- Helper Methods for Instruction Classes ---
def _fetch_reg_name(cpu, offset=1):
    name = cpu.register_names.get(cpu.memory[cpu.registers["PC"] + offset])
    if name is None: raise ValueError("Invalid register code.")
    return name

def _fetch_pair_names(cpu, offset=1):
    names = cpu.register_pair_names.get(cpu.memory[cpu.registers["PC"] + offset])
    if names is None: raise ValueError("Invalid register pair code.")
    return names

def _fetch_addr16(cpu, offset=1):
    pc = cpu.registers["PC"]
    return (cpu.memory[pc + offset] << 8) | cpu.memory[pc + offset + 1]

def _fetch_val8(cpu, offset=1):
    return cpu.memory[cpu.registers["PC"] + offset]


# --- Concrete Instruction Classes ---

class InvalidInstruction(Instruction):
    """A concrete class for unassigned opcodes."""
    def assemble(self, asm, operands, all_symbols):
        raise AssemblyError("Cannot assemble an invalid instruction.")
    def execute(self, cpu):
        pc = cpu.registers["PC"]
        opcode = cpu.memory[pc]
        print(f"Unknown instruction 0x{opcode:02X} at PC=0x{pc:04X}. Halting.")
        cpu.halted = True
        return 1

# --- Group: No Operand Instructions ---

class Nop(Instruction):
    mnemonic, opcode, size = "NOP", 0x00, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError("NOP takes no operands")
        return [self.opcode]
    def execute(self, cpu): return self.size

class Hlt(Instruction):
    mnemonic, opcode, size = "HLT", 0xFF, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError("HLT takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu.halted = True
        return self.size

class Ret(Instruction):
    mnemonic, opcode, size = "RET", 0x23, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu._check_stack_op(2, False)
        hi = cpu.memory[cpu.registers["SP"]]
        lo = cpu.memory[cpu.registers["SP"] + 1]
        cpu.registers["SP"] += 2
        cpu.registers["PC"] = (hi << 8) | lo
        return 0

class PushA(Instruction):
    mnemonic, opcode, size = "PUSHA", 0x53, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu._check_stack_op(10, True)
        cpu.registers["SP"] -= 10
        regs = [f"R{i}" for i in range(10)]
        for i, r in enumerate(regs):
            cpu.memory[cpu.registers["SP"] + i] = cpu.registers[r]
        return self.size

class PopA(Instruction):
    mnemonic, opcode, size = "POPA", 0x54, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu._check_stack_op(10, False)
        regs = [f"R{i}" for i in range(10)]
        for i, r in enumerate(regs):
            cpu.registers[r] = cpu.memory[cpu.registers["SP"] + i]
        cpu.registers["SP"] += 10
        return self.size

class PushF(Instruction):
    mnemonic, opcode, size = "PUSHF", 0x60, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu._check_stack_op(1, True)
        cpu.registers["SP"] -= 1
        cpu.memory[cpu.registers["SP"]] = cpu._pack_flags()
        return self.size

class PopF(Instruction):
    mnemonic, opcode, size = "POPF", 0x61, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu._check_stack_op(1, False)
        cpu._unpack_flags(cpu.memory[cpu.registers["SP"]])
        cpu.registers["SP"] += 1
        return self.size

class Sti(Instruction):
    mnemonic, opcode, size = "STI", 0x70, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu.flags['I'] = 1
        return self.size

class Cli(Instruction):
    mnemonic, opcode, size = "CLI", 0x71, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu.flags['I'] = 0
        return self.size

class Iret(Instruction):
    mnemonic, opcode, size = "IRET", 0x72, 1
    def assemble(self, asm, operands, all_symbols):
        if operands: raise AssemblyError(f"{self.mnemonic} takes no operands")
        return [self.opcode]
    def execute(self, cpu):
        cpu._check_stack_op(3, False)
        flags_val = cpu.memory[cpu.registers["SP"]]
        pc_hi = cpu.memory[cpu.registers["SP"] + 1]
        pc_lo = cpu.memory[cpu.registers["SP"] + 2]
        cpu._unpack_flags(flags_val)
        cpu.registers["PC"] = (pc_hi << 8) | pc_lo
        cpu.registers["SP"] += 3
        return 0

# --- Group: Single Operand Instructions ---

class IncReg(Instruction):
    mnemonic, opcode, size = "INC", 0x01, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("INC requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        v1 = cpu.registers[reg]
        res = (v1 + 1) & 0xFF
        cpu._set_flags(res, v1=v1, v2=1)
        cpu.registers[reg] = res
        return self.size

class DecReg(Instruction):
    mnemonic, opcode, size = "DEC", 0x02, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("DEC requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        v1 = cpu.registers[reg]
        res = (v1 - 1) & 0xFF
        cpu._set_flags(res, v1=v1, v2=1, is_sub=True)
        cpu.registers[reg] = res
        return self.size

class NotReg(Instruction):
    mnemonic, opcode, size = "NOT", 0x43, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("NOT requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        res = (~cpu.registers[reg]) & 0xFF
        cpu._set_flags(res)
        cpu.registers[reg] = res
        return self.size

class PushReg(Instruction):
    mnemonic, opcode, size = "PUSH", 0x20, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("PUSH requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        cpu._check_stack_op(1, True)
        cpu.registers["SP"] -= 1
        cpu.memory[cpu.registers["SP"]] = cpu.registers[reg]
        return self.size

class PopReg(Instruction):
    mnemonic, opcode, size = "POP", 0x21, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("POP requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        cpu._check_stack_op(1, False)
        val = cpu.memory[cpu.registers["SP"]]
        cpu.registers[reg] = val
        cpu.registers["SP"] += 1
        cpu._set_flags(val)
        return self.size

class StoreInd(Instruction):
    mnemonic, opcode, size = "STOREIND", 0x15, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("STOREIND requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        addr = (cpu.registers["R0"] << 8) | cpu.registers["R1"]
        cpu.memory[addr] = cpu.registers[reg]
        return self.size

class LoadInd(Instruction):
    mnemonic, opcode, size = "LOADIND", 0x16, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register(operands[0]): raise AssemblyError("LOADIND requires one register operand")
        return [self.opcode, asm.registers[operands[0].upper()]]
    def execute(self, cpu):
        reg = _fetch_reg_name(cpu)
        addr = (cpu.registers["R0"] << 8) | cpu.registers["R1"]
        val = cpu.memory[addr]
        cpu.registers[reg] = val
        cpu._set_flags(val)
        return self.size

class PushRp(Instruction):
    mnemonic, opcode, size = "PUSHRP", 0x62, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register_pair(operands[0]): raise AssemblyError("PUSHRP requires one register pair operand")
        return [self.opcode, asm.register_pairs[operands[0].upper()]]
    def execute(self, cpu):
        (r_hi, r_lo) = _fetch_pair_names(cpu)
        cpu._check_stack_op(2, True)
        cpu.registers["SP"] -= 2
        cpu.memory[cpu.registers["SP"]] = cpu.registers[r_hi]
        cpu.memory[cpu.registers["SP"] + 1] = cpu.registers[r_lo]
        return self.size

class PopRp(Instruction):
    mnemonic, opcode, size = "POPRP", 0x63, 2
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1 or not asm.is_register_pair(operands[0]): raise AssemblyError("POPRP requires one register pair operand")
        return [self.opcode, asm.register_pairs[operands[0].upper()]]
    def execute(self, cpu):
        (r_hi, r_lo) = _fetch_pair_names(cpu)
        cpu._check_stack_op(2, False)
        cpu.registers[r_hi] = cpu.memory[cpu.registers["SP"]]
        cpu.registers[r_lo] = cpu.memory[cpu.registers["SP"] + 1]
        cpu.registers["SP"] += 2
        return self.size

# --- Group: Jump/Call Instructions ---
class BaseJump(Instruction): # Helper base class for jumps
    size = 3
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 1: raise AssemblyError(f"{self.mnemonic} requires one address operand")
        addr = asm.parse_value(operands[0], 16, all_symbols)
        return [self.opcode, (addr >> 8) & 0xFF, addr & 0xFF]

class Jmp(BaseJump):
    mnemonic, opcode = "JMP", 0x06
    def execute(self, cpu):
        addr = _fetch_addr16(cpu)
        cpu.registers["PC"] = addr
        return 0

class Call(BaseJump):
    mnemonic, opcode = "CALL", 0x22
    def execute(self, cpu):
        addr = _fetch_addr16(cpu)
        cpu._check_stack_op(2, True)
        ret_addr = (cpu.registers["PC"] + self.size) & 0xFFFF
        cpu.registers["SP"] -= 2
        cpu.memory[cpu.registers["SP"]] = (ret_addr >> 8) & 0xFF
        cpu.memory[cpu.registers["SP"] + 1] = ret_addr & 0xFF
        cpu.registers["PC"] = addr
        return 0

class Jz(BaseJump):
    mnemonic, opcode = "JZ", 0x30
    def execute(self, cpu):
        if cpu.flags['Z'] == 1:
            cpu.registers["PC"] = _fetch_addr16(cpu)
            return 0
        return self.size

class Jnz(BaseJump):
    mnemonic, opcode = "JNZ", 0x31
    def execute(self, cpu):
        if cpu.flags['Z'] == 0:
            cpu.registers["PC"] = _fetch_addr16(cpu)
            return 0
        return self.size

class Jc(BaseJump):
    mnemonic, opcode = "JC", 0x32
    def execute(self, cpu):
        if cpu.flags['C'] == 1:
            cpu.registers["PC"] = _fetch_addr16(cpu)
            return 0
        return self.size

class Jnc(BaseJump):
    mnemonic, opcode = "JNC", 0x33
    def execute(self, cpu):
        if cpu.flags['C'] == 0:
            cpu.registers["PC"] = _fetch_addr16(cpu)
            return 0
        return self.size

class Jn(BaseJump):
    mnemonic, opcode = "JN", 0x34
    def execute(self, cpu):
        if cpu.flags['N'] == 1:
            cpu.registers["PC"] = _fetch_addr16(cpu)
            return 0
        return self.size

class Jnn(BaseJump):
    mnemonic, opcode = "JNN", 0x35
    def execute(self, cpu):
        if cpu.flags['N'] == 0:
            cpu.registers["PC"] = _fetch_addr16(cpu)
            return 0
        return self.size

# --- Group: Two Operand Instructions ---

class MovRegReg(Instruction):
    mnemonic, opcode, size = "MOV", 0x03, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for MOV r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1_name, r2_name = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        val = cpu.registers[r2_name]
        cpu.registers[r1_name] = val
        cpu._set_flags(val)
        return self.size

class MovRegVal8(Instruction):
    mnemonic, opcode, size = "MOV", 0x08, 3
    def assemble(self, asm, operands, all_symbols):
        val_type = asm.parse_operand_type(operands[1], all_symbols)
        if not (asm.is_register(operands[0]) and val_type == 'v8'): raise AssemblyError("Not a match for MOV r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg_name, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        cpu.registers[reg_name] = val
        cpu._set_flags(val)
        return self.size

class AddRegReg(Instruction):
    mnemonic, opcode, size = "ADD", 0x04, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for ADD r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        v1, v2 = cpu.registers[r1], cpu.registers[r2]
        res16 = v1 + v2
        cpu._set_carry(res16 > 0xFF); res = res16 & 0xFF
        cpu.registers[r1] = res; cpu._set_flags(res, v1=v1, v2=v2)
        return self.size

class AddRegVal8(Instruction):
    mnemonic, opcode, size = "ADD", 0x09, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for ADD r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        v1 = cpu.registers[reg]
        res16 = v1 + v2
        cpu._set_carry(res16 > 0xFF); res = res16 & 0xFF
        cpu.registers[reg] = res; cpu._set_flags(res, v1=v1, v2=v2)
        return self.size

class SubRegReg(Instruction):
    mnemonic, opcode, size = "SUB", 0x05, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for SUB r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        v1, v2 = cpu.registers[r1], cpu.registers[r2]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
        cpu.registers[r1] = res; cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
        return self.size

class SubRegVal8(Instruction):
    mnemonic, opcode, size = "SUB", 0x0A, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for SUB r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        v1 = cpu.registers[reg]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
        cpu.registers[reg] = res; cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
        return self.size

class CmpRegReg(Instruction):
    mnemonic, opcode, size = "CMP", 0x0E, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for CMP r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        v1, v2 = cpu.registers[r1], cpu.registers[r2]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
        cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
        return self.size

class CmpRegVal8(Instruction):
    mnemonic, opcode, size = "CMP", 0x0F, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for CMP r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        v1 = cpu.registers[reg]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
        cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
        return self.size

class AndRegReg(Instruction):
    mnemonic, opcode, size = "AND", 0x40, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for AND r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        res = cpu.registers[r1] & cpu.registers[r2]
        cpu.registers[r1] = res; cpu._set_flags(res); cpu.flags['V'] = 0
        return self.size

class AndRegVal8(Instruction):
    mnemonic, opcode, size = "AND", 0x44, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for AND r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        res = cpu.registers[reg] & val
        cpu.registers[reg] = res; cpu._set_flags(res); cpu.flags['V'] = 0
        return self.size

class OrRegReg(Instruction):
    mnemonic, opcode, size = "OR", 0x41, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for OR r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        res = cpu.registers[r1] | cpu.registers[r2]
        cpu.registers[r1] = res; cpu._set_flags(res); cpu.flags['V'] = 0
        return self.size

class OrRegVal8(Instruction):
    mnemonic, opcode, size = "OR", 0x45, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for OR r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        res = cpu.registers[reg] | val
        cpu.registers[reg] = res; cpu._set_flags(res); cpu.flags['V'] = 0
        return self.size

class XorRegReg(Instruction):
    mnemonic, opcode, size = "XOR", 0x42, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for XOR r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        res = cpu.registers[r1] ^ cpu.registers[r2]
        cpu.registers[r1] = res; cpu._set_flags(res); cpu.flags['V'] = 0
        return self.size

class XorRegVal8(Instruction):
    mnemonic, opcode, size = "XOR", 0x46, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for XOR r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        res = cpu.registers[reg] ^ val
        cpu.registers[reg] = res; cpu._set_flags(res); cpu.flags['V'] = 0
        return self.size

class MulRegReg(Instruction):
    mnemonic, opcode, size = "MUL", 0x50, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for MUL r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        v1, v2 = cpu.registers[r1], cpu.registers[r2]
        res16 = v1 * v2
        cpu._set_carry(res16 > 0xFF); res = res16 & 0xFF
        cpu.registers[r1] = res; cpu._set_flags(res)
        return self.size

class DivRegReg(Instruction):
    mnemonic, opcode, size = "DIV", 0x64, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for DIV r, r")
        return [self.opcode, asm.registers[operands[0].upper()], asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
        v1, v2 = cpu.registers[r1], cpu.registers[r2]
        if v2 == 0:
            cpu.running = False
            return 0
        res = v1 // v2; rem = v1 % v2
        cpu.registers[r1] = res; cpu.registers[r2] = rem; cpu._set_flags(res)
        return self.size

# --- Group: Shift/Rotate Instructions ---

class ShlRegVal8(Instruction):
    mnemonic, opcode, size = "SHL", 0x47, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for SHL r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        carry = 0; temp_v1 = cpu.registers[reg]
        for _ in range(v2):
            carry = 1 if (temp_v1 & 0x80) else 0
            temp_v1 = (temp_v1 << 1) & 0xFF
        cpu._set_carry(carry); cpu.registers[reg] = temp_v1; cpu._set_flags(temp_v1)
        return self.size

class ShrRegVal8(Instruction):
    mnemonic, opcode, size = "SHR", 0x48, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for SHR r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        carry = 0; temp_v1 = cpu.registers[reg]
        for _ in range(v2):
            carry = 1 if (temp_v1 & 0x01) else 0
            temp_v1 >>= 1
        cpu._set_carry(carry); cpu.registers[reg] = temp_v1; cpu._set_flags(temp_v1)
        return self.size

class RolRegVal8(Instruction):
    mnemonic, opcode, size = "ROL", 0x65, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for ROL r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        v1 = cpu.registers[reg]
        shifts = v2 % 8
        res = ((v1 << shifts) | (v1 >> (8 - shifts))) & 0xFF
        cpu._set_carry(1 if (res & 0x01) else 0); cpu.registers[reg] = res; cpu._set_flags(res)
        return self.size

class RorRegVal8(Instruction):
    mnemonic, opcode, size = "ROR", 0x66, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) == 'v8'): raise AssemblyError("Not a match for ROR r, v8")
        return [self.opcode, asm.registers[operands[0].upper()], asm.parse_value(operands[1], 8, all_symbols)]
    def execute(self, cpu):
        reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        v1 = cpu.registers[reg]
        shifts = v2 % 8
        res = ((v1 >> shifts) | (v1 << (8 - shifts))) & 0xFF
        cpu._set_carry(1 if (res & 0x80) else 0); cpu.registers[reg] = res; cpu._set_flags(res)
        return self.size

# --- Group: Memory & 16-bit Value Instructions ---

class Load(Instruction):
    mnemonic, opcode, size = "LOAD", 0x0B, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) in ['v8', 'v16']): raise AssemblyError("Not a match for LOAD r, v16")
        addr = asm.parse_value(operands[1], 16, all_symbols)
        return [self.opcode, asm.registers[operands[0].upper()], (addr >> 8) & 0xFF, addr & 0xFF]
    def execute(self, cpu):
        reg, addr = _fetch_reg_name(cpu, 1), _fetch_addr16(cpu, 2)
        val = cpu.memory[addr]
        cpu.registers[reg] = val; cpu._set_flags(val)
        return self.size

class Store(Instruction):
    mnemonic, opcode, size = "STORE", 0x07, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.parse_operand_type(operands[0], all_symbols) in ['v8', 'v16'] and asm.is_register(operands[1])): raise AssemblyError("Not a match for STORE v16, r")
        addr = asm.parse_value(operands[0], 16, all_symbols)
        return [self.opcode, (addr >> 8) & 0xFF, addr & 0xFF, asm.registers[operands[1].upper()]]
    def execute(self, cpu):
        addr, reg = _fetch_addr16(cpu, 1), _fetch_reg_name(cpu, 3)
        cpu.memory[addr] = cpu.registers[reg]
        return self.size

class MovRpVal16(Instruction):
    mnemonic, opcode, size = "MOV", 0x51, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) in ['v8', 'v16']): raise AssemblyError("Not a match for MOV rp, v16")
        val16 = asm.parse_value(operands[1], 16, all_symbols)
        return [self.opcode, asm.register_pairs[operands[0].upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
    def execute(self, cpu):
        (r_hi, r_lo), val16 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
        cpu.registers[r_hi] = (val16 >> 8) & 0xFF
        cpu.registers[r_lo] = val16 & 0xFF
        return self.size

class AddRpRp(Instruction):
    mnemonic, opcode, size = "ADD", 0x52, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.is_register_pair(operands[1])): raise AssemblyError("Not a match for ADD rp, rp")
        return [self.opcode, asm.register_pairs[operands[0].upper()], asm.register_pairs[operands[1].upper()]]
    def execute(self, cpu):
        (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
        v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
        v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
        res32 = v1 + v2; cpu._set_carry(res32 > 0xFFFF); res = res32 & 0xFFFF
        cpu._set_flags(res, True, v1, v2)
        cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
        return self.size

class AddRpVal16(Instruction):
    mnemonic, opcode, size = "ADD", 0x58, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) in ['v8', 'v16']): raise AssemblyError("Not a match for ADD rp, v16")
        val16 = asm.parse_value(operands[1], 16, all_symbols)
        return [self.opcode, asm.register_pairs[operands[0].upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
    def execute(self, cpu):
        (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
        v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
        res32 = v1 + v2; cpu._set_carry(res32 > 0xFFFF); res = res32 & 0xFFFF
        cpu._set_flags(res, True, v1, v2)
        cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
        return self.size

class SubRpRp(Instruction):
    mnemonic, opcode, size = "SUB", 0x59, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.is_register_pair(operands[1])): raise AssemblyError("Not a match for SUB rp, rp")
        return [self.opcode, asm.register_pairs[operands[0].upper()], asm.register_pairs[operands[1].upper()]]
    def execute(self, cpu):
        (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
        v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
        v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
        cpu._set_flags(res, True, v1, v2, is_sub=True)
        cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
        return self.size

class SubRpVal16(Instruction):
    mnemonic, opcode, size = "SUB", 0x5A, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) in ['v8', 'v16']): raise AssemblyError("Not a match for SUB rp, v16")
        val16 = asm.parse_value(operands[1], 16, all_symbols)
        return [self.opcode, asm.register_pairs[operands[0].upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
    def execute(self, cpu):
        (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
        v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
        cpu._set_flags(res, True, v1, v2, is_sub=True)
        cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
        return self.size

class CmpRpRp(Instruction):
    mnemonic, opcode, size = "CMP", 0x5B, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.is_register_pair(operands[1])): raise AssemblyError("Not a match for CMP rp, rp")
        return [self.opcode, asm.register_pairs[operands[0].upper()], asm.register_pairs[operands[1].upper()]]
    def execute(self, cpu):
        (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
        v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
        v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
        cpu._set_flags(res, True, v1, v2, is_sub=True)
        return self.size

class CmpRpVal16(Instruction):
    mnemonic, opcode, size = "CMP", 0x5C, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register_pair(operands[0]) and asm.parse_operand_type(operands[1], all_symbols) in ['v8', 'v16']): raise AssemblyError("Not a match for CMP rp, v16")
        val16 = asm.parse_value(operands[1], 16, all_symbols)
        return [self.opcode, asm.register_pairs[operands[0].upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
    def execute(self, cpu):
        (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
        v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
        cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
        cpu._set_flags(res, True, v1, v2, is_sub=True)
        return self.size

# --- Group: Indexed Addressing Instructions ---

class LoadIx(Instruction):
    mnemonic, opcode, size = "LOADIX", 0x55, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_indexed_pair(operands[1])): raise AssemblyError("Not a match for LOADIX r, [rp+v8]")
        reg, (reg_pair_code, offset) = operands[0], asm.parse_indexed_operand(operands[1], all_symbols)
        return [self.opcode, asm.registers[reg.upper()], reg_pair_code, offset]
    def execute(self, cpu):
        r_dest, (p_hi, p_lo), offset = _fetch_reg_name(cpu, 1), _fetch_pair_names(cpu, 2), _fetch_val8(cpu, 3)
        base = (cpu.registers[p_hi] << 8) | cpu.registers[p_lo]
        val = cpu.memory[base + offset]
        cpu.registers[r_dest] = val; cpu._set_flags(val)
        return self.size

class StoreIx(Instruction):
    mnemonic, opcode, size = "STOREIX", 0x56, 4
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_indexed_pair(operands[0]) and asm.is_register(operands[1])): raise AssemblyError("Not a match for STOREIX [rp+v8], r")
        (reg_pair_code, offset), reg = asm.parse_indexed_operand(operands[0], all_symbols), operands[1]
        return [self.opcode, reg_pair_code, offset, asm.registers[reg.upper()]]
    def execute(self, cpu):
        (p_hi, p_lo), offset, r_src = _fetch_pair_names(cpu, 1), _fetch_val8(cpu, 2), _fetch_reg_name(cpu, 3)
        base = (cpu.registers[p_hi] << 8) | cpu.registers[p_lo]
        cpu.memory[base + offset] = cpu.registers[r_src]
        return self.size

class LoadSp(Instruction):
    mnemonic, opcode, size = "LOADSP", 0x57, 3
    def assemble(self, asm, operands, all_symbols):
        if not (asm.is_register(operands[0]) and asm.is_indexed_sp(operands[1])): raise AssemblyError("Not a match for LOADSP r, [sp+v8]")
        reg, (_, offset) = operands[0], asm.parse_indexed_operand(operands[1], all_symbols)
        return [self.opcode, asm.registers[reg.upper()], offset]
    def execute(self, cpu):
        r_dest, offset = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
        val = cpu.memory[cpu.registers["SP"] + offset]
        cpu.registers[r_dest] = val; cpu._set_flags(val)
        return self.size


class Assembler:
    def __init__(self):
        self.registers = {f"R{i}": 0x10 + i for i in range(10)}
        self.register_pairs = {"R0": 0xA0, "R2": 0xA2, "R4": 0xA4, "R6": 0xA6, "R8": 0xA8}
        self.current_pass = 0 # Bug Fix: To track assembly pass

        self.instructions = {
            "NOP": [Nop()], "HLT": [Hlt()], "RET": [Ret()], "PUSHA": [PushA()], "POPA": [PopA()],
            "PUSHF": [PushF()], "POPF": [PopF()], "STI": [Sti()], "CLI": [Cli()], "IRET": [Iret()],
            "INC": [IncReg()], "DEC": [DecReg()], "NOT": [NotReg()], "PUSH": [PushReg()], "POP": [PopReg()],
            "STOREIND": [StoreInd()], "LOADIND": [LoadInd()], "PUSHRP": [PushRp()], "POPRP": [PopRp()],
            "JMP": [Jmp()], "CALL": [Call()], "JZ": [Jz()], "JNZ": [Jnz()], "JC": [Jc()], "JNC": [Jnc()], "JN": [Jn()], "JNN": [Jnn()],
            "LOAD": [Load()], "STORE": [Store()], "LOADIX": [LoadIx()], "STOREIX": [StoreIx()], "LOADSP": [LoadSp()],
            "MOV": [MovRegReg(), MovRegVal8(), MovRpVal16()],
            "ADD": [AddRegReg(), AddRegVal8(), AddRpRp(), AddRpVal16()],
            "SUB": [SubRegReg(), SubRegVal8(), SubRpRp(), SubRpVal16()],
            "CMP": [CmpRegReg(), CmpRegVal8(), CmpRpRp(), CmpRpVal16()],
            "AND": [AndRegReg(), AndRegVal8()], "OR": [OrRegReg(), OrRegVal8()], "XOR": [XorRegReg(), XorRegVal8()],
            "MUL": [MulRegReg()], "DIV": [DivRegReg()],
            "SHL": [ShlRegVal8()], "SHR": [ShrRegVal8()], "ROL": [RolRegVal8()], "ROR": [RorRegVal8()],
        }

    def is_register(self, op_str): return op_str.upper() in self.registers
    def is_register_pair(self, op_str): return op_str.upper() in self.register_pairs
    def is_indexed_sp(self, op_str): return bool(re.match(r"\[\s*SP\s*\+\s*[^\]]+\]", op_str.upper()))
    def is_indexed_pair(self, op_str): return bool(re.match(r"\[\s*(R0|R2|R4|R6|R8)\s*\+\s*[^\]]+\]", op_str.upper()))
    
    def parse_indexed_operand(self, operand, all_symbols):
        match = re.match(r"\[\s*(R0|R2|R4|R6|R8|SP)\s*\+\s*([^\]]+)\]", operand.upper())
        if not match: return None, None
        reg_name, val_str = match.groups()
        reg_code = self.register_pairs.get(reg_name)
        offset = self.parse_value(val_str, 8, all_symbols)
        return reg_code, offset
        
    def _evaluate_constant_expression(self, expr_str, constants):
        work_expr = expr_str.strip().upper()
        for const_name in sorted(constants.keys(), key=len, reverse=True):
            work_expr = re.sub(r'\b' + re.escape(const_name) + r'\b', str(constants[const_name]), work_expr)
        temp_expr = re.sub(r'0X[0-9A-F]+', '', work_expr)
        temp_expr = re.sub(r'0B[01]+', '', temp_expr)
        temp_expr = re.sub(r'[0-9]+', '', temp_expr)
        temp_expr = re.sub(r'[+\-*/&|<>^~()]', '', temp_expr)
        temp_expr = temp_expr.strip()
        if temp_expr: return None
        try:
            safe_dict = {'__builtins__': {}}
            return eval(work_expr, safe_dict, {})
        except Exception:
            return None

    def parse_operand_type(self, operand_str, constants_dict):
        op_upper = operand_str.upper()
        if self.is_register(op_upper): return "r"
        if self.is_register_pair(op_upper): return "rp"
        if self.is_indexed_sp(op_upper): return "idx_sp"
        if self.is_indexed_pair(op_upper): return "idx"
        val = self._evaluate_constant_expression(operand_str, constants_dict)
        if val is not None:
            return "v8" if 0 <= val <= 0xFF else "v16"
        return "v16"
    
    def parse_value(self, operand, bits, all_symbols):
        val = self._evaluate_constant_expression(operand, all_symbols)
        if val is None:
            # Bug Fix: Handle forward references in Pass 1
            op_str = operand.strip().upper()
            if self.current_pass == 1 and re.match(r'^[A-Z_][A-Z0-9_]*$', op_str):
                 return 0 # It's a forward reference, return dummy value for size calculation.
            raise AssemblyError(f"Cannot resolve expression '{operand}'")

        max_val = (1 << bits) - 1
        if not (0 <= val <= max_val):
            if bits == 8 and -128 <= val < 0: val &= 0xFF
            elif bits == 16 and -32768 <= val < 0: val &= 0xFFFF
            else: print(f"Warning: Value of '{operand}' ({val}) is out of {bits}-bit range.")
        return val

    def assemble(self, assembly_code):
        lines = assembly_code.strip().splitlines()

        def define_constants(lines_list):
            constants, remaining_lines = {}, []
            for line in lines_list:
                clean_line = line.split(';', 1)[0].strip()
                if not clean_line: continue
                parts = clean_line.split()
                if len(parts) >= 3 and parts[1].upper() == 'EQU':
                    val = self._evaluate_constant_expression(parts[2], constants)
                    if val is None:
                        print(f"Error: Could not evaluate EQU value for '{parts[0]}'")
                        return None, None
                    constants[parts[0].upper()] = val
                else:
                    remaining_lines.append(line)
            return constants, remaining_lines
            
        def split_operands(operand_string):
            if not operand_string: return []
            return [op.strip() for op in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", operand_string)]

        # --- Pass 1 ---
        print("[Assembler] Pass 1: Defining constants, labels, and data...")
        self.current_pass = 1 # Bug Fix: Set current pass to 1
        constants, lines = define_constants(lines)
        if constants is None: return None
        
        labels, code_intermediate, current_address = {}, [], 0
        for line_num, line in enumerate(lines, 1):
            line = line.split(';', 1)[0].strip()
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
            operands_str = parts[1] if len(parts) > 1 else ""
            operands = split_operands(operands_str)
            
            if op == "ORG":
                new_addr = self.parse_value(operands[0], 16, constants)
                if not (0 <= new_addr < 65535):
                    print(f"Error (Line {line_num}): ORG address 0x{new_addr:04X} is out of bounds.")
                    return None
                current_address = new_addr
                continue

            if op in ["DB", "DW"]:
                data_size = 0
                if op == "DB":
                    for op_str in operands: data_size += len(op_str[1:-1].encode('ascii')) if op_str.startswith('"') else 1
                elif op == "DW": data_size += len(operands) * 2
                code_intermediate.append({'line': line_num, 'address': current_address, 'op': op, 'operands': operands, 'size': data_size})
                current_address += data_size
                continue

            instruction_variants = self.instructions.get(op)
            if not instruction_variants:
                print(f"Error (Line {line_num}): Unknown instruction '{op}'")
                return None

            matched_instruction = None
            all_symbols_pass1 = {**constants, **labels}
            for instr_obj in instruction_variants:
                try:
                    instr_obj.assemble(self, operands, all_symbols_pass1)
                    matched_instruction = instr_obj
                    break 
                except (AssemblyError, KeyError, IndexError, TypeError):
                    continue 

            if not matched_instruction:
                print(f"Error (Line {line_num}): Invalid operands for '{op}': {operands}")
                return None
            
            code_intermediate.append({
                'line': line_num, 'address': current_address, 'label': label_def, 
                'op': op, 'operands': operands, 'size': matched_instruction.size,
                'instruction_obj': matched_instruction
            })
            current_address += matched_instruction.size
        print(f"[Assembler] Pass 1 Complete. Labels: {labels}")

        # --- Pass 2 ---
        print("[Assembler] Pass 2: Generating bytecode...")
        self.current_pass = 2 # Bug Fix: Set current pass to 2
        final_code = bytearray()
        all_symbols = {**constants, **labels}

        for item in code_intermediate:
            if item['op'] == 'NOP_LABEL': continue
            line_num = item['line']

            if item['op'] in ["DB", "DW"]:
                for op_str in item['operands']:
                    if item['op'] == "DB":
                        if op_str.startswith('"') and op_str.endswith('"'):
                            final_code.extend(op_str[1:-1].encode('ascii'))
                        else: final_code.append(self.parse_value(op_str, 8, all_symbols))
                    elif item['op'] == "DW":
                        val = self.parse_value(op_str, 16, all_symbols)
                        final_code.extend([(val >> 8) & 0xFF, val & 0xFF])
                continue
            
            try:
                instr_obj = item['instruction_obj']
                bytecode = instr_obj.assemble(self, item['operands'], all_symbols)
                final_code.extend(bytecode)
            except (AssemblyError, KeyError, IndexError, TypeError) as e:
                print(f"Assembly Error (Line {line_num}): Could not assemble '{item['op']}'. Details: {e}")
                import traceback; traceback.print_exc()
                return None
        
        print(f"[Assembler] Pass 2 Complete. Final code size: {len(final_code)} bytes.")
        return final_code


# --- CPU CLASS (Refactored) ---
class CPU:
    def __init__(self, memory_size, stack_size, gui=None, update_callback=None):
        self.memory_size = memory_size; self.stack_size = stack_size
        self.memory = bytearray(self.memory_size)
        
        self.registers = {"PC": 0, "SP": self.memory_size}
        for i in range(10): self.registers[f"R{i}"] = 0
            
        self.flags = {'Z': 0, 'C': 0, 'N': 0, 'V': 0, 'A': 0, 'I': 0}
        self.halted = False
        
        self.screen_width = 100; self.screen_height = 100;
        self.screen_size = self.screen_width * self.screen_height
        self.stack_base = self.memory_size
        self.stack_limit = self.memory_size - self.stack_size
        self.screen_address = self.stack_limit - self.screen_size
        self.keyboard_data_address = self.screen_address - 1
        self.keyboard_status_address = self.screen_address - 2
        self.font_addr = self.keyboard_status_address - 760
        self.mem_used = ( 760 + 1 + 1 + self.screen_size + self.stack_size)
        self.mem_free = self.memory_size - self.mem_used

        self.running = False; self.stop_event = threading.Event()
        self.gui = gui; self.update_callback = update_callback
        
        self.register_names = {0x10 + i: f"R{i}" for i in range(10)}
        self.register_pair_names = {0xA0: ("R0", "R1"), 0xA2: ("R2", "R3"), 0xA4: ("R4", "R5"), 0xA6: ("R6", "R7"), 0xA8: ("R8", "R9")}
        
        self.interrupt_requests = {'KEYBOARD': False}
        self.interrupt_vector_table = {'KEYBOARD': 0x0100}

        self.instruction_set = [InvalidInstruction()] * 256
        
        all_cpu_instructions = [
            Nop(), Hlt(), Ret(), PushA(), PopA(), PushF(), PopF(), Sti(), Cli(), Iret(),
            IncReg(), DecReg(), NotReg(), PushReg(), PopReg(), StoreInd(), LoadInd(), PushRp(), PopRp(),
            Jmp(), Call(), Jz(), Jnz(), Jc(), Jnc(), Jn(), Jnn(),
            Load(), Store(), LoadIx(), StoreIx(), LoadSp(),
            MovRegReg(), MovRegVal8(), MovRpVal16(),
            AddRegReg(), AddRegVal8(), AddRpRp(), AddRpVal16(),
            SubRegReg(), SubRegVal8(), SubRpRp(), SubRpVal16(),
            CmpRegReg(), CmpRegVal8(), CmpRpRp(), CmpRpVal16(),
            AndRegReg(), AndRegVal8(), OrRegReg(), OrRegVal8(), XorRegReg(), XorRegVal8(),
            MulRegReg(), DivRegReg(),
            ShlRegVal8(), ShrRegVal8(), RolRegVal8(), RorRegVal8(),
        ]
        for instr in all_cpu_instructions:
            self.instruction_set[instr.opcode] = instr

    def _set_flags(self, result, is_16bit=False, v1=0, v2=0, is_sub=False):
        mask = 0xFFFF if is_16bit else 0xFF; sign_bit = 0x8000 if is_16bit else 0x80
        self.flags['Z'] = 1 if (result & mask) == 0 else 0
        self.flags['N'] = 1 if (result & sign_bit) else 0
        op2 = -v2 if is_sub else v2
        if (v1 & sign_bit) == (op2 & sign_bit) and (v1 & sign_bit) != (result & sign_bit): self.flags['V'] = 1
        else: self.flags['V'] = 0
        if is_sub: self.flags['A'] = 1 if (v1 & 0x0F) < (v2 & 0x0F) else 0
        else: self.flags['A'] = 1 if ((v1 & 0x0F) + (v2 & 0x0F)) > 0x0F else 0
    def _set_carry(self, val): self.flags['C'] = 1 if val else 0
    def _pack_flags(self): return (self.flags['Z'] << 7) | (self.flags['N'] << 6) | (self.flags['C'] << 5) | (self.flags['V'] << 4) | (self.flags['A'] << 3) | (self.flags['I'] << 2)
    def _unpack_flags(self, byte): self.flags['Z']=(byte>>7)&1; self.flags['N']=(byte>>6)&1; self.flags['C']=(byte>>5)&1; self.flags['V']=(byte>>4)&1; self.flags['A']=(byte>>3)&1; self.flags['I']=(byte>>2)&1
    def _check_stack_op(self, size, is_push):
        new_sp = self.registers['SP'] - size if is_push else self.registers['SP'] + size
        if not (self.stack_limit <= new_sp <= self.stack_base): print(f"Stack {'Overflow' if is_push else 'Underflow'} Error. Halting."); self.running = False; return False
        return True
    def load_program(self, program_code, start_address):
        if start_address + len(program_code) > len(self.memory): print(f"Error: Program is too large. Size: {len(program_code)}"); return False
        self.memory[start_address:start_address + len(program_code)] = program_code
        return True
    
    def execute_instruction(self):
        pc = self.registers["PC"]
        opcode = self.memory[pc]
        instruction_to_execute = self.instruction_set[opcode]
        
        try:
            pc_increment = instruction_to_execute.execute(self)
            if self.running and pc_increment > 0:
                self.registers["PC"] = (pc + pc_increment) & 0xFFFF
        except (TypeError, IndexError, KeyError, ValueError) as e:
            print(f"CRITICAL CPU ERROR at PC=0x{pc:04X} (Op: 0x{opcode:02X}): {e}");
            import traceback; traceback.print_exc();
            self.running = False

    def handle_interrupts(self):
        if not self.flags['I'] or not self.interrupt_requests['KEYBOARD']: return False
        self.halted = False; self.interrupt_requests['KEYBOARD'] = False
        self._check_stack_op(3, True); current_pc = self.registers['PC']
        self.registers['SP'] -= 3
        self.memory[self.registers['SP']] = self._pack_flags()
        self.memory[self.registers['SP'] + 1] = (current_pc >> 8) & 0xFF
        self.memory[self.registers['SP'] + 2] = current_pc & 0xFF
        self.flags['I'] = 0; self.registers['PC'] = self.interrupt_vector_table['KEYBOARD']
        return True

    def run(self):
        self.running = True
        self.halted = False
        self.stop_event.clear()
        while self.running:
            if self.stop_event.is_set():
                self.running = False
                break

            # If HLT was executed, the program is done. Stop running.
            if self.halted:
                self.running = False
                continue

            pc_before = self.registers["PC"]
            if not (0 <= pc_before < self.memory_size):
                print(f"PC Error (0x{pc_before:04X}). Halting.")
                self.running = False
                break
            
            if not self.handle_interrupts():
                self.execute_instruction()
        
        print(f"CPU execution finished. Final PC: 0x{self.registers['PC']:04X}.")
        if self.update_callback:
            self.update_callback()

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
    def load_program_from_dialog(self):
        file_path = filedialog.askopenfilename(title="Open Assembly File", filetypes=(("Assembly Files", "*.asm"), ("All files", "*.*")))
        if not file_path: self.status_message = "File load cancelled."; self.info_dirty = True; return
        try:
             with open( file_path, "r" ) as f: code_text = f.read()
        except Exception as e: self.status_message = f"Error reading {os.path.basename(file_path)}"; self.info_dirty=True; return
        assembler = Assembler()
        assembled_code = assembler.assemble( code_text )
        if assembled_code is None: self.status_message = "Assembly Error."; self.info_dirty=True; self.code_loaded = False; return
        if self.cpu_thread and self.cpu_thread.is_alive(): self.stop_simulator()
        self.cpu.__init__( self.cpu.memory_size, self.cpu.stack_size, self.cpu.gui, self.cpu.update_callback )
        self.rebuild_memory_view(); self.vram_surface.fill( ( 0,0,0 ) )
        if self.cpu.load_program(font_data, self.cpu.font_addr): print("[GUI Loop] Font loaded into memory.")
        start_address = 0
        if self.cpu.load_program( assembled_code, start_address ):
            self.cpu.registers[ "PC" ] = 0; self.status_message = f"Loaded {os.path.basename(file_path)}. Ready."; self.code_loaded = True; self.info_dirty = True
        else: self.status_message = "Error loading program."; self.code_loaded = False; self.info_dirty = True
    def run_cpu(self):
        if self.cpu_thread and self.cpu_thread.is_alive(): self.status_message = "Simulator is already running."; self.info_dirty = True; return
        if not self.code_loaded: self.status_message = "No program loaded."; self.info_dirty = True; return
        self.status_message = f"Running from 0x{self.cpu.registers['PC']:04X}..."; self.info_dirty = True
        self.cpu_thread = threading.Thread(target=self.cpu.run, daemon=True); self.cpu_thread.start()
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