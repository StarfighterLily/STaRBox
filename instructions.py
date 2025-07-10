from abc import ABC, abstractmethod

# --- INSTRUCTION SET DEFINITION ---
class AssemblyError( Exception ):
    pass

class Instruction( ABC ):
    """Abstract Base Class for all instructions."""
    mnemonic = "INVALID"

    @abstractmethod
    def assemble( self, asm, operands, all_symbols ):
        """Try to assemble the given operands. Return bytecode list or raise AssemblyError."""
        pass

    @abstractmethod
    def execute( self, cpu ):
        """Execute the instruction. Return the number of bytes to increment PC."""
        pass

# --- Helper Methods for Instruction Classes ---
def _fetch_reg_name( cpu, offset=1 ):
    name = cpu.register_names.get( cpu.memory[ cpu.registers[ "PC" ] + offset ] )
    if name is None:
        raise ValueError( "Invalid register code." )
    return name

def _fetch_pair_names( cpu, offset=1 ):
    names = cpu.register_pair_names.get( cpu.memory[ cpu.registers[ "PC" ] + offset ] )
    if names is None:
        raise ValueError( "Invalid register pair code." )
    return names

def _fetch_addr16( cpu, offset=1 ):
    pc = cpu.registers[ "PC" ]
    return ( cpu.memory[ pc + offset ] << 8 ) | cpu.memory[ pc + offset + 1 ]

def _fetch_val8( cpu, offset=1 ):
    return cpu.memory[ cpu.registers["PC"] + offset ]

# --- Concrete Instruction Classes ---
class InvalidInstruction( Instruction ):
    def assemble( self, asm, operands, all_symbols ):
        raise AssemblyError( "Cannot assemble an invalid instruction." )
    def execute( self, cpu ):
        pc = cpu.registers[ "PC" ]
        opcode = cpu.memory[ pc ]
        print( f"Unknown instruction 0x{opcode:02X} at PC=0x{pc:04X}. Halting." )
        cpu.halted = True
        return 1

# --- Group: No Operand Instructions ---
class Nop( Instruction ):
    mnemonic, opcode, size = "NOP", 0x00, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( "NOP takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        return self.size

class Hlt( Instruction ):
    mnemonic, opcode, size = "HLT", 0xFF, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( "HLT takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu.halted = True
        return self.size

class Ret( Instruction ):
    mnemonic, opcode, size = "RET", 0x23, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu._check_stack_op( 2, False )
        hi = cpu.memory[ cpu.registers[ "SP" ] ]
        lo = cpu.memory[ cpu.registers[ "SP" ] + 1 ]
        cpu.registers[ "SP" ] += 2
        cpu.registers[ "PC" ] = ( hi << 8 ) | lo
        return 0

class PushA( Instruction ):
    mnemonic, opcode, size = "PUSHA", 0x53, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu._check_stack_op( 10, True )
        cpu.registers[ "SP" ] -= 10
        regs = [ f"R{i}" for i in range( 10 ) ]
        for i, r in enumerate( regs ):
            cpu.memory[ cpu.registers[ "SP" ] + i ] = cpu.registers[ r ]
        return self.size

class PopA( Instruction ):
    mnemonic, opcode, size = "POPA", 0x54, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu._check_stack_op( 10, False )
        regs = [ f"R{i}" for i in range( 10 ) ]
        for i, r in enumerate( regs ):
            cpu.registers[ r ] = cpu.memory[ cpu.registers[ "SP" ] + i ]
        cpu.registers[ "SP" ] += 10
        return self.size

class PushF( Instruction ):
    mnemonic, opcode, size = "PUSHF", 0x60, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu._check_stack_op( 1, True )
        cpu.registers[ "SP" ] -= 1
        cpu.memory[ cpu.registers[ "SP" ] ] = cpu._pack_flags()
        return self.size

class PopF( Instruction ):
    mnemonic, opcode, size = "POPF", 0x61, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu._check_stack_op( 1, False )
        cpu._unpack_flags( cpu.memory[ cpu.registers[ "SP" ] ] )
        cpu.registers[ "SP" ] += 1
        return self.size

class Sti( Instruction ):
    mnemonic, opcode, size = "STI", 0x70, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu.flags[ 'I' ] = 1
        return self.size

class Cli( Instruction ):
    mnemonic, opcode, size = "CLI", 0x71, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu.flags[ 'I' ] = 0
        return self.size

class Iret( Instruction ):
    mnemonic, opcode, size = "IRET", 0x72, 1
    def assemble( self, asm, operands, all_symbols ):
        if operands: raise AssemblyError( f"{self.mnemonic} takes no operands" )
        return [ self.opcode ]
    def execute( self, cpu ):
        cpu._check_stack_op( 3, False )
        flags_val = cpu.memory[ cpu.registers[ "SP" ] ]
        pc_hi = cpu.memory[ cpu.registers[ "SP" ] + 1 ]
        pc_lo = cpu.memory[ cpu.registers[ "SP" ] + 2 ]
        cpu._unpack_flags( flags_val )
        cpu.registers[ "PC" ] = ( pc_hi << 8 ) | pc_lo
        cpu.registers[ "SP" ] += 3
        return 0

# --- Group: Single Operand Instructions (Unchanged) ---
class IncReg( Instruction ):
    mnemonic, opcode, size = "INC", 0x01, 2
    def assemble( self, asm, operands, all_symbols ):
        if len( operands ) != 1 or not asm.is_register( operands[ 0 ] ):
            raise AssemblyError( "INC requires one register operand" )
        return [ self.opcode, asm.registers[ operands[0].upper() ] ]
    def execute( self, cpu ):
        reg = _fetch_reg_name( cpu )
        v1 = cpu.registers[ reg ]
        res = ( v1 + 1 ) & 0xFF
        cpu._set_flags( res, v1=v1, v2=1 )
        cpu.registers[ reg ] = res
        return self.size

class DecReg( Instruction ):
    mnemonic, opcode, size = "DEC", 0x02, 2
    def assemble( self, asm, operands, all_symbols ):
        if len( operands ) != 1 or not asm.is_register(operands[0]):
            raise AssemblyError( "DEC requires one register operand" )
        return [ self.opcode, asm.registers[ operands[ 0 ].upper() ] ]
    def execute( self, cpu ):
        reg = _fetch_reg_name( cpu )
        v1 = cpu.registers[ reg ]
        res = ( v1 - 1 ) & 0xFF
        cpu._set_flags( res, v1=v1, v2=1, is_sub=True )
        cpu.registers[ reg ] = res
        return self.size

class NotReg( Instruction ):
    mnemonic, opcode, size = "NOT", 0x43, 2
    def assemble( self, asm, operands, all_symbols ):
        if len( operands ) != 1 or not asm.is_register( operands[ 0 ] ):
            raise AssemblyError( "NOT requires one register operand" )
        return [ self.opcode, asm.registers[ operands[ 0 ].upper() ] ]
    def execute( self, cpu ):
        reg = _fetch_reg_name( cpu )
        res = ( ~cpu.registers[ reg ] ) & 0xFF
        cpu._set_flags( res )
        cpu.registers[ reg ] = res
        return self.size

class StoreInd( Instruction ):
    mnemonic, opcode, size = "STOREIND", 0x15, 2
    def assemble( self, asm, operands, all_symbols ):
        if len( operands ) != 1 or not asm.is_register( operands[ 0 ] ):
            raise AssemblyError( "STOREIND requires one register operand" )
        return [ self.opcode, asm.registers[ operands[ 0 ].upper() ] ]
    def execute( self, cpu ):
        reg = _fetch_reg_name( cpu )
        addr = ( cpu.registers[ "R0" ] << 8 ) | cpu.registers[ "R1" ]
        cpu.memory[ addr ] = cpu.registers[ reg ]
        return self.size

class LoadInd( Instruction ):
    mnemonic, opcode, size = "LOADIND", 0x16, 2
    def assemble( self, asm, operands, all_symbols ):
        if len( operands ) != 1 or not asm.is_register( operands[ 0 ] ):
            raise AssemblyError( "LOADIND requires one register operand" )
        return [ self.opcode, asm.registers[operands[ 0 ].upper() ] ]
    def execute( self, cpu ):
        reg = _fetch_reg_name( cpu )
        addr = ( cpu.registers[ "R0" ] << 8 ) | cpu.registers[ "R1" ]
        val = cpu.memory[ addr ]
        cpu.registers[ reg ] = val
        cpu._set_flags( val )
        return self.size

# --- MERGED INSTRUCTION: PUSH ---
class Push( Instruction ):
    mnemonic = "PUSH"
    def assemble( self, asm, operands, all_symbols ):
        if len(operands) != 1: raise AssemblyError("PUSH requires one operand")
        op1 = operands[0]
        if asm.is_register(op1):
            return [ 0x20, asm.registers[op1.upper()] ]
        elif asm.is_register_pair(op1):
            return [ 0x62, asm.register_pairs[op1.upper()] ]
        else:
            raise AssemblyError("PUSH requires a register or register pair operand")

    def execute( self, cpu ):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x20: # PUSH reg
            reg = _fetch_reg_name( cpu )
            cpu._check_stack_op( 1, True )
            cpu.registers[ "SP" ] -= 1
            cpu.memory[ cpu.registers[ "SP" ] ] = cpu.registers[ reg ]
            return 2
        elif opcode == 0x62: # PUSH rp
            ( r_hi, r_lo ) = _fetch_pair_names(cpu)
            cpu._check_stack_op( 2, True )
            cpu.registers[ "SP" ] -= 2
            cpu.memory[ cpu.registers[ "SP" ] ] = cpu.registers[ r_hi ]
            cpu.memory[ cpu.registers[ "SP" ] + 1 ] = cpu.registers[ r_lo ]
            return 2
        # This part should not be reached with correct assembly
        raise RuntimeError(f"Invalid opcode {opcode} for PUSH")

# --- MERGED INSTRUCTION: POP ---
class Pop( Instruction ):
    mnemonic = "POP"
    def assemble( self, asm, operands, all_symbols ):
        if len(operands) != 1: raise AssemblyError("POP requires one operand")
        op1 = operands[0]
        if asm.is_register(op1):
            return [ 0x21, asm.registers[op1.upper()] ]
        elif asm.is_register_pair(op1):
            return [ 0x63, asm.register_pairs[op1.upper()] ]
        else:
            raise AssemblyError("POP requires a register or register pair operand")

    def execute( self, cpu ):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x21: # POP reg
            reg = _fetch_reg_name( cpu )
            cpu._check_stack_op( 1, False )
            val = cpu.memory[ cpu.registers[ "SP" ] ]
            cpu.registers[ reg ] = val
            cpu.registers[ "SP" ] += 1
            cpu._set_flags( val )
            return 2
        elif opcode == 0x63: # POP rp
            ( r_hi, r_lo ) = _fetch_pair_names( cpu )
            cpu._check_stack_op( 2, False )
            cpu.registers[ r_hi ] = cpu.memory[ cpu.registers[ "SP" ] ]
            cpu.registers[ r_lo ] = cpu.memory[ cpu.registers[ "SP" ] + 1 ]
            cpu.registers[ "SP" ] += 2
            return 2
        # This part should not be reached with correct assembly
        raise RuntimeError(f"Invalid opcode {opcode} for POP")

# --- Group: Jump/Call Instructions (Unchanged) ---
class BaseJump( Instruction ): # Helper base class for jumps
    size = 3
    def assemble( self, asm, operands, all_symbols ):
        if len( operands ) != 1:
            raise AssemblyError( f"{self.mnemonic} requires one address operand" )
        addr = asm.parse_value( operands[ 0 ], 16, all_symbols )
        return [ self.opcode, ( addr >> 8 ) & 0xFF, addr & 0xFF ]

class Jmp( BaseJump ):
    mnemonic, opcode = "JMP", 0x06
    def execute( self, cpu ):
        addr = _fetch_addr16( cpu )
        cpu.registers[ "PC" ] = addr
        return 0

class Call( BaseJump ):
    mnemonic, opcode = "CALL", 0x22
    def execute( self, cpu ):
        addr = _fetch_addr16( cpu )
        cpu._check_stack_op( 2, True )
        ret_addr = ( cpu.registers[ "PC" ] + self.size ) & 0xFFFF
        cpu.registers[ "SP" ] -= 2
        cpu.memory[ cpu.registers[ "SP" ] ] = ( ret_addr >> 8 ) & 0xFF
        cpu.memory[ cpu.registers[ "SP" ] + 1 ] = ret_addr & 0xFF
        cpu.registers[ "PC" ] = addr
        return 0

class Jz( BaseJump ):
    mnemonic, opcode = "JZ", 0x30
    def execute( self, cpu ):
        if cpu.flags[ 'Z' ] == 1:
            cpu.registers[ "PC" ] = _fetch_addr16( cpu )
            return 0
        return self.size

class Jnz( BaseJump ):
    mnemonic, opcode = "JNZ", 0x31
    def execute( self, cpu ):
        if cpu.flags[ 'Z' ] == 0:
            cpu.registers[ "PC" ] = _fetch_addr16( cpu )
            return 0
        return self.size

class Jc( BaseJump ):
    mnemonic, opcode = "JC", 0x32
    def execute( self, cpu ):
        if cpu.flags[ 'C' ] == 1:
            cpu.registers[ "PC" ] = _fetch_addr16( cpu )
            return 0
        return self.size

class Jnc( BaseJump ):
    mnemonic, opcode = "JNC", 0x33
    def execute( self, cpu ):
        if cpu.flags[ 'C' ] == 0:
            cpu.registers[ "PC" ] = _fetch_addr16( cpu )
            return 0
        return self.size

class Jn( BaseJump ):
    mnemonic, opcode = "JN", 0x34
    def execute( self, cpu ):
        if cpu.flags[ 'N' ] == 1:
            cpu.registers[ "PC" ] = _fetch_addr16( cpu )
            return 0
        return self.size

class Jnn( BaseJump ):
    mnemonic, opcode = "JNN", 0x35
    def execute( self, cpu ):
        if cpu.flags[ 'N' ] == 0:
            cpu.registers[ "PC" ] = _fetch_addr16( cpu )
            return 0
        return self.size

# --- MERGED INSTRUCTION: MOV ---
class Mov(Instruction):
    mnemonic = "MOV"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("MOV requires two operands")
        op1, op2 = operands[0], operands[1]
        # MOV r, r
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x03, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        # MOV r, v8
        val_type = asm.parse_operand_type(op2, all_symbols)
        if asm.is_register(op1) and val_type == 'v8':
            return [0x08, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        # MOV rp, v16
        if asm.is_register_pair(op1) and val_type in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x51, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for MOV: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x03: # MOV r, r
            r1_name, r2_name = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            val = cpu.registers[r2_name]
            cpu.registers[r1_name] = val
            cpu._set_flags(val)
            return 3
        elif opcode == 0x08: # MOV r, v8
            reg_name, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            cpu.registers[reg_name] = val
            cpu._set_flags(val)
            return 3
        elif opcode == 0x51: # MOV rp, v16
            (r_hi, r_lo), val16 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            cpu.registers[r_hi] = (val16 >> 8) & 0xFF
            cpu.registers[r_lo] = val16 & 0xFF
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for MOV")

# --- MERGED INSTRUCTION: ADD ---
class Add(Instruction):
    mnemonic = "ADD"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("ADD requires two operands")
        op1, op2 = operands[0], operands[1]
        val_type2 = asm.parse_operand_type(op2, all_symbols)
        # ADD r, r
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x04, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        # ADD r, v8
        if asm.is_register(op1) and val_type2 == 'v8':
            return [0x09, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        # ADD rp, rp
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x52, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        # ADD rp, v16
        if asm.is_register_pair(op1) and val_type2 in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x58, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for ADD: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x04: # ADD r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            v1, v2 = cpu.registers[r1], cpu.registers[r2]
            res16 = v1 + v2
            cpu._set_carry(res16 > 0xFF)
            res = res16 & 0xFF
            cpu.registers[r1] = res; cpu._set_flags(res, v1=v1, v2=v2)
            return 3
        elif opcode == 0x09: # ADD r, v8
            reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            v1 = cpu.registers[reg]
            res16 = v1 + v2
            cpu._set_carry(res16 > 0xFF)
            res = res16 & 0xFF
            cpu.registers[reg] = res; cpu._set_flags(res, v1=v1, v2=v2)
            return 3
        elif opcode == 0x52: # ADD rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            res32 = v1 + v2; cpu._set_carry(res32 > 0xFFFF); res = res32 & 0xFFFF
            cpu._set_flags(res, True, v1, v2)
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            return 3
        elif opcode == 0x58: # ADD rp, v16
            (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
            res32 = v1 + v2; cpu._set_carry(res32 > 0xFFFF); res = res32 & 0xFFFF
            cpu._set_flags(res, True, v1, v2)
            cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for ADD")

# --- MERGED INSTRUCTION: SUB ---
class Sub(Instruction):
    mnemonic = "SUB"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("SUB requires two operands")
        op1, op2 = operands[0], operands[1]
        val_type2 = asm.parse_operand_type(op2, all_symbols)
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x05, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register(op1) and val_type2 == 'v8':
            return [0x0A, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x59, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        if asm.is_register_pair(op1) and val_type2 in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x5A, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for SUB: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x05: # SUB r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            v1, v2 = cpu.registers[r1], cpu.registers[r2]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
            cpu.registers[r1] = res; cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
            return 3
        elif opcode == 0x0A: # SUB r, v8
            reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            v1 = cpu.registers[reg]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
            cpu.registers[reg] = res; cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
            return 3
        elif opcode == 0x59: # SUB rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
            cpu._set_flags(res, True, v1, v2, is_sub=True)
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            return 3
        elif opcode == 0x5A: # SUB rp, v16
            (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
            cpu._set_flags(res, True, v1, v2, is_sub=True)
            cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for SUB")

# --- MERGED INSTRUCTION: CMP ---
class Cmp(Instruction):
    mnemonic = "CMP"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("CMP requires two operands")
        op1, op2 = operands[0], operands[1]
        val_type2 = asm.parse_operand_type(op2, all_symbols)
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x0E, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register(op1) and val_type2 == 'v8':
            return [0x0F, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x5B, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        if asm.is_register_pair(op1) and val_type2 in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x5C, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for CMP: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x0E: # CMP r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            v1, v2 = cpu.registers[r1], cpu.registers[r2]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
            cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
            return 3
        elif opcode == 0x0F: # CMP r, v8
            reg, v2 = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            v1 = cpu.registers[reg]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFF
            cpu._set_flags(res, v1=v1, v2=v2, is_sub=True)
            return 3
        elif opcode == 0x5B: # CMP rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
            cpu._set_flags(res, True, v1, v2, is_sub=True)
            return 3
        elif opcode == 0x5C: # CMP rp, v16
            (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
            cpu._set_carry(v1 >= v2); res = (v1 - v2) & 0xFFFF
            cpu._set_flags(res, True, v1, v2, is_sub=True)
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for CMP")

# --- MERGED INSTRUCTION: AND ---
class And(Instruction):
    mnemonic = "AND"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("AND requires two operands")
        op1, op2 = operands[0], operands[1]
        val_type2 = asm.parse_operand_type(op2, all_symbols)
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x40, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register(op1) and val_type2 == 'v8':
            return [0x44, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x68, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        if asm.is_register_pair(op1) and val_type2 in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x69, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for AND: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x40: # AND r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            res = cpu.registers[r1] & cpu.registers[r2]
            cpu.registers[r1] = res; cpu._set_flags(res); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x44: # AND r, v8
            reg, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            res = cpu.registers[reg] & val
            cpu.registers[reg] = res; cpu._set_flags(res); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x68: # AND rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            res = v1 & v2
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x69: # AND rp, v16
            (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
            res = v1 & v2
            cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True); cpu.flags['V'] = 0
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for AND")

# --- MERGED INSTRUCTION: OR ---
class Or(Instruction):
    mnemonic = "OR"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("OR requires two operands")
        op1, op2 = operands[0], operands[1]
        val_type2 = asm.parse_operand_type(op2, all_symbols)
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x41, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register(op1) and val_type2 == 'v8':
            return [0x45, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x6A, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        if asm.is_register_pair(op1) and val_type2 in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x6B, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for OR: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x41: # OR r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            res = cpu.registers[r1] | cpu.registers[r2]
            cpu.registers[r1] = res; cpu._set_flags(res); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x45: # OR r, v8
            reg, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            res = cpu.registers[reg] | val
            cpu.registers[reg] = res; cpu._set_flags(res); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x6A: # OR rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            res = v1 | v2
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x6B: # OR rp, v16
            (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
            res = v1 | v2
            cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True); cpu.flags['V'] = 0
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for OR")

# --- MERGED INSTRUCTION: XOR ---
class Xor(Instruction):
    mnemonic = "XOR"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("XOR requires two operands")
        op1, op2 = operands[0], operands[1]
        val_type2 = asm.parse_operand_type(op2, all_symbols)
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x42, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register(op1) and val_type2 == 'v8':
            return [0x46, asm.registers[op1.upper()], asm.parse_value(op2, 8, all_symbols)]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x6C, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        if asm.is_register_pair(op1) and val_type2 in ['v8', 'v16']:
            val16 = asm.parse_value(op2, 16, all_symbols)
            return [0x6D, asm.register_pairs[op1.upper()], (val16 >> 8) & 0xFF, val16 & 0xFF]
        raise AssemblyError(f"Invalid operands for XOR: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x42: # XOR r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            res = cpu.registers[r1] ^ cpu.registers[r2]
            cpu.registers[r1] = res; cpu._set_flags(res); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x46: # XOR r, v8
            reg, val = _fetch_reg_name(cpu, 1), _fetch_val8(cpu, 2)
            res = cpu.registers[reg] ^ val
            cpu.registers[reg] = res; cpu._set_flags(res); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x6C: # XOR rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            res = v1 ^ v2
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True); cpu.flags['V'] = 0
            return 3
        elif opcode == 0x6D: # XOR rp, v16
            (r_hi, r_lo), v2 = _fetch_pair_names(cpu, 1), _fetch_addr16(cpu, 2)
            v1 = (cpu.registers[r_hi] << 8) | cpu.registers[r_lo]
            res = v1 ^ v2
            cpu.registers[r_hi] = (res >> 8) & 0xFF; cpu.registers[r_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True); cpu.flags['V'] = 0
            return 4
        raise RuntimeError(f"Invalid opcode {opcode} for XOR")

# --- MERGED INSTRUCTION: MUL ---
class Mul(Instruction):
    mnemonic = "MUL"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("MUL requires two operands")
        op1, op2 = operands[0], operands[1]
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x50, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x6E, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        raise AssemblyError(f"Invalid operands for MUL: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x50: # MUL r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            v1, v2 = cpu.registers[r1], cpu.registers[r2]
            res16 = v1 * v2
            cpu._set_carry(res16 > 0xFF); res = res16 & 0xFF
            cpu.registers[r1] = res; cpu._set_flags(res)
            return 3
        elif opcode == 0x6E: # MUL rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            res32 = v1 * v2
            cpu._set_carry(res32 > 0xFFFF); res = res32 & 0xFFFF
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            cpu._set_flags(res, is_16bit=True)
            return 3
        raise RuntimeError(f"Invalid opcode {opcode} for MUL")

# --- MERGED INSTRUCTION: DIV ---
class Div(Instruction):
    mnemonic = "DIV"
    def assemble(self, asm, operands, all_symbols):
        if len(operands) != 2: raise AssemblyError("DIV requires two operands")
        op1, op2 = operands[0], operands[1]
        if asm.is_register(op1) and asm.is_register(op2):
            return [0x64, asm.registers[op1.upper()], asm.registers[op2.upper()]]
        if asm.is_register_pair(op1) and asm.is_register_pair(op2):
            return [0x6F, asm.register_pairs[op1.upper()], asm.register_pairs[op2.upper()]]
        raise AssemblyError(f"Invalid operands for DIV: {op1}, {op2}")

    def execute(self, cpu):
        opcode = cpu.memory[cpu.registers["PC"]]
        if opcode == 0x64: # DIV r, r
            r1, r2 = _fetch_reg_name(cpu, 1), _fetch_reg_name(cpu, 2)
            v1, v2 = cpu.registers[r1], cpu.registers[r2]
            if v2 == 0: cpu.running = False; return 0
            res = v1 // v2; rem = v1 % v2
            cpu.registers[r1] = res; cpu.registers[r2] = rem; cpu._set_flags(res)
            return 3
        elif opcode == 0x6F: # DIV rp, rp
            (d_hi, d_lo), (s_hi, s_lo) = _fetch_pair_names(cpu, 1), _fetch_pair_names(cpu, 2)
            v1 = (cpu.registers[d_hi] << 8) | cpu.registers[d_lo]
            v2 = (cpu.registers[s_hi] << 8) | cpu.registers[s_lo]
            if v2 == 0: cpu.running = False; return 0
            res = v1 // v2; rem = v1 % v2
            cpu.registers[d_hi] = (res >> 8) & 0xFF; cpu.registers[d_lo] = res & 0xFF
            cpu.registers[s_hi] = (rem >> 8) & 0xFF; cpu.registers[s_lo] = rem & 0xFF
            cpu._set_flags(res, is_16bit=True)
            return 3
        raise RuntimeError(f"Invalid opcode {opcode} for DIV")


# --- Group: Shift/Rotate Instructions (Unchanged) ---
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

# --- Group: Memory & 16-bit Value Instructions (Unchanged) ---
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

# --- Group: Indexed Addressing Instructions (Unchanged) ---
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