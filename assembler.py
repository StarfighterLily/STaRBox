import re
# The import list has been updated to use the new merged instruction classes.
from instructions import (Instruction, AssemblyError, Nop, Hlt, Ret, PushA, PopA, PushF, PopF, Sti,
                          Cli, Iret, IncReg, DecReg, NotReg, StoreInd, LoadInd, Jmp, Call, Jz, Jnz, Jc,
                          Jnc, Jn, Jnn, Load, Store, LoadIx, StoreIx, LoadSp, ShlRegVal8, ShrRegVal8,
                          RolRegVal8, RorRegVal8, Mov, Add, Sub, Cmp, And, Or, Xor, Mul, Div, Push, Pop)

class Assembler:
    def __init__( self ):
        self.registers = { f"R{i}": 0x10 + i for i in range( 10 ) }
        self.register_pairs = { "R0": 0xA0, "R2": 0xA2, "R4": 0xA4, "R6": 0xA6, "R8": 0xA8 }
        self.current_pass = 0

        # The instructions dictionary now uses the new merged classes, simplifying the assembler's logic.
        self.instructions = {
            "NOP": [ Nop() ], "HLT": [ Hlt() ], "RET": [ Ret() ], "PUSHA": [ PushA() ], "POPA": [ PopA() ],
            "PUSHF": [ PushF() ], "POPF": [ PopF() ], "STI": [ Sti() ], "CLI": [ Cli() ], "IRET": [ Iret() ],
            "INC": [ IncReg() ], "DEC": [ DecReg() ], "NOT": [ NotReg() ],
            "STOREIND": [ StoreInd() ], "LOADIND": [ LoadInd() ],
            "JMP": [ Jmp() ], "CALL": [ Call() ], "JZ": [ Jz() ], "JNZ": [ Jnz() ], "JC": [ Jc() ], "JNC": [ Jnc() ], "JN": [ Jn() ], "JNN": [ Jnn() ],
            "LOAD": [ Load() ], "STORE": [ Store() ], "LOADIX": [ LoadIx() ], "STOREIX": [ StoreIx() ], "LOADSP": [ LoadSp() ],
            "SHL": [ ShlRegVal8() ], "SHR": [ ShrRegVal8() ], "ROL": [ RolRegVal8() ], "ROR": [ RorRegVal8() ],
            # Merged Instructions
            "MOV": [ Mov() ], "ADD": [ Add() ], "SUB": [ Sub() ], "CMP": [ Cmp() ],
            "AND": [ And() ], "OR": [ Or() ], "XOR": [ Xor() ], "MUL": [ Mul() ], "DIV": [ Div() ],
            "PUSH": [ Push() ], "POP": [ Pop() ],
        }

    def is_register( self, op_str ):
        return op_str.upper() in self.registers

    def is_register_pair(self, op_str):
        return op_str.upper() in self.register_pairs

    def is_indexed_sp( self, op_str ):
        return bool( re.match( r"\[\s*SP\s*\+\s*[^\]]+\]", op_str.upper() ) )

    def is_indexed_pair( self, op_str ):
        return bool( re.match( r"\[\s*(R0|R2|R4|R6|R8)\s*\+\s*[^\]]+\]", op_str.upper() ) )

    def parse_indexed_operand( self, operand, all_symbols ):
        match = re.match( r"\[\s*(R0|R2|R4|R6|R8|SP)\s*\+\s*([^\]]+)\]", operand.upper() )
        if not match:
            return None, None
        reg_name, val_str = match.groups()
        reg_code = self.register_pairs.get( reg_name )
        offset = self.parse_value( val_str, 8, all_symbols )
        return reg_code, offset
        
    def _evaluate_constant_expression( self, expr_str, constants ):
        work_expr = expr_str.strip().upper()
        for const_name in sorted( constants.keys(), key=len, reverse=True ):
            work_expr = re.sub( r'\b' + re.escape( const_name ) + r'\b', str( constants[ const_name ] ), work_expr )
        temp_expr = re.sub( r'0X[0-9A-F]+', '', work_expr )
        temp_expr = re.sub( r'0B[01]+', '', temp_expr )
        temp_expr = re.sub( r'[0-9]+', '', temp_expr )
        temp_expr = re.sub( r'[+\-*/&|<>^~()]', '', temp_expr )
        temp_expr = temp_expr.strip()
        if temp_expr: return None
        try:
            safe_dict = { '__builtins__': {} }
            return eval( work_expr, safe_dict, {} )
        except Exception:
            return None

    def parse_operand_type( self, operand_str, constants_dict ):
        op_upper = operand_str.upper()
        if self.is_register( op_upper ):
            return "r"
        if self.is_register_pair( op_upper ):
            return "rp"
        if self.is_indexed_sp( op_upper ):
            return "idx_sp"
        if self.is_indexed_pair( op_upper ):
            return "idx"
        val = self._evaluate_constant_expression( operand_str, constants_dict )
        if val is not None:
            return "v8" if 0 <= val <= 0xFF else "v16"
        return "v16"
    
    def parse_value( self, operand, bits, all_symbols ):
        val = self._evaluate_constant_expression( operand, all_symbols )
        if val is None:
            op_str = operand.strip().upper()
            if self.current_pass == 1 and re.match( r'^[A-Z_][A-Z0-9_]*$', op_str ):
                 return 0
            raise AssemblyError( f"Cannot resolve expression '{operand}'" )

        max_val = ( 1 << bits ) - 1
        if not ( 0 <= val <= max_val ):
            if bits == 8 and -128 <= val < 0:
                val &= 0xFF
            elif bits == 16 and -32768 <= val < 0:
                val &= 0xFFFF
            else:
                print( f"Warning: Value of '{operand}' ({val}) is out of {bits}-bit range." )
        return val

    def assemble( self, assembly_code ):
        lines = assembly_code.strip().splitlines()

        def define_constants( lines_list ):
            constants, remaining_lines = {}, []
            for line in lines_list:
                clean_line = line.split( ';', 1 )[ 0 ].strip()
                if not clean_line: continue
                parts = clean_line.split()
                if len( parts ) >= 3 and parts[ 1 ].upper() == 'EQU':
                    val = self._evaluate_constant_expression( parts[ 2 ], constants )
                    if val is None:
                        print( f"Error: Could not evaluate EQU value for '{parts[ 0 ]}'" )
                        return None, None
                    constants[ parts[ 0 ].upper() ] = val
                else:
                    remaining_lines.append( line )
            return constants, remaining_lines

        def split_operands( operand_string ):
            if not operand_string: return []
            return [ op.strip() for op in re.split( r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", operand_string ) ]

        # --- Pass 1 ---
        print( "[Assembler] Pass 1: Defining constants, labels, and data..." )
        self.current_pass = 1 # Bug Fix: Set current pass to 1
        constants, lines = define_constants( lines )
        if constants is None:
            return None

        labels, code_intermediate, current_address = {}, [], 0
        for line_num, line in enumerate( lines, 1 ):
            line = line.split( ';', 1 )[ 0 ].strip()
            if not line: continue

            label_def = None
            if ':' in line:
                label, instruction_part = line.split( ':', 1 )
                label = label.strip().upper()
                if label in labels:
                    print( f"Error (Line {line_num}): Duplicate label '{label}'" )
                    return None
                labels[ label ] = current_address
                label_def = label
                line = instruction_part.strip()

            if not line:
                code_intermediate.append( { 'line': line_num, 'address': current_address, 'label': label_def, 'op': 'NOP_LABEL', 'operands': [], 'size': 0 } )
                continue

            parts = line.split( maxsplit = 1 )
            op = parts[ 0 ].upper()
            operands_str = parts[ 1 ] if len( parts ) > 1 else ""
            operands = split_operands( operands_str )
            
            if op == "ORG":
                new_addr = self.parse_value( operands[ 0 ], 16, constants )
                if not ( 0 <= new_addr < 65535 ):
                    print( f"Error (Line {line_num}): ORG address 0x{new_addr:04X} is out of bounds." )
                    return None
                current_address = new_addr
                continue

            if op in [ "DB", "DW" ]:
                data_size = 0
                if op == "DB":
                    for op_str in operands:
                        data_size += len( op_str[ 1:-1 ].encode( 'ascii' ) ) if op_str.startswith( '"' ) else 1
                elif op == "DW":
                    data_size += len( operands ) * 2
                code_intermediate.append( { 'line': line_num, 'address': current_address, 'op': op, 'operands': operands, 'size': data_size } )
                current_address += data_size
                continue

            instruction_variants = self.instructions.get( op )
            if not instruction_variants:
                print( f"Error (Line {line_num}): Unknown instruction '{op}'" )
                return None

            matched_instruction = None
            bytecode_for_size_calc = None # Must get bytecode to determine size
            all_symbols_pass1 = { **constants, **labels }
            for instr_obj in instruction_variants:
                try:
                    bytecode_for_size_calc = instr_obj.assemble( self, operands, all_symbols_pass1 )
                    matched_instruction = instr_obj
                    break 
                except ( AssemblyError, KeyError, IndexError, TypeError ):
                    continue 

            if not matched_instruction:
                print( f"Error (Line {line_num}): Invalid operands for '{op}': {operands}" )
                return None
            
            # The size is now determined from the length of the assembled bytecode
            instruction_size = len(bytecode_for_size_calc)
            code_intermediate.append( {
                'line': line_num, 'address': current_address, 'label': label_def, 
                'op': op, 'operands': operands, 'size': instruction_size,
                'instruction_obj': matched_instruction
            } )
            current_address += instruction_size
        print( f"[Assembler] Pass 1 Complete. Labels: {labels}" )

        # --- Pass 2 ---
        print( "[Assembler] Pass 2: Generating bytecode..." )
        self.current_pass = 2
        final_code = bytearray()
        all_symbols = { **constants, **labels }

        for item in code_intermediate:
            if item[ 'op' ] == 'NOP_LABEL':
                continue
            line_num = item[ 'line' ]

            if item[ 'op' ] in [ "DB", "DW" ]:
                for op_str in item[ 'operands' ]:
                    if item[ 'op' ] == "DB":
                        if op_str.startswith( '"' ) and op_str.endswith( '"' ):
                            final_code.extend( op_str[ 1:-1 ].encode( 'ascii' ) )
                        else:
                            final_code.append( self.parse_value( op_str, 8, all_symbols ) )
                    elif item[ 'op' ] == "DW":
                        val = self.parse_value( op_str, 16, all_symbols )
                        final_code.extend( [ ( val >> 8 ) & 0xFF, val & 0xFF ] )
                continue

            try:
                instr_obj = item[ 'instruction_obj' ]
                bytecode = instr_obj.assemble( self, item[ 'operands' ], all_symbols )
                final_code.extend( bytecode )
            except ( AssemblyError, KeyError, IndexError, TypeError ) as e:
                print( f"Assembly Error (Line {line_num}): Could not assemble '{item[ 'op' ]}'. Details: {e}" )
                import traceback; traceback.print_exc()
                return None
        
        print( f"[Assembler] Pass 2 Complete. Final code size: {len( final_code )} bytes." )
        return final_code