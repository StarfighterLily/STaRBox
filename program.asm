# Bouncing Pixel Demo for STaRbox Simulator
# Requires the custom STOREIND (0x15) instruction.
# Uses hardcoded constants instead of EQU.

# --- Variable Addresses (in low memory) ---
# ADDR_VAR_X      EQU 0x0000 # (Using 0x0000 directly)
# ADDR_VAR_Y      EQU 0x0001 # (Using 0x0001 directly)
# ADDR_VAR_DX     EQU 0x0002 # (Using 0x0002 directly)
# ADDR_VAR_DY     EQU 0x0003 # (Using 0x0003 directly)

# --- Program Start ---
START:
    # Initialize position (X=50, Y=50)
    MOVI R2, 50         # R2 = Initial X
    STORE 0x0000, R2    # Store X @ ADDR_VAR_X
    MOVI R3, 50         # R3 = Initial Y
    STORE 0x0001, R3    # Store Y @ ADDR_VAR_Y

    # Initialize velocity (DX=1, DY=1)
    MOVI R4, 1          # R4 = Initial DX
    STORE 0x0002, R4    # Store DX @ ADDR_VAR_DX
    MOVI R4, 1          # R4 = Initial DY
    STORE 0x0003, R4    # Store DY @ ADDR_VAR_DY

# --- Main Loop ---
MAIN_LOOP:
    # Load current position
    LOAD R2, 0x0000     # R2 = X from ADDR_VAR_X
    LOAD R3, 0x0001     # R3 = Y from ADDR_VAR_Y

    # Calculate screen address for current X, Y
    CALL CALC_ADDRESS    # Output: R0=AddrHi, R1=AddrLo

    # Erase old pixel
    MOVI R4, 0           # R4 = Color BLACK (0)
    STOREIND R4          # Memory[R0:R1] = R4 (BLACK)

    # Load velocities
    LOAD R4, 0x0002     # R4 = DX from ADDR_VAR_DX

    # Update position
    ADD R2, R4           # R2 = R2 + DX (X = X + DX)

    LOAD R4, 0x0003     # R4 = DY from ADDR_VAR_DY
    ADD R3, R4           # R3 = R3 + DY (Y = Y + DY)

    # --- Check X Bounds ---
    CMPI R2, 0           # Compare X with 0
    JNN CHECK_X_HIGH     # If X >= 0 (Not Negative), check high bound
    # X is < 0 (Negative flag set)
    CALL NEGATE_DX       # Negate DX
    MOVI R2, 0           # Clamp X to 0
    JMP CHECK_Y_BOUNDS   # Skip high bound check

CHECK_X_HIGH:
    CMPI R2, 100         # Compare X with SCREEN_WIDTH (100)
    JC CHECK_Y_BOUNDS    # If X < 100 (Carry set), X is in bounds
    # X is >= 100
    CALL NEGATE_DX       # Negate DX
    MOVI R4, 99          # R4 = SCREEN_WIDTH - 1
    MOV R2, R4           # Clamp X to 99
    # Fall through to CHECK_Y_BOUNDS

    # --- Check Y Bounds (similar logic) ---
CHECK_Y_BOUNDS:
    CMPI R3, 0           # Compare Y with 0
    JNN CHECK_Y_HIGH     # If Y >= 0, check high bound
    # Y is < 0
    CALL NEGATE_DY       # Negate DY
    MOVI R3, 0           # Clamp Y to 0
    JMP STORE_POS        # Skip high bound check

CHECK_Y_HIGH:
    CMPI R3, 100         # Compare Y with SCREEN_HEIGHT (100)
    JC STORE_POS         # If Y < 100, Y is in bounds
    # Y is >= 100
    CALL NEGATE_DY       # Negate DY
    MOVI R4, 99          # R4 = SCREEN_HEIGHT - 1
    MOV R3, R4           # Clamp Y to 99
    # Fall through to STORE_POS

STORE_POS:
    # Store updated position back to memory
    STORE 0x0000, R2     # Store new X @ ADDR_VAR_X
    STORE 0x0001, R3     # Store new Y @ ADDR_VAR_Y

    # Calculate screen address for new X, Y
    CALL CALC_ADDRESS    # Output: R0=AddrHi, R1=AddrLo

    # Draw new pixel
    MOVI R4, 15          # R4 = Color WHITE (15)
    STOREIND R4          # Memory[R0:R1] = R4 (WHITE)

    # --- Simple Delay Loop ---
    # Adjust count in R4 for faster/slower speed
    MOVI R4, 255          # Delay counter
DELAY_LOOP:
    DEC R4
    JNZ DELAY_LOOP

    JMP MAIN_LOOP        # Repeat forever

# =======================================================
# Subroutine: CALC_ADDRESS
# Calculates screen address: 0xD900 + (Y * 100) + X
# Input: R2 = X, R3 = Y
# Output: R0 = Address High Byte, R1 = Address Low Byte
# Clobbers: R4
# =======================================================
CALC_ADDRESS:
    # Calculate Y * 100
    MOVI R0, 0           # Accumulator High Byte = 0
    MOVI R1, 0           # Accumulator Low Byte = 0
MUL_LOOP:
    CMPI R3, 0           # Is Y counter zero?
    JZ MUL_DONE          # If yes, multiplication is done

    # Add SCREEN_WIDTH (100) to R0:R1 accumulator
    MOVI R4, 100         # R4 = 100 (0x64)
    ADD R1, R4           # Add 100 to low byte (R1)
    JNC NO_CARRY1        # If R1 addition didn't overflow (carry=0)
    INC R0               # Else, increment high byte (R0)
NO_CARRY1:
    DEC R3               # Decrement Y counter
    JMP MUL_LOOP
MUL_DONE:
    # Add X (from R2) to the result (Y * 100)
    MOV R4, R2           # Move X into R4 temporarily
    ADD R1, R4           # Add X to low byte (R1)
    JNC NO_CARRY2        # If no carry from low byte add
    INC R0               # Else, increment high byte (R0)
NO_CARRY2:
    # Add SCREEN_BASE (0xD900)
    MOVI R4, 0x00        # R4 = SCREEN_BASE_LO
    ADD R1, R4           # Add low byte of base (no effect)
    JNC NO_CARRY3        # Check carry (shouldn't happen)
    INC R0               # Increment high byte if carry occurred
NO_CARRY3:
    MOVI R4, 0xD9        # R4 = SCREEN_BASE_HI
    ADD R0, R4           # Add high byte of base to R0

    RET # Result R0:R1 holds final address

# =======================================================
# Subroutine: NEGATE_DX
# Clobbers: R4
# =======================================================
NEGATE_DX:
    LOAD R4, 0x0002      # Load DX into R4 from ADDR_VAR_DX
    CALL NEGATE_R4       # Call common negate routine
    STORE 0x0002, R4     # Store negated DX back @ ADDR_VAR_DX
    RET

# =======================================================
# Subroutine: NEGATE_DY
# Clobbers: R4
# =======================================================
NEGATE_DY:
    LOAD R4, 0x0003      # Load DY into R4 from ADDR_VAR_DY
    CALL NEGATE_R4       # Call common negate routine
    STORE 0x0003, R4     # Store negated DY back @ ADDR_VAR_DY
    RET

# =======================================================
# Subroutine: NEGATE_R4
# Input: R4 = value. Output: R4 = -value
# Clobbers: R0
# =======================================================
NEGATE_R4:
    MOVI R0, 0           # R0 = 0
    SUB R0, R4           # R0 = 0 - R4
    MOV R4, R0           # Move result back to R4
    RET

# --- End of Program ---