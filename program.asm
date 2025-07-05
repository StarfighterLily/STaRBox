# Screen Border Demo for STaRbox Simulator
# Draws a white border and then halts.

# --- Program Start ---
START:
    CALL DRAW_BOX

DONE:
    JMP DONE

# =======================================================
# Subroutine: DRAW_BOX
# =======================================================
DRAW_BOX:
    # --- Draw Top Line (Y=0) ---
    MOVI R3, 0
    MOVI R2, 0
DRAW_TOP_LOOP:
    CMPI R2, 100
    JC DRAW_BOTTOM_LINE
    CALL CALC_ADDRESS
    MOVI R4, 14
    STOREIND R4
    INC R2
    JMP DRAW_TOP_LOOP

DRAW_BOTTOM_LINE:
    # --- Draw Bottom Line (Y=99) ---
    MOVI R3, 99
    MOVI R2, 0
DRAW_BOTTOM_LOOP:
    CMPI R2, 100
    JC DRAW_LEFT_LINE
    CALL CALC_ADDRESS
    MOVI R4, 14
    STOREIND R4
    INC R2
    JMP DRAW_BOTTOM_LOOP

DRAW_LEFT_LINE:
    # --- Draw Left Line (X=0) ---
    MOVI R2, 0
    MOVI R3, 1
DRAW_LEFT_LOOP:
    CMPI R3, 99
    JC DRAW_RIGHT_LINE
    CALL CALC_ADDRESS
    MOVI R4, 14
    STOREIND R4
    INC R3
    JMP DRAW_LEFT_LOOP

DRAW_RIGHT_LINE:
    # --- Draw Right Line (X=99) ---
    MOVI R2, 99
    MOVI R3, 1
DRAW_RIGHT_LOOP:
    CMPI R3, 99
    JC DRAW_BOX_DONE
    CALL CALC_ADDRESS
    MOVI R4, 14
    STOREIND R4
    INC R3
    JMP DRAW_RIGHT_LOOP

DRAW_BOX_DONE:
    RET

# =======================================================
# Subroutine: CALC_ADDRESS
# Calculates screen address: 0xD8F0 + (Y * 100) + X
# =======================================================
CALC_ADDRESS:
    PUSH R2
    PUSH R3
    PUSH R4

    MOVI R0, 0xD8
    MOVI R1, 0xF0

ADD_Y_LOOP:
    CMPI R3, 0
    JZ ADD_X
    MOVI R4, 100
    ADD R1, R4
    JNC NO_Y_CARRY
    INC R0
NO_Y_CARRY:
    DEC R3
    JMP ADD_Y_LOOP

ADD_X:
    ADD R1, R2
    JNC FINISHED
    INC R0

FINISHED:
    POP R4
    POP R3
    POP R2
    RET