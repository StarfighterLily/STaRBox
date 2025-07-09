; Fills the entire video memory with white pixels using 8-bit operations
; to manually handle a 16-bit address, avoiding assembler ambiguity.

SCREEN_START_HI EQU 0xD6
SCREEN_START_LO EQU 0xEF

SCREEN_END_HI   EQU 0xFD
SCREEN_END_LO   EQU 0xFF

WHITE           EQU 15

; --- Main Program ---
START:
    MOV R0, SCREEN_START_HI  ; Set high byte of address pointer
    MOV R1, SCREEN_START_LO  ; Set low byte of address pointer
    MOV R2, WHITE            ; Load the color white into R2

LOOP:
    STOREIND R2              ; Write the color to the address in R0:R1

    ; Check if we have reached the last pixel address (0xFDFF)
    CMP R0, SCREEN_END_HI
    JNZ CONTINUE_LOOP        ; If high byte doesn't match, keep looping
    CMP R1, SCREEN_END_LO
    JZ DONE                  ; If both high and low bytes match, we're done

CONTINUE_LOOP:
    ; Increment the 16-bit address in R0:R1 manually
    INC R1                   ; Increment the low byte of the address
    JNZ LOOP                 ; If it didn't wrap around (is not zero), continue loop
    
    INC R0                   ; If R1 wrapped to zero, increment the high byte
    JMP LOOP                 ; And continue the loop

DONE:
    HLT                      ; Halt the CPU