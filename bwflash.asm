; Fills the entire video memory with white, then black, continuously.
; This creates a flashing effect to test rendering performance.

SCREEN_START_HI EQU 0xD6
SCREEN_START_LO EQU 0xEF

SCREEN_END_HI   EQU 0xFD
SCREEN_END_LO   EQU 0xFF

WHITE           EQU 15
BLACK           EQU 0      ; Added constant for black

; --- Main Program ---
MASTER_LOOP:
    ; --- Fill screen with WHITE ---
    MOV R0, SCREEN_START_HI  ; Reset high byte of address pointer
    MOV R1, SCREEN_START_LO  ; Reset low byte of address pointer
    MOV R2, WHITE            ; Load the color white into R2

WHITE_LOOP:
    STOREIND R2              ; Write the color to the address in R0:R1

    ; Check if the white fill is done
    CMP R0, SCREEN_END_HI
    JNZ CONTINUE_WHITE_LOOP
    CMP R1, SCREEN_END_LO
    JZ START_BLACK_FILL      ; Done with white, so start the black fill

CONTINUE_WHITE_LOOP:
    ; Increment the 16-bit address in R0:R1
    INC R1
    JNZ WHITE_LOOP           ; If low byte is not zero, continue
    INC R0                   ; Otherwise, increment high byte
    JMP WHITE_LOOP

START_BLACK_FILL:
    ; --- Fill screen with BLACK ---
    MOV R0, SCREEN_START_HI  ; Reset high byte of address pointer
    MOV R1, SCREEN_START_LO  ; Reset low byte of address pointer
    MOV R2, BLACK            ; Load the color black into R2

BLACK_LOOP:
    STOREIND R2              ; Write the color to the address in R0:R1

    ; Check if the black fill is done
    CMP R0, SCREEN_END_HI
    JNZ CONTINUE_BLACK_LOOP
    CMP R1, SCREEN_END_LO
    JZ MASTER_LOOP           ; Done with black, so repeat the entire process

CONTINUE_BLACK_LOOP:
    ; Increment the 16-bit address in R0:R1
    INC R1
    JNZ BLACK_LOOP           ; If low byte is not zero, continue
    INC R0                   ; Otherwise, increment high byte
    JMP BLACK_LOOP

; The HLT is no longer reachable but is good practice to have at the end
DONE:
    HLT