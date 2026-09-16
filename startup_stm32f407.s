.syntax unified
.cpu cortex-m4
.fpu softvfp
.thumb

.global Reset_Handler
.global Default_Handler
.global _init
.global _fini
.global SysTick_Handler
.global DMA1_Stream3_IRQHandler

/*
 * Copied from ../stm32f4_audio_examples/audio_in_to_wav/startup_stm32f407.s
 * unchanged. DMA1_Stream3_IRQHandler (IRQ 14, SPI2_RX / PDM mic capture) and
 * SysTick_Handler are defined in pdm_mic.c / hal.c, referenced via relative
 * path from that same shared repo (see ../CMakeLists.txt and ../README.md).
 * The FPU-enable-first / .got-before-.data / CCM-zero / init_newlib-before-
 * __libc_init_array ordering here already satisfies everything ExecuTorch
 * needs (see ../first_et_project's "Gotchas" list) -- no changes required
 * for this project beyond what audio_in_to_wav already has.
 */

/* ------------------------------------------------------------------ */
/*  Interrupt Vector Table                                              */
/* ------------------------------------------------------------------ */
.section .isr_vector,"a",%progbits
.type isr_vector, %object
isr_vector:
.word _estack           /* initial stack pointer */
.word Reset_Handler     /* reset handler — Thumb bit set by .thumb_func */
.word Default_Handler   /* NMI */
.word Default_Handler   /* HardFault */
.word Default_Handler   /* MemManage */
.word Default_Handler   /* BusFault */
.word Default_Handler   /* UsageFault */
.word 0
.word 0
.word 0
.word 0
.word Default_Handler   /* SVCall */
.word Default_Handler   /* Debug Monitor */
.word 0
.word Default_Handler   /* PendSV */
.word SysTick_Handler   /* SysTick */
/* External interrupts 0-13 */
.rept 14
.word Default_Handler
.endr
.word DMA1_Stream3_IRQHandler  /* IRQ 14 = DMA1_Stream3 (SPI2_RX / PDM mic) */
/* External interrupts 15-239 */
.rept 225
.word Default_Handler
.endr
.size isr_vector, .-isr_vector

/* ------------------------------------------------------------------ */
/*  Reset Handler                                                       */
/* ------------------------------------------------------------------ */
.section .text.Reset_Handler
.thumb_func
.type Reset_Handler, %function
Reset_Handler:

    /*
     * 1. ENABLE THE FPU — must be the very first thing.
     *
     * Set CP10 and CP11 to full access in CPACR at 0xE000ED88.
     * Without this, any floating-point instruction causes a NOCP
     * UsageFault (CFSR bit 19 = 0x00080000).
     */
    ldr r0, =0xE000ED88
    ldr r1, [r0]
    orr r1, r1, #(0xF << 20)
    str r1, [r0]
    dsb
    isb

    /*
     * 2. COPY .data FROM FLASH TO RAM.
     *
     * Initialized variables live in flash (LMA = _sidata) and must be
     * copied to RAM (VMA = _sdata.._edata) before any C code runs.
     * .got is placed first in .data (see linker script).
     */
    ldr r0, =_sdata
    ldr r1, =_edata
    ldr r2, =_sidata
copy_loop:
    cmp r0, r1
    bge zero_bss
    ldr r3, [r2], #4
    str r3, [r0], #4
    b copy_loop

    /*
     * 3. ZERO .bss SECTION.
     */
zero_bss:
    ldr r0, =_sbss
    ldr r1, =_ebss
    mov r2, #0
bss_loop:
    cmp r0, r1
    bge zero_ccm
    str r2, [r0], #4
    b bss_loop

    /*
     * 3b. ZERO .ccmram (NOLOAD — no flash image, must be zeroed like .bss).
     */
zero_ccm:
    ldr r0, =_sccmram
    ldr r1, =_eccmram
    mov r2, #0
ccm_zero_loop:
    cmp r0, r1
    bge call_init_newlib
    str r2, [r0], #4
    b ccm_zero_loop

    /*
     * 4. INITIALIZE NEWLIB REENTRANT STRUCTURE.
     *
     * newlib-nano's stdio (snprintf etc.) requires _impure_ptr to be
     * initialized before use. init_newlib() calls _REENT_INIT_PTR().
     */
call_init_newlib:
    bl init_newlib

    /*
     * 5. RUN C GLOBAL CONSTRUCTORS via __libc_init_array.
     *
     * __libc_init_array calls _init() before iterating .init_array.
     * We use -nostartfiles (no crti.o/crtn.o) so _init() must be
     * provided as a stub (see below).
     */
call_init:
    bl __libc_init_array

call_main:
    bl main
hang:
    b hang
.size Reset_Handler, .-Reset_Handler

/* ------------------------------------------------------------------ */
/*  Default (fault) Handler                                             */
/*                                                                      */
/*  When stopped here under GDB, inspect:                               */
/*    r1 = PC at fault                                                  */
/*    r2 = LR at fault                                                  */
/*    r3 = CFSR value                                                   */
/*  Common CFSR values:                                                 */
/*    0x00000001 = IACCVIOL   (instruction access violation)            */
/*    0x00020000 = UNALIGNED  (unaligned memory access)                 */
/*    0x00080000 = NOCP       (FPU not enabled)                         */
/*    0x00010000 = UNDEFINSTR (undefined instruction)                   */
/* ------------------------------------------------------------------ */
.section .text.Default_Handler
.thumb_func
.type Default_Handler, %function
Default_Handler:
    tst lr, #4
    ite eq
    mrseq r0, msp
    mrsne r0, psp
    ldr r1, [r0, #24]    /* PC at fault */
    ldr r2, [r0, #20]    /* LR at fault */
    ldr r3, =0xE000ED28  /* CFSR address */
    ldr r3, [r3]         /* CFSR value */
fault_loop:
    b fault_loop
.size Default_Handler, .-Default_Handler

/* ------------------------------------------------------------------ */
/*  _init / _fini stubs                                                 */
/*                                                                      */
/*  Required because we use -nostartfiles (no crti.o/crtn.o).          */
/*  __libc_init_array calls _init() before iterating .init_array.      */
/* ------------------------------------------------------------------ */
.section .text._init
.thumb_func
.global _init
.type _init, %function
_init:
    bx lr
.size _init, .-_init

.section .text._fini
.thumb_func
.global _fini
.type _fini, %function
_fini:
    bx lr
.size _fini, .-_fini
