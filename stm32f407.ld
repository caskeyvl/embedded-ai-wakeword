/*
 * stm32f407.ld — Linker script for STM32F407VG (Cortex-M4)
 *
 * Copied from ../stm32f4_audio_examples/audio_in_to_wav/stm32f407.ld
 * unchanged. That project's CCMRAM region (64KB @ 0x10000000) holds
 * pdm_mic.c's PCM ring buffer; this project additionally places
 * spectrogram.c's FFT/Hann-window scratch buffers there (see
 * src/spectrogram.c's CCMRAM attribute and README.md "Memory budget") --
 * no linker-script change was needed for that, just more things placed
 * in the same .ccmram section via __attribute__((section(".ccmram"))).
 *
 * Memory map:
 *   FLASH:  1 MB  at 0x08000000  (code, rodata, init values, embedded .pte)
 *   RAM:  128 KB  at 0x20000000  (stack, .data, .bss, heap, DMA buffers,
 *                                 ExecuTorch method_pool/planned_pool)
 *   CCMRAM: 64 KB at 0x10000000  (CPU-only fast RAM, no DMA: pcm_ring +
 *                                 spectrogram scratch)
 */

MEMORY
{
  FLASH  (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM    (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
  CCMRAM (rwx) : ORIGIN = 0x10000000, LENGTH = 64K
}

ENTRY(Reset_Handler)

SECTIONS
{
  /* Vector table must be first in flash at 0x08000000 */
  .isr_vector :
  {
    KEEP(*(.isr_vector))
  } > FLASH

  /* Code and read-only data */
  .text :
  {
    *(.text*)
    *(.rodata*)
  } > FLASH

  /*
   * ARM exception unwinding tables — required by libgcc even with
   * -fno-exceptions; omitting these causes undefined __exidx_start/end.
   */
  .ARM.extab :
  {
    *(.ARM.extab*)
    *(.gnu.linkonce.armextab.*)
  } > FLASH

  .ARM.exidx :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    *(.gnu.linkonce.armexidx.*)
    __exidx_end = .;
  } > FLASH

  /* C constructor/destructor tables — iterated by __libc_init_array */
  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > FLASH

  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(.fini_array*))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > FLASH

  /*
   * Initialized data — stored in FLASH (LMA), copied to RAM (VMA) by
   * Reset_Handler before main(). .got MUST come before .data* so the
   * Global Offset Table is covered by the startup copy loop.
   */
  .data :
  {
    _sdata = .;
    *(.got)
    *(.got.plt)
    *(.data*)
    _edata = .;
  } > RAM AT > FLASH

  /* Flash address of .data load image — used by Reset_Handler copy loop */
  _sidata = LOADADDR(.data);

  /* Zero-initialized data — zeroed by Reset_Handler before main() */
  .bss :
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM

  /* Stack grows downward from top of RAM */
  _estack = ORIGIN(RAM) + LENGTH(RAM);

  /*
   * CCM RAM — 64KB at 0x10000000, CPU-only (no DMA access).
   * Variables placed here with __attribute__((section(".ccmram"))).
   */
  .ccmram (NOLOAD) :
  {
    _sccmram = .;
    *(.ccmram)
    *(.ccmram*)
    _eccmram = .;
  } > CCMRAM
}
