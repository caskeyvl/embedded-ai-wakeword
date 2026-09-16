/* syscalls.c — Minimal newlib platform stubs for bare-metal STM32F407.
 *
 * Base (heap allocator, init_newlib, POSIX stubs) copied unmodified from
 * ../../stm32f4_audio_examples/audio_in_to_wav/syscalls.c. ExecuTorch-
 * specific additions (byte-safe memcmp/memcpy/memset, snprintf/vsnprintf,
 * et_pal_emit_log_message stub) copied from ../../first_et_project/src/syscalls.c
 * -- required because this project runs ExecuTorch on-device, which
 * audio_in_to_wav does not. See README.md "Gotchas". */

#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <reent.h>

/* ── Heap allocator ──────────────────────────────────────────────────────── */
extern uint32_t _ebss;
static uint8_t *heap_end = NULL;

void *_sbrk(int incr) {
    if (heap_end == NULL)
        heap_end = (uint8_t *)&_ebss;
    uint8_t *prev = heap_end;
    heap_end += incr;
    return (void *)prev;
}

/* ── Newlib reentrant structure init ─────────────────────────────────────── */
/*
 * Called from Reset_Handler before __libc_init_array.
 * newlib-nano's snprintf uses _impure_ptr internally; this initializes it.
 * ExecuTorch calls snprintf() in Method::resolve_operator(), so this must
 * run before main() calls load_method().
 */
void init_newlib(void) {
    _REENT_INIT_PTR(_impure_ptr);
}

/* ── Byte-safe memcmp/memcpy/memset ──────────────────────────────────────── */
/*
 * newlib's word-wide implementations fault on unaligned accesses that
 * ExecuTorch's FlatBuffer parsing triggers.
 */
int memcmp(const void *s1, const void *s2, size_t n) {
    const unsigned char *p1 = (const unsigned char *)s1;
    const unsigned char *p2 = (const unsigned char *)s2;
    while (n--) {
        if (*p1 != *p2) return (int)*p1 - (int)*p2;
        p1++;
        p2++;
    }
    return 0;
}

void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *s, int c, size_t n) {
    unsigned char *p = (unsigned char *)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

/* ── snprintf / vsnprintf overrides ──────────────────────────────────────── */
/*
 * newlib's snprintf requires FILE* stream infrastructure that doesn't exist
 * on bare-metal. ExecuTorch uses snprintf() to build operator name strings
 * and error messages; hal.c's uart_printf() also uses this for UART output.
 * This is not a complete printf -- unsupported specifiers pass through
 * literally. Supports %f (fixed 4-decimal) for confidence-value formatting.
 */
int snprintf(char *buf, size_t size, const char *fmt, ...) {
    if (buf == NULL || size == 0) return 0;

    va_list args;
    va_start(args, fmt);

    size_t pos = 0;
    size_t remaining = size - 1;
    const char *f = fmt;

    while (*f && remaining > 0) {
        if (*f != '%') {
            buf[pos++] = *f++;
            remaining--;
            continue;
        }
        f++; /* skip '%' */

        int zero_pad = 0;
        if (*f == '0') { zero_pad = 1; f++; }

        int width = 0;
        while (*f >= '0' && *f <= '9') { width = width * 10 + (*f++ - '0'); }

        int is_long = 0;
        if (*f == 'l') { is_long = 1; f++; }
        if (*f == 'z') { f++; } /* size_t -- treat as unsigned long */

        char tmp[24];
        int tlen = 0;

        switch (*f) {
            case 's': {
                const char *s = va_arg(args, const char *);
                if (!s) s = "(null)";
                while (*s && remaining > 0) { buf[pos++] = *s++; remaining--; }
                break;
            }
            case 'c': {
                if (remaining > 0) { buf[pos++] = (char)va_arg(args, int); remaining--; }
                break;
            }
            case 'd': {
                long val = is_long ? va_arg(args, long) : (long)va_arg(args, int);
                if (val < 0 && remaining > 0) { buf[pos++] = '-'; remaining--; val = -val; }
                do { tmp[tlen++] = '0' + (int)(val % 10); val /= 10; } while (val && tlen < 23);
                if (zero_pad) while (tlen < width && tlen < 23) tmp[tlen++] = '0';
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'u': {
                unsigned long val = is_long ? va_arg(args, unsigned long)
                                            : (unsigned long)va_arg(args, unsigned int);
                do { tmp[tlen++] = '0' + (int)(val % 10); val /= 10; } while (val && tlen < 23);
                if (zero_pad) while (tlen < width && tlen < 23) tmp[tlen++] = '0';
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'x': case 'X': {
                unsigned long val = is_long ? va_arg(args, unsigned long)
                                            : (unsigned long)va_arg(args, unsigned int);
                const char *hex = (*f == 'x') ? "0123456789abcdef" : "0123456789ABCDEF";
                do { tmp[tlen++] = hex[val & 0xF]; val >>= 4; } while (val && tlen < 23);
                if (zero_pad) while (tlen < width && tlen < 23) tmp[tlen++] = '0';
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'p': {
                unsigned long val = (unsigned long)va_arg(args, void *);
                buf[pos++] = '0'; if (remaining) { buf[pos++] = 'x'; remaining--; }
                do { tmp[tlen++] = "0123456789abcdef"[val & 0xF]; val >>= 4; } while (val && tlen < 23);
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'f': {
                /* Fixed 4-decimal formatting -- sufficient for this
                 * project's confidence values (all in [0,1]). */
                double val = va_arg(args, double);
                if (val < 0 && remaining > 0) { buf[pos++] = '-'; remaining--; val = -val; }
                long ip = (long)val;
                long fp = (long)((val - (double)ip) * 10000.0 + 0.5);
                if (fp >= 10000) { ip++; fp -= 10000; }
                do { tmp[tlen++] = '0' + (int)(ip % 10); ip /= 10; } while (ip && tlen < 23);
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                if (remaining > 0) { buf[pos++] = '.'; remaining--; }
                char frac[4] = {
                    (char)('0' + (fp / 1000) % 10), (char)('0' + (fp / 100) % 10),
                    (char)('0' + (fp / 10) % 10),   (char)('0' + fp % 10)
                };
                for (int i = 0; i < 4 && remaining > 0; i++) { buf[pos++] = frac[i]; remaining--; }
                break;
            }
            case '%':
                if (remaining > 0) { buf[pos++] = '%'; remaining--; }
                break;
            default:
                if (remaining > 0) { buf[pos++] = '%'; remaining--; }
                if (remaining > 0) { buf[pos++] = *f;  remaining--; }
                break;
        }
        f++;
    }

    buf[pos] = '\0';
    va_end(args);
    return (int)pos;
}

int vsnprintf(char *buf, size_t size, const char *fmt, va_list ap) {
    if (buf == NULL || size == 0) return 0;
    size_t pos = 0, remaining = size - 1;
    const char *f = fmt;
    while (*f && remaining > 0) {
        if (*f == '%' && *(f + 1) == 's') {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            while (*s && remaining > 0) { buf[pos++] = *s++; remaining--; }
            f += 2;
        } else if (*f == '%' && *(f + 1) == 'd') {
            int val = va_arg(ap, int);
            char tmp[12]; int tlen = 0;
            if (val < 0 && remaining > 0) { buf[pos++] = '-'; remaining--; val = -val; }
            do { tmp[tlen++] = '0' + (val % 10); val /= 10; } while (val);
            while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
            f += 2;
        } else if (*f == '%' && *(f + 1) == 'u') {
            /* Unlike %d/%f, this specifier was missing entirely until it was
             * caught by lab1_wakeword_soln/README-provided-discussion.md's
             * Step 8 dry run: an unrecognized specifier falls through to the
             * `else` branch below and is copied out literally (e.g. a stray
             * "%u" in the UART output), rather than erroring -- easy to
             * mistake for a logic bug in your own code instead of a missing
             * format specifier here. */
            unsigned int val = va_arg(ap, unsigned int);
            char tmp[12]; int tlen = 0;
            do { tmp[tlen++] = '0' + (int)(val % 10); val /= 10; } while (val);
            while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
            f += 2;
        } else if (*f == '%' && *(f + 1) == 'f') {
            /* Fixed 4-decimal formatting -- see snprintf's %f for rationale. */
            double val = va_arg(ap, double);
            if (val < 0 && remaining > 0) { buf[pos++] = '-'; remaining--; val = -val; }
            long ip = (long)val;
            long fp = (long)((val - (double)ip) * 10000.0 + 0.5);
            if (fp >= 10000) { ip++; fp -= 10000; }
            char tmp[12]; int tlen = 0;
            do { tmp[tlen++] = '0' + (int)(ip % 10); ip /= 10; } while (ip);
            while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
            if (remaining > 0) { buf[pos++] = '.'; remaining--; }
            char frac[4] = {
                (char)('0' + (fp / 1000) % 10), (char)('0' + (fp / 100) % 10),
                (char)('0' + (fp / 10) % 10),   (char)('0' + fp % 10)
            };
            for (int i = 0; i < 4 && remaining > 0; i++) { buf[pos++] = frac[i]; remaining--; }
            f += 2;
        } else {
            buf[pos++] = *f++; remaining--;
        }
    }
    buf[pos] = '\0';
    return (int)pos;
}

/* ── ExecuTorch PAL logging stub ──────────────────────────────────────────── */
/*
 * The default et_pal_emit_log_message() uses fprintf(stderr), which requires
 * FILE* infrastructure that crashes on bare-metal. -DET_LOG_ENABLED=0 doesn't
 * suppress every call site, so a no-op override is still required.
 */
void et_pal_emit_log_message(
        long timestamp, int level,
        const char *filename, const char *function, unsigned long line,
        const char *message, unsigned long length) {
    (void)timestamp; (void)level;    (void)filename;
    (void)function;  (void)line;     (void)message; (void)length;
}

/* ── POSIX syscall stubs ─────────────────────────────────────────────────── */
int _write(int fd, char *buf, int len)      { (void)fd; (void)buf; return len; }
int _read(int fd, char *buf, int len)       { (void)fd; (void)buf; (void)len; return 0; }
int _close(int fd)                          { (void)fd; return -1; }
int _isatty(int fd)                         { (void)fd; return 1; }
int _lseek(int fd, int off, int w)          { (void)fd; (void)off; (void)w; return 0; }
void _exit(int s)                           { (void)s; while (1); }
int _kill(int p, int s)                     { (void)p; (void)s; return -1; }
int _getpid(void)                           { return 1; }

int _fstat(int fd, struct stat *st) {
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}

int _gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; }
    return 0;
}
