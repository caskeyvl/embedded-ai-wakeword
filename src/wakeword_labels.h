#ifndef WAKEWORD_LABELS_H
#define WAKEWORD_LABELS_H

/* Class label order — MUST match models/train_wakeword.py's label_of()
 * mapping exactly (index i here == class index i in the model's output
 * logits). Target commands (yes, no), followed by silence and unknown.
 * See ../README-training-parameters.md. */

#define WAKEWORD_NUM_CLASSES 4

static const char *const WAKEWORD_CLASS_NAMES[WAKEWORD_NUM_CLASSES] = {
    "yes", "no", "silence", "unknown"
};

#endif /* WAKEWORD_LABELS_H */
