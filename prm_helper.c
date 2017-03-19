#include "xcorr2.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

struct prm_handler prm_open(const char *fname) {
    FILE *fin = fopen(fname, "r");
    char buf[256];
    struct prm_handler handler;

    if (fin == NULL) {
        perror("prm_open");
        exit(EXIT_FAILURE);
    }

    handler.entry = g_hash_table_new_full(g_str_hash, g_str_equal, free, free);

    bool broken = false;
    while (fgets(buf, sizeof(buf), fin) != NULL) {
        char *vfield = NULL, *kfield, *p;

        for (p=buf; *p; p++)
            if (*p == '=')
                vfield = p;

        if (p == buf || *(p-1) != '\n') {
            broken = true;
            continue;
        } else if (broken) {
            broken = false;
            continue;
        }
        
        if (vfield == NULL)
            continue;

        for (*(--p) = 0; isblank(*p); p--)  // rstrip on vfield
            *p = 0;

        for (kfield = buf; isblank(*kfield); kfield++)  // lstrip on kfield
            continue;
        for (p = vfield-1; isblank(*p); p--)  // rstrip on kfield
            *p = 0;

        for (*(vfield++)=0; isblank(*vfield); vfield++)  // lstrip on vfield
            continue;

        char *key = strdup(kfield);
        char *value = strdup(vfield);
        g_hash_table_insert(handler.entry, key, value);
    }

    fclose(fin);

    return handler;
}

void prm_close(struct prm_handler *handler) {
    g_hash_table_destroy(handler->entry);
    handler->entry = NULL;
}

const char *prm_get_str(const struct prm_handler handler, const char *key) {
    return g_hash_table_lookup(handler.entry, key);
}

int prm_get_int(const struct prm_handler handler, const char *key) {
    char *raw_value = g_hash_table_lookup(handler.entry, key);
    return atoi(raw_value);
}

double prm_get_f64(const struct prm_handler handler, const char *key) {
    char *raw_value = g_hash_table_lookup(handler.entry, key);
    return atof(raw_value);
}
