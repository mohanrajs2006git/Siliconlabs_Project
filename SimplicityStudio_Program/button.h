#ifndef BUTTON_H
#define BUTTON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void button_init(void);
void button_task(void);

extern volatile bool button_active;

#ifdef __cplusplus
}
#endif

#endif // BUTTON_H
