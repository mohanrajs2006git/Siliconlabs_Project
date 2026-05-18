#ifndef SPEAKER_H
#define SPEAKER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define TRACK_RESTROOM    1
#define TRACK_PAIN        2
#define TRACK_EMERGENCY   3
#define TRACK_HELP        4
#define TRACK_FOOD        5
#define TRACK_WATER       6

void speaker_init(void);
void speaker_play(uint8_t track);

#ifdef __cplusplus
}
#endif

#endif // SPEAKER_H
