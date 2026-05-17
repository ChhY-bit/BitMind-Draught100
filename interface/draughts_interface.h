#ifndef DRAUGHTS_INTERFACE_H
#define DRAUGHTS_INTERFACE_H

#include "draughts.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES 512

typedef struct {
    DraughtsState state;
    DraughtsMove legal_moves[DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES];
    int legal_move_count;
    int game_result;
} DraughtsEnvHandle;

typedef struct {
    unsigned char squares[DRAUGHTS_SQUARE_COUNT];
    unsigned char side_to_move;
    int game_result;
    int legal_move_count;
} DraughtsObservation;

typedef struct {
    int reward;
    int done;
    int game_result;
    int legal_move_count;
} DraughtsStepResult;

/*
 * Input: env points to writable environment storage, observation may be NULL.
 * Output: resets to the standard initial position and refreshes cached legal moves.
 */
int draughts_env_reset(DraughtsEnvHandle *env, DraughtsObservation *observation);

/*
 * Input: env points to a current environment, observation points to writable storage.
 * Output: copies a compact observation from the C environment.
 */
int draughts_env_get_observation(const DraughtsEnvHandle *env, DraughtsObservation *observation);

/*
 * Input: env points to a current environment, move_count points to writable storage.
 * Output: writes the cached legal action count for the current side.
 */
int draughts_env_get_legal_move_count(const DraughtsEnvHandle *env, int *move_count);

/*
 * Input: env points to a current environment, action_id is a zero-based legal action index.
 * Output: copies the cached C move for that action id.
 */
int draughts_env_get_legal_move(
    const DraughtsEnvHandle *env,
    int action_id,
    DraughtsMove *move
);

/*
 * Input: env points to a current environment, action_id is a zero-based legal action index.
 * Output: applies the selected action, refreshes legal moves and writes result data.
 */
int draughts_env_step(
    DraughtsEnvHandle *env,
    int action_id,
    DraughtsObservation *observation,
    DraughtsStepResult *step_result
);

#ifdef __cplusplus
}
#endif

#endif
