#include "draughts_interface.h"

static int refresh_env(DraughtsEnvHandle *env)
{
    DraughtsGameResult game_result;
    int result;
    int move_count = 0;

    if (env == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    result = draughts_get_game_result(&env->state, &game_result);
    if (result != DRAUGHTS_OK) {
        return result;
    }
    env->game_result = game_result;

    if (env->game_result != DRAUGHTS_GAME_ONGOING) {
        env->legal_move_count = 0;
        return DRAUGHTS_OK;
    }

    result = draughts_generate_legal_moves(
        &env->state,
        env->legal_moves,
        DRAUGHTS_INTERFACE_MAX_LEGAL_MOVES,
        &move_count
    );
    if (result != DRAUGHTS_OK) {
        return result;
    }

    env->legal_move_count = move_count;
    return DRAUGHTS_OK;
}

static int fill_observation(const DraughtsEnvHandle *env, DraughtsObservation *observation)
{
    int i;

    if (env == NULL || observation == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    for (i = 0; i < DRAUGHTS_SQUARE_COUNT; ++i) {
        observation->squares[i] = env->state.squares[i];
    }
    observation->side_to_move = env->state.side_to_move;
    observation->game_result = env->game_result;
    observation->legal_move_count = env->legal_move_count;

    return DRAUGHTS_OK;
}

static int reward_for_result(DraughtsSide acting_side, int game_result)
{
    if (game_result == DRAUGHTS_GAME_DRAW || game_result == DRAUGHTS_GAME_ONGOING) {
        return 0;
    }
    if (game_result == DRAUGHTS_GAME_BLACK_WIN) {
        return acting_side == DRAUGHTS_BLACK ? 1 : -1;
    }
    if (game_result == DRAUGHTS_GAME_WHITE_WIN) {
        return acting_side == DRAUGHTS_WHITE ? 1 : -1;
    }

    return 0;
}

int draughts_env_reset(DraughtsEnvHandle *env, DraughtsObservation *observation)
{
    int result;

    if (env == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    result = draughts_init(&env->state);
    if (result != DRAUGHTS_OK) {
        return result;
    }

    result = refresh_env(env);
    if (result != DRAUGHTS_OK) {
        return result;
    }

    if (observation != NULL) {
        return fill_observation(env, observation);
    }

    return DRAUGHTS_OK;
}

int draughts_env_get_observation(const DraughtsEnvHandle *env, DraughtsObservation *observation)
{
    return fill_observation(env, observation);
}

int draughts_env_get_legal_move_count(const DraughtsEnvHandle *env, int *move_count)
{
    if (env == NULL || move_count == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    *move_count = env->legal_move_count;
    return DRAUGHTS_OK;
}

int draughts_env_get_legal_move(
    const DraughtsEnvHandle *env,
    int action_id,
    DraughtsMove *move
)
{
    if (env == NULL || move == NULL) {
        return DRAUGHTS_ERR_NULL;
    }
    if (action_id < 0 || action_id >= env->legal_move_count) {
        return DRAUGHTS_ERR_INVALID_ARGUMENT;
    }

    *move = env->legal_moves[action_id];
    return DRAUGHTS_OK;
}

int draughts_env_step(
    DraughtsEnvHandle *env,
    int action_id,
    DraughtsObservation *observation,
    DraughtsStepResult *step_result
)
{
    DraughtsMove move;
    DraughtsSide acting_side;
    int result;

    if (env == NULL || step_result == NULL) {
        return DRAUGHTS_ERR_NULL;
    }
    if (env->game_result != DRAUGHTS_GAME_ONGOING) {
        return DRAUGHTS_ERR_INVALID_MOVE;
    }
    if (action_id < 0 || action_id >= env->legal_move_count) {
        return DRAUGHTS_ERR_INVALID_ARGUMENT;
    }

    result = draughts_get_side_to_move(&env->state, &acting_side);
    if (result != DRAUGHTS_OK) {
        return result;
    }

    move = env->legal_moves[action_id];
    result = draughts_apply_move(&env->state, &move);
    if (result != DRAUGHTS_OK) {
        return result;
    }

    result = refresh_env(env);
    if (result != DRAUGHTS_OK) {
        return result;
    }

    step_result->done = env->game_result != DRAUGHTS_GAME_ONGOING;
    step_result->game_result = env->game_result;
    step_result->legal_move_count = env->legal_move_count;
    step_result->reward = reward_for_result(acting_side, env->game_result);

    if (observation != NULL) {
        return fill_observation(env, observation);
    }

    return DRAUGHTS_OK;
}
