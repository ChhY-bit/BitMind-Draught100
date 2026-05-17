#include "draughts.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RANDOM_MOVE_BUFFER 512

typedef struct {
    unsigned long games;
    unsigned long black_wins;
    unsigned long white_wins;
    unsigned long draws;
    unsigned long max_plies_reached;
    unsigned long total_plies;
} RandomStats;

static unsigned int next_random(unsigned int *seed)
{
    *seed = (*seed * 1103515245u) + 12345u;
    return (*seed / 65536u) % 32768u;
}

static int parse_positive_arg(const char *text, unsigned long *value)
{
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);

    if (text == NULL || *text == '\0' || end == NULL || *end != '\0' || parsed == 0) {
        return 0;
    }

    *value = parsed;
    return 1;
}

static int validate_state(const DraughtsState *state)
{
    int square;
    DraughtsSide side;

    if (draughts_get_side_to_move(state, &side) != DRAUGHTS_OK) {
        return 0;
    }

    for (square = 1; square <= DRAUGHTS_SQUARE_COUNT; ++square) {
        DraughtsPiece piece;

        if (draughts_get_piece(state, square, &piece) != DRAUGHTS_OK) {
            return 0;
        }
        if (piece < DRAUGHTS_EMPTY || piece > DRAUGHTS_WHITE_KING) {
            return 0;
        }
    }

    return 1;
}

static void record_result(RandomStats *stats, DraughtsGameResult result, int plies)
{
    ++stats->games;
    stats->total_plies += (unsigned long)plies;

    if (result == DRAUGHTS_GAME_BLACK_WIN) {
        ++stats->black_wins;
    } else if (result == DRAUGHTS_GAME_WHITE_WIN) {
        ++stats->white_wins;
    } else if (result == DRAUGHTS_GAME_DRAW) {
        ++stats->draws;
    } else {
        ++stats->max_plies_reached;
    }
}

static void print_progress(
    const RandomStats *stats,
    unsigned long requested_games,
    clock_t start_time
)
{
    double elapsed = (double)(clock() - start_time) / (double)CLOCKS_PER_SEC;
    double avg_plies = 0.0;

    if (stats->games > 0) {
        avg_plies = (double)stats->total_plies / (double)stats->games;
    }

    printf(
        "random-play: %lu/%lu games ok, black=%lu white=%lu draw=%lu max-plies=%lu, "
        "avg-plies=%.2f, elapsed=%.3fs\n",
        stats->games,
        requested_games,
        stats->black_wins,
        stats->white_wins,
        stats->draws,
        stats->max_plies_reached,
        avg_plies,
        elapsed
    );
}

static void print_move(const DraughtsMove *move)
{
    int i;

    printf("path:");
    for (i = 0; i < move->path_len; ++i) {
        printf(" %d", move->path[i]);
    }
    printf(" captures:");
    for (i = 0; i < move->capture_count; ++i) {
        printf(" %d", move->captures[i]);
    }
    printf("\n");
}

static int play_one_game(
    unsigned int *seed,
    unsigned long max_plies,
    RandomStats *stats,
    unsigned long game_index
)
{
    DraughtsState state;
    DraughtsGameResult result;
    unsigned long ply;

    if (draughts_init(&state) != DRAUGHTS_OK) {
        printf("random-play error: failed to initialize game %lu\n", game_index);
        return 0;
    }

    for (ply = 0; ply < max_plies; ++ply) {
        DraughtsMove moves[RANDOM_MOVE_BUFFER];
        DraughtsMove chosen;
        int move_count = 0;
        int selected;
        int is_legal = 0;
        int status;

        if (!validate_state(&state)) {
            printf("random-play error: invalid state before game %lu ply %lu\n", game_index, ply);
            return 0;
        }

        status = draughts_get_game_result(&state, &result);
        if (status != DRAUGHTS_OK) {
            printf("random-play error: result failed in game %lu ply %lu: %d\n", game_index, ply, status);
            return 0;
        }
        if (result != DRAUGHTS_GAME_ONGOING) {
            record_result(stats, result, (int)ply);
            return 1;
        }

        status = draughts_generate_legal_moves(&state, moves, RANDOM_MOVE_BUFFER, &move_count);
        if (status != DRAUGHTS_OK) {
            printf("random-play error: move generation failed in game %lu ply %lu: %d count=%d\n",
                game_index,
                ply,
                status,
                move_count
            );
            return 0;
        }
        if (move_count <= 0) {
            printf("random-play error: ongoing game %lu ply %lu has no legal moves\n", game_index, ply);
            return 0;
        }

        selected = (int)(next_random(seed) % (unsigned int)move_count);
        chosen = moves[selected];
        status = draughts_is_legal_move(&state, &chosen, &is_legal);
        if (status != DRAUGHTS_OK || !is_legal) {
            printf("random-play error: selected illegal move in game %lu ply %lu: status=%d legal=%d\n",
                game_index,
                ply,
                status,
                is_legal
            );
            return 0;
        }

        status = draughts_apply_move(&state, &chosen);
        if (status != DRAUGHTS_OK) {
            printf("random-play error: apply failed in game %lu ply %lu: %d\n", game_index, ply, status);
            print_move(&chosen);
            return 0;
        }
    }

    record_result(stats, DRAUGHTS_GAME_ONGOING, (int)max_plies);
    return 1;
}

int main(int argc, char **argv)
{
    unsigned long games = 1000;
    unsigned long max_plies = 300;
    unsigned long log_interval = 200;
    unsigned long seed_arg = 1;
    unsigned int seed;
    unsigned long game;
    RandomStats stats = { 0, 0, 0, 0, 0, 0 };
    clock_t start_time = clock();

    if (argc > 1 && !parse_positive_arg(argv[1], &games)) {
        printf("Usage: %s [games] [max_plies] [log_interval] [seed]\n", argv[0]);
        return 2;
    }
    if (argc > 2 && !parse_positive_arg(argv[2], &max_plies)) {
        printf("Usage: %s [games] [max_plies] [log_interval] [seed]\n", argv[0]);
        return 2;
    }
    if (argc > 3 && !parse_positive_arg(argv[3], &log_interval)) {
        printf("Usage: %s [games] [max_plies] [log_interval] [seed]\n", argv[0]);
        return 2;
    }
    if (argc > 4 && !parse_positive_arg(argv[4], &seed_arg)) {
        printf("Usage: %s [games] [max_plies] [log_interval] [seed]\n", argv[0]);
        return 2;
    }
    if (argc > 5) {
        printf("Usage: %s [games] [max_plies] [log_interval] [seed]\n", argv[0]);
        return 2;
    }

    seed = (unsigned int)seed_arg;
    printf(
        "random-play: starting games=%lu max_plies=%lu log_interval=%lu seed=%lu\n",
        games,
        max_plies,
        log_interval,
        seed_arg
    );

    for (game = 1; game <= games; ++game) {
        if (!play_one_game(&seed, max_plies, &stats, game)) {
            print_progress(&stats, games, start_time);
            return 1;
        }

        if (game % log_interval == 0 || game == games) {
            print_progress(&stats, games, start_time);
        }
    }

    return 0;
}
