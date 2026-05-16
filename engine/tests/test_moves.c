#include "draughts.h"

#include <stdio.h>

static int failures = 0;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures; \
        } \
    } while (0)

static int has_move(const DraughtsMove *moves, int move_count, int from, int to)
{
    int i;

    for (i = 0; i < move_count; ++i) {
        if (moves[i].path_len == 2
            && moves[i].path[0] == from
            && moves[i].path[1] == to
            && moves[i].capture_count == 0) {
            return 1;
        }
    }

    return 0;
}

static void test_initial_black_quiet_moves(void)
{
    DraughtsState state;
    DraughtsMove moves[16];
    int move_count = -1;

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 16, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 9);

    CHECK(has_move(moves, move_count, 16, 21));
    CHECK(has_move(moves, move_count, 17, 21));
    CHECK(has_move(moves, move_count, 17, 22));
    CHECK(has_move(moves, move_count, 18, 22));
    CHECK(has_move(moves, move_count, 18, 23));
    CHECK(has_move(moves, move_count, 19, 23));
    CHECK(has_move(moves, move_count, 19, 24));
    CHECK(has_move(moves, move_count, 20, 24));
    CHECK(has_move(moves, move_count, 20, 25));
}

static void test_initial_white_quiet_moves(void)
{
    DraughtsState state;
    DraughtsMove moves[16];
    int move_count = -1;

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 16, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 9);

    CHECK(has_move(moves, move_count, 31, 26));
    CHECK(has_move(moves, move_count, 31, 27));
    CHECK(has_move(moves, move_count, 32, 27));
    CHECK(has_move(moves, move_count, 32, 28));
    CHECK(has_move(moves, move_count, 33, 28));
    CHECK(has_move(moves, move_count, 33, 29));
    CHECK(has_move(moves, move_count, 34, 29));
    CHECK(has_move(moves, move_count, 34, 30));
    CHECK(has_move(moves, move_count, 35, 30));
}

static void test_blocked_and_side_filter(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 42, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 47, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 48, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 43, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 0);

    CHECK(draughts_set_piece(&state, 47, DRAUGHTS_EMPTY) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_move(moves, move_count, 42, 47));
    CHECK(!has_move(moves, move_count, 43, 38));
}

static void test_edges_and_empty_position(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 46, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 0);

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 0);
}

static void test_buffer_and_invalid_arguments(void)
{
    DraughtsState state;
    DraughtsMove moves[4];
    int move_count = -1;

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 4, &move_count) == DRAUGHTS_ERR_BUFFER_TOO_SMALL);
    CHECK(move_count == 9);
    CHECK(has_move(moves, 4, 16, 21));

    move_count = -1;
    CHECK(draughts_generate_legal_moves(&state, NULL, 0, &move_count) == DRAUGHTS_ERR_BUFFER_TOO_SMALL);
    CHECK(move_count == 9);

    CHECK(draughts_generate_legal_moves(NULL, moves, 4, &move_count) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_generate_legal_moves(&state, moves, 4, NULL) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_generate_legal_moves(&state, NULL, 4, &move_count) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_generate_legal_moves(&state, moves, -1, &move_count) == DRAUGHTS_ERR_INVALID_ARGUMENT);
}

int main(void)
{
    test_initial_black_quiet_moves();
    test_initial_white_quiet_moves();
    test_blocked_and_side_filter();
    test_edges_and_empty_position();
    test_buffer_and_invalid_arguments();

    if (failures != 0) {
        printf("%d test failure(s)\n", failures);
        return 1;
    }

    printf("All move tests passed\n");
    return 0;
}
