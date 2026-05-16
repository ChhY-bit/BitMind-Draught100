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

static int has_capture_move(
    const DraughtsMove *moves,
    int move_count,
    const int *path,
    int path_len,
    const int *captures,
    int capture_count
)
{
    int i;

    for (i = 0; i < move_count; ++i) {
        int j;
        int matches = 1;

        if (moves[i].path_len != path_len || moves[i].capture_count != capture_count) {
            continue;
        }

        for (j = 0; j < path_len; ++j) {
            if (moves[i].path[j] != path[j]) {
                matches = 0;
            }
        }
        for (j = 0; j < capture_count; ++j) {
            if (moves[i].captures[j] != captures[j]) {
                matches = 0;
            }
        }

        if (matches) {
            return 1;
        }
    }

    return 0;
}

static int has_any_capture_move(const DraughtsMove *moves, int move_count)
{
    int i;

    for (i = 0; i < move_count; ++i) {
        if (moves[i].capture_count > 0) {
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

static void test_capture_is_mandatory(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int path[] = { 21, 32 };
    const int captures[] = { 27 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 24, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, path, 2, captures, 1));
    CHECK(!has_move(moves, move_count, 24, 29));
}

static void test_men_can_capture_backward(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int black_path[] = { 32, 21 };
    const int black_captures[] = { 27 };
    const int white_path[] = { 21, 32 };
    const int white_captures[] = { 27 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 32, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, black_path, 2, black_captures, 1));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, white_path, 2, white_captures, 1));
}

static void test_multi_jump_path_and_captures(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int path[] = { 21, 32, 43 };
    const int captures[] = { 27, 38 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 38, DRAUGHTS_WHITE_KING) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, path, 3, captures, 2));
}

static void test_max_capture_rule_keeps_only_longest(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int long_path[] = { 22, 31, 42 };
    const int long_captures[] = { 27, 37 };
    const int short_path[] = { 22, 33 };
    const int short_captures[] = { 28 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 22, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, long_path, 3, long_captures, 2));
    CHECK(!has_capture_move(moves, move_count, short_path, 2, short_captures, 1));
}

static void test_equal_max_capture_routes_are_all_kept(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int left_path[] = { 23, 32, 41 };
    const int left_captures[] = { 28, 37 };
    const int right_path[] = { 23, 34, 43 };
    const int right_captures[] = { 29, 39 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 23, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 29, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 39, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 2);
    CHECK(has_capture_move(moves, move_count, left_path, 3, left_captures, 2));
    CHECK(has_capture_move(moves, move_count, right_path, 3, right_captures, 2));
}

static void test_capture_blocked_by_own_piece_or_nonempty_landing(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 3);
    CHECK(has_move(moves, move_count, 21, 26));
    CHECK(!has_any_capture_move(moves, move_count));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 32, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_move(moves, move_count, 21, 26));
}

static void test_captured_piece_remains_barrier_during_multi_jump(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int path[] = { 23, 32 };
    const int captures[] = { 28 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 23, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, path, 2, captures, 1));
}

static void test_capture_buffer_too_small_reports_total_best_moves(void)
{
    DraughtsState state;
    DraughtsMove moves[1];
    int move_count = -1;
    const int left_path[] = { 23, 32, 41 };
    const int left_captures[] = { 28, 37 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 23, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 29, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 39, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 1, &move_count) == DRAUGHTS_ERR_BUFFER_TOO_SMALL);
    CHECK(move_count == 2);
    CHECK(has_capture_move(moves, 1, left_path, 3, left_captures, 2));

    move_count = -1;
    CHECK(draughts_generate_legal_moves(&state, NULL, 0, &move_count) == DRAUGHTS_ERR_BUFFER_TOO_SMALL);
    CHECK(move_count == 2);
}

int main(void)
{
    test_initial_black_quiet_moves();
    test_initial_white_quiet_moves();
    test_blocked_and_side_filter();
    test_edges_and_empty_position();
    test_buffer_and_invalid_arguments();
    test_capture_is_mandatory();
    test_men_can_capture_backward();
    test_multi_jump_path_and_captures();
    test_max_capture_rule_keeps_only_longest();
    test_equal_max_capture_routes_are_all_kept();
    test_capture_blocked_by_own_piece_or_nonempty_landing();
    test_captured_piece_remains_barrier_during_multi_jump();
    test_capture_buffer_too_small_reports_total_best_moves();

    if (failures != 0) {
        printf("%d test failure(s)\n", failures);
        return 1;
    }

    printf("All move tests passed\n");
    return 0;
}
