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

static DraughtsMove make_move(const int *path, int path_len, const int *captures, int capture_count)
{
    DraughtsMove move;
    int i;

    move.path_len = (unsigned char)path_len;
    move.capture_count = (unsigned char)capture_count;
    for (i = 0; i < path_len; ++i) {
        move.path[i] = (unsigned char)path[i];
    }
    for (i = 0; i < capture_count; ++i) {
        move.captures[i] = (unsigned char)captures[i];
    }

    return move;
}

static int piece_at(const DraughtsState *state, int square, DraughtsPiece expected)
{
    DraughtsPiece piece;

    if (draughts_get_piece(state, square, &piece) != DRAUGHTS_OK) {
        return 0;
    }

    return piece == expected;
}

static int side_to_move_is(const DraughtsState *state, DraughtsSide expected)
{
    DraughtsSide side;

    if (draughts_get_side_to_move(state, &side) != DRAUGHTS_OK) {
        return 0;
    }

    return side == expected;
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

static void test_king_quiet_moves_from_center(void)
{
    DraughtsState state;
    DraughtsMove moves[20];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 20, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 17);

    CHECK(has_move(moves, move_count, 28, 22));
    CHECK(has_move(moves, move_count, 28, 17));
    CHECK(has_move(moves, move_count, 28, 11));
    CHECK(has_move(moves, move_count, 28, 6));
    CHECK(has_move(moves, move_count, 28, 23));
    CHECK(has_move(moves, move_count, 28, 19));
    CHECK(has_move(moves, move_count, 28, 14));
    CHECK(has_move(moves, move_count, 28, 10));
    CHECK(has_move(moves, move_count, 28, 5));
    CHECK(has_move(moves, move_count, 28, 32));
    CHECK(has_move(moves, move_count, 28, 37));
    CHECK(has_move(moves, move_count, 28, 41));
    CHECK(has_move(moves, move_count, 28, 46));
    CHECK(has_move(moves, move_count, 28, 33));
    CHECK(has_move(moves, move_count, 28, 39));
    CHECK(has_move(moves, move_count, 28, 44));
    CHECK(has_move(moves, move_count, 28, 50));
}

static void test_white_king_quiet_moves_and_side_filter(void)
{
    DraughtsState state;
    DraughtsMove moves[20];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 20, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 17);
    CHECK(has_move(moves, move_count, 28, 23));
    CHECK(has_move(moves, move_count, 28, 50));
    CHECK(!has_move(moves, move_count, 27, 21));
}

static void test_king_quiet_moves_stop_at_any_piece(void)
{
    DraughtsState state;
    DraughtsMove moves[40];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 39, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 40, &move_count) == DRAUGHTS_OK);
    CHECK(has_move(moves, move_count, 28, 23));
    CHECK(!has_move(moves, move_count, 28, 19));
    CHECK(!has_move(moves, move_count, 28, 14));
    CHECK(!has_move(moves, move_count, 28, 10));
    CHECK(has_move(moves, move_count, 28, 33));
    CHECK(!has_move(moves, move_count, 28, 39));
    CHECK(!has_move(moves, move_count, 28, 44));
    CHECK(!has_move(moves, move_count, 28, 50));
}

static void test_king_single_capture_has_all_landing_squares(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int path_14[] = { 28, 14 };
    const int path_10[] = { 28, 10 };
    const int path_5[] = { 28, 5 };
    const int captures[] = { 19 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 3);
    CHECK(has_capture_move(moves, move_count, path_14, 2, captures, 1));
    CHECK(has_capture_move(moves, move_count, path_10, 2, captures, 1));
    CHECK(has_capture_move(moves, move_count, path_5, 2, captures, 1));
    CHECK(!has_move(moves, move_count, 28, 23));
}

static void test_king_multi_capture_keeps_longest_routes(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int path[] = { 28, 14, 25 };
    const int captures[] = { 19, 20 };
    const int short_path[] = { 28, 10 };
    const int short_captures[] = { 19 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 20, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, path, 3, captures, 2));
    CHECK(!has_capture_move(moves, move_count, short_path, 2, short_captures, 1));
}

static void test_king_captured_piece_barrier_blocks_deeper_target(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int path_14[] = { 28, 14 };
    const int path_10[] = { 28, 10 };
    const int path_5[] = { 28, 5 };
    const int path_41[] = { 28, 41 };
    const int path_46[] = { 28, 46 };
    const int captures_19[] = { 19 };
    const int captures_37[] = { 37 };
    int i;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 5);
    CHECK(has_capture_move(moves, move_count, path_14, 2, captures_19, 1));
    CHECK(has_capture_move(moves, move_count, path_10, 2, captures_19, 1));
    CHECK(has_capture_move(moves, move_count, path_5, 2, captures_19, 1));
    CHECK(has_capture_move(moves, move_count, path_41, 2, captures_37, 1));
    CHECK(has_capture_move(moves, move_count, path_46, 2, captures_37, 1));
    for (i = 0; i < move_count; ++i) {
        CHECK(moves[i].capture_count == 1);
    }
}

static void test_king_capture_blockers(void)
{
    DraughtsState state;
    DraughtsMove moves[40];
    int move_count = -1;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 23, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 10, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 40, &move_count) == DRAUGHTS_OK);
    CHECK(!has_any_capture_move(moves, move_count));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 14, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_generate_legal_moves(&state, moves, 40, &move_count) == DRAUGHTS_OK);
    CHECK(!has_any_capture_move(moves, move_count));
}

static void test_king_and_man_captures_use_same_max_rule(void)
{
    DraughtsState state;
    DraughtsMove moves[8];
    int move_count = -1;
    const int man_path[] = { 22, 31, 42 };
    const int man_captures[] = { 27, 37 };
    const int king_path[] = { 28, 14 };
    const int king_captures[] = { 19 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 22, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 8, &move_count) == DRAUGHTS_OK);
    CHECK(move_count == 1);
    CHECK(has_capture_move(moves, move_count, man_path, 3, man_captures, 2));
    CHECK(!has_capture_move(moves, move_count, king_path, 2, king_captures, 1));
}

static void test_king_capture_buffer_too_small_reports_total_landings(void)
{
    DraughtsState state;
    DraughtsMove moves[1];
    int move_count = -1;
    const int path[] = { 28, 14 };
    const int captures[] = { 19 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_generate_legal_moves(&state, moves, 1, &move_count) == DRAUGHTS_ERR_BUFFER_TOO_SMALL);
    CHECK(move_count == 3);
    CHECK(has_capture_move(moves, 1, path, 2, captures, 1));

    move_count = -1;
    CHECK(draughts_generate_legal_moves(&state, NULL, 0, &move_count) == DRAUGHTS_ERR_BUFFER_TOO_SMALL);
    CHECK(move_count == 3);
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

static void test_is_legal_move_accepts_generated_and_rejects_nonlegal(void)
{
    DraughtsState state;
    DraughtsMove move;
    int is_legal = -1;
    const int valid_path[] = { 16, 21 };
    const int wrong_direction_path[] = { 16, 11 };
    const int mandatory_capture_quiet_path[] = { 24, 29 };
    const int capture_path[] = { 21, 32 };
    const int captures[] = { 27 };
    const int invalid_square_path[] = { 16, 51 };

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    move = make_move(valid_path, 2, NULL, 0);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 1);

    move = make_move(wrong_direction_path, 2, NULL, 0);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 0);

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 24, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(capture_path, 2, captures, 1);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 1);
    move = make_move(mandatory_capture_quiet_path, 2, NULL, 0);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 0);

    move = make_move(invalid_square_path, 2, NULL, 0);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 0);
}

static void test_is_legal_move_rejects_shorter_capture_route(void)
{
    DraughtsState state;
    DraughtsMove move;
    int is_legal = -1;
    const int long_path[] = { 22, 31, 42 };
    const int long_captures[] = { 27, 37 };
    const int short_path[] = { 22, 33 };
    const int short_captures[] = { 28 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 22, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    move = make_move(long_path, 3, long_captures, 2);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 1);

    move = make_move(short_path, 2, short_captures, 1);
    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 0);
}

static void test_apply_quiet_move_switches_side(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int path[] = { 16, 21 };

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    move = make_move(path, 2, NULL, 0);

    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 16, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 21, DRAUGHTS_BLACK_MAN));
    CHECK(side_to_move_is(&state, DRAUGHTS_WHITE));
}

static void test_apply_single_capture_removes_piece(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int path[] = { 21, 32 };
    const int captures[] = { 27 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(path, 2, captures, 1);

    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 21, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 27, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 32, DRAUGHTS_BLACK_MAN));
    CHECK(side_to_move_is(&state, DRAUGHTS_WHITE));
}

static void test_apply_multi_capture_removes_all_pieces(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int path[] = { 21, 32, 43 };
    const int captures[] = { 27, 38 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 38, DRAUGHTS_WHITE_KING) == DRAUGHTS_OK);
    move = make_move(path, 3, captures, 2);

    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 21, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 27, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 38, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 43, DRAUGHTS_BLACK_MAN));
    CHECK(side_to_move_is(&state, DRAUGHTS_WHITE));
}

static void test_apply_king_capture_removes_piece_and_keeps_king(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int path[] = { 28, 10 };
    const int captures[] = { 19 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(path, 2, captures, 1);

    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 28, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 19, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 10, DRAUGHTS_BLACK_KING));
    CHECK(side_to_move_is(&state, DRAUGHTS_WHITE));
}

static void test_apply_king_capture_can_return_to_origin(void)
{
    DraughtsState state;
    DraughtsMove move;
    int is_legal = 0;
    const int path[] = { 49, 38, 24, 35, 49 };
    const int captures[] = { 43, 29, 30, 40 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 49, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 43, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 29, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 30, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 40, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(path, 5, captures, 4);

    CHECK(draughts_is_legal_move(&state, &move, &is_legal) == DRAUGHTS_OK);
    CHECK(is_legal == 1);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 49, DRAUGHTS_BLACK_KING));
    CHECK(piece_at(&state, 43, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 29, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 30, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 40, DRAUGHTS_EMPTY));
    CHECK(side_to_move_is(&state, DRAUGHTS_WHITE));
}

static void test_apply_promotes_black_and_white_men_on_final_square(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int black_path[] = { 41, 46 };
    const int white_path[] = { 10, 5 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 41, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    move = make_move(black_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 46, DRAUGHTS_BLACK_KING));
    CHECK(side_to_move_is(&state, DRAUGHTS_WHITE));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 10, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);
    move = make_move(white_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 5, DRAUGHTS_WHITE_KING));
    CHECK(side_to_move_is(&state, DRAUGHTS_BLACK));
}

static void test_apply_does_not_promote_when_jump_only_passes_back_rank(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int path[] = { 37, 48, 39 };
    const int captures[] = { 42, 43 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 42, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 43, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(path, 3, captures, 2);

    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_OK);
    CHECK(piece_at(&state, 48, DRAUGHTS_EMPTY));
    CHECK(piece_at(&state, 39, DRAUGHTS_BLACK_MAN));
}

static void test_apply_rejects_invalid_moves_without_modifying_state(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int empty_start_path[] = { 21, 26 };
    const int wrong_side_path[] = { 31, 26 };
    const int occupied_final_path[] = { 16, 21 };
    const int bad_square_path[] = { 16, 51 };
    const int bad_capture_path[] = { 21, 32 };
    const int bad_capture[] = { 27 };
    const int duplicate_capture_path[] = { 21, 32, 43 };
    const int duplicate_captures[] = { 27, 27 };

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    move = make_move(empty_start_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 21, DRAUGHTS_EMPTY));

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    move = make_move(wrong_side_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 31, DRAUGHTS_WHITE_MAN));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 16, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(occupied_final_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 16, DRAUGHTS_BLACK_MAN));
    CHECK(piece_at(&state, 21, DRAUGHTS_WHITE_MAN));

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    move = make_move(bad_square_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_SQUARE);

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    move = make_move(bad_capture_path, 2, bad_capture, 1);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 21, DRAUGHTS_BLACK_MAN));
    CHECK(piece_at(&state, 27, DRAUGHTS_BLACK_MAN));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(duplicate_capture_path, 3, duplicate_captures, 2);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
}

static void test_apply_rejects_structural_but_not_legal_moves(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int wrong_direction_path[] = { 16, 11 };
    const int mandatory_capture_quiet_path[] = { 24, 29 };
    const int short_capture_path[] = { 22, 33 };
    const int short_capture[] = { 28 };

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    move = make_move(wrong_direction_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 16, DRAUGHTS_BLACK_MAN));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 24, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(mandatory_capture_quiet_path, 2, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 24, DRAUGHTS_BLACK_MAN));
    CHECK(piece_at(&state, 29, DRAUGHTS_EMPTY));

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 22, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    move = make_move(short_capture_path, 2, short_capture, 1);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
    CHECK(piece_at(&state, 22, DRAUGHTS_BLACK_MAN));
    CHECK(piece_at(&state, 28, DRAUGHTS_WHITE_MAN));
    CHECK(piece_at(&state, 33, DRAUGHTS_EMPTY));
}

static void test_apply_rejects_invalid_move_shape_and_nulls(void)
{
    DraughtsState state;
    DraughtsMove move;
    const int path[] = { 16, 21, 26 };
    const int capture_path[] = { 21, 32, 43 };
    const int captures[] = { 27 };

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    CHECK(draughts_apply_move(NULL, &move) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_apply_move(&state, NULL) == DRAUGHTS_ERR_NULL);

    move = make_move(path, 3, NULL, 0);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);

    move = make_move(capture_path, 3, captures, 1);
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);

    move = make_move(path, 2, NULL, 0);
    move.path_len = 1;
    CHECK(draughts_apply_move(&state, &move) == DRAUGHTS_ERR_INVALID_MOVE);
}

int main(void)
{
    test_initial_black_quiet_moves();
    test_initial_white_quiet_moves();
    test_king_quiet_moves_from_center();
    test_white_king_quiet_moves_and_side_filter();
    test_king_quiet_moves_stop_at_any_piece();
    test_king_single_capture_has_all_landing_squares();
    test_king_multi_capture_keeps_longest_routes();
    test_king_captured_piece_barrier_blocks_deeper_target();
    test_king_capture_blockers();
    test_king_and_man_captures_use_same_max_rule();
    test_king_capture_buffer_too_small_reports_total_landings();
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
    test_is_legal_move_accepts_generated_and_rejects_nonlegal();
    test_is_legal_move_rejects_shorter_capture_route();
    test_apply_quiet_move_switches_side();
    test_apply_single_capture_removes_piece();
    test_apply_multi_capture_removes_all_pieces();
    test_apply_king_capture_removes_piece_and_keeps_king();
    test_apply_king_capture_can_return_to_origin();
    test_apply_promotes_black_and_white_men_on_final_square();
    test_apply_does_not_promote_when_jump_only_passes_back_rank();
    test_apply_rejects_invalid_moves_without_modifying_state();
    test_apply_rejects_structural_but_not_legal_moves();
    test_apply_rejects_invalid_move_shape_and_nulls();

    if (failures != 0) {
        printf("%d test failure(s)\n", failures);
        return 1;
    }

    printf("All move tests passed\n");
    return 0;
}
