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

static void test_square_mapping(void)
{
    int row = -1;
    int col = -1;
    int square = -1;

    CHECK(draughts_square_to_coord(1, &row, &col) == DRAUGHTS_OK);
    CHECK(row == 0);
    CHECK(col == 1);

    CHECK(draughts_square_to_coord(5, &row, &col) == DRAUGHTS_OK);
    CHECK(row == 0);
    CHECK(col == 9);

    CHECK(draughts_square_to_coord(6, &row, &col) == DRAUGHTS_OK);
    CHECK(row == 1);
    CHECK(col == 0);

    CHECK(draughts_square_to_coord(50, &row, &col) == DRAUGHTS_OK);
    CHECK(row == 9);
    CHECK(col == 8);

    CHECK(draughts_coord_to_square(0, 1, &square) == DRAUGHTS_OK);
    CHECK(square == 1);

    CHECK(draughts_coord_to_square(1, 0, &square) == DRAUGHTS_OK);
    CHECK(square == 6);

    CHECK(draughts_coord_to_square(9, 8, &square) == DRAUGHTS_OK);
    CHECK(square == 50);

    CHECK(draughts_is_playable_coord(0, 0) == 0);
    CHECK(draughts_is_playable_coord(0, 1) == 1);
    CHECK(draughts_is_playable_coord(1, 0) == 1);
    CHECK(draughts_is_playable_coord(10, 0) == 0);

    CHECK(draughts_square_to_coord(0, &row, &col) == DRAUGHTS_ERR_INVALID_SQUARE);
    CHECK(draughts_square_to_coord(51, &row, &col) == DRAUGHTS_ERR_INVALID_SQUARE);
    CHECK(draughts_coord_to_square(0, 0, &square) == DRAUGHTS_ERR_INVALID_COORD);
}

static void test_initial_position(void)
{
    DraughtsState state;
    DraughtsPiece piece;
    DraughtsSide side;
    int black_count = 0;
    int white_count = 0;
    int empty_count = 0;
    int i;

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    CHECK(draughts_get_side_to_move(&state, &side) == DRAUGHTS_OK);
    CHECK(side == DRAUGHTS_BLACK);

    for (i = 1; i <= DRAUGHTS_SQUARE_COUNT; ++i) {
        CHECK(draughts_get_piece(&state, i, &piece) == DRAUGHTS_OK);
        if (piece == DRAUGHTS_BLACK_MAN) {
            ++black_count;
            CHECK(i >= 1 && i <= 20);
        } else if (piece == DRAUGHTS_WHITE_MAN) {
            ++white_count;
            CHECK(i >= 31 && i <= 50);
        } else if (piece == DRAUGHTS_EMPTY) {
            ++empty_count;
            CHECK(i >= 21 && i <= 30);
        } else {
            CHECK(0);
        }
    }

    CHECK(black_count == 20);
    CHECK(white_count == 20);
    CHECK(empty_count == 10);
}

static void test_get_set_clear_and_copy(void)
{
    DraughtsState state;
    DraughtsState copy;
    DraughtsPiece piece;
    DraughtsSide side;
    int i;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_get_side_to_move(&state, &side) == DRAUGHTS_OK);
    CHECK(side == DRAUGHTS_BLACK);

    for (i = 1; i <= DRAUGHTS_SQUARE_COUNT; ++i) {
        CHECK(draughts_get_piece(&state, i, &piece) == DRAUGHTS_OK);
        CHECK(piece == DRAUGHTS_EMPTY);
    }

    CHECK(draughts_set_piece(&state, 23, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_get_piece(&state, 23, &piece) == DRAUGHTS_OK);
    CHECK(piece == DRAUGHTS_BLACK_KING);

    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);
    CHECK(draughts_copy(&copy, &state) == DRAUGHTS_OK);

    CHECK(draughts_set_piece(&state, 23, DRAUGHTS_EMPTY) == DRAUGHTS_OK);
    CHECK(draughts_get_piece(&copy, 23, &piece) == DRAUGHTS_OK);
    CHECK(piece == DRAUGHTS_BLACK_KING);

    CHECK(draughts_get_side_to_move(&copy, &side) == DRAUGHTS_OK);
    CHECK(side == DRAUGHTS_WHITE);

    CHECK(draughts_get_piece(NULL, 1, &piece) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_get_piece(&state, 1, NULL) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_get_piece(&state, 0, &piece) == DRAUGHTS_ERR_INVALID_SQUARE);
    CHECK(draughts_set_piece(&state, 51, DRAUGHTS_EMPTY) == DRAUGHTS_ERR_INVALID_SQUARE);
    CHECK(draughts_set_piece(&state, 1, (DraughtsPiece)99) == DRAUGHTS_ERR_INVALID_PIECE);
    CHECK(draughts_set_side_to_move(&state, (DraughtsSide)99) == DRAUGHTS_ERR_INVALID_SIDE);
    state.side_to_move = 99;
    CHECK(draughts_get_side_to_move(&state, &side) == DRAUGHTS_ERR_INVALID_SIDE);
}

static void test_game_result_by_material(void)
{
    DraughtsState state;
    DraughtsGameResult result;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 31, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_WHITE_WIN);

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 20, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_BLACK_WIN);
}

static void test_game_result_ongoing_when_current_side_has_move(void)
{
    DraughtsState state;
    DraughtsGameResult result;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 16, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 50, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);

    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_ONGOING);

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_ONGOING);
}

static void test_game_result_current_side_no_legal_move_loses(void)
{
    DraughtsState state;
    DraughtsGameResult result;

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 46, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_WHITE_WIN);

    CHECK(draughts_set_side_to_move(&state, DRAUGHTS_WHITE) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_BLACK_WIN);

    CHECK(draughts_clear(&state) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 46, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 41, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 47, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK);
    CHECK(draughts_set_piece(&state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_OK);
    CHECK(result == DRAUGHTS_GAME_WHITE_WIN);
}

static void test_game_result_invalid_arguments_and_state(void)
{
    DraughtsState state;
    DraughtsGameResult result;

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    CHECK(draughts_get_game_result(NULL, &result) == DRAUGHTS_ERR_NULL);
    CHECK(draughts_get_game_result(&state, NULL) == DRAUGHTS_ERR_NULL);

    state.side_to_move = 99;
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_ERR_INVALID_SIDE);

    CHECK(draughts_init(&state) == DRAUGHTS_OK);
    state.squares[0] = 99;
    CHECK(draughts_get_game_result(&state, &result) == DRAUGHTS_ERR_INVALID_PIECE);
}

int main(void)
{
    test_square_mapping();
    test_initial_position();
    test_get_set_clear_and_copy();
    test_game_result_by_material();
    test_game_result_ongoing_when_current_side_has_move();
    test_game_result_current_side_no_legal_move_loses();
    test_game_result_invalid_arguments_and_state();

    if (failures != 0) {
        printf("%d test failure(s)\n", failures);
        return 1;
    }

    printf("All board tests passed\n");
    return 0;
}
