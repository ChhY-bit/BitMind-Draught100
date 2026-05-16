#include "draughts.h"

#include <string.h>

static int is_valid_square_id(int square_id)
{
    return square_id >= 1 && square_id <= DRAUGHTS_SQUARE_COUNT;
}

static int is_valid_piece(DraughtsPiece piece)
{
    return piece >= DRAUGHTS_EMPTY && piece <= DRAUGHTS_WHITE_KING;
}

static int is_valid_side(DraughtsSide side)
{
    return side == DRAUGHTS_BLACK || side == DRAUGHTS_WHITE;
}

int draughts_is_playable_coord(int row, int col)
{
    if (row < 0 || row >= DRAUGHTS_BOARD_ROWS || col < 0 || col >= DRAUGHTS_BOARD_COLS) {
        return 0;
    }

    return ((row + col) % 2) == 1;
}

int draughts_square_to_coord(int square_id, int *row, int *col)
{
    int zero_based;
    int board_row;
    int offset_in_row;
    int board_col;

    if (!is_valid_square_id(square_id)) {
        return DRAUGHTS_ERR_INVALID_SQUARE;
    }

    zero_based = square_id - 1;
    board_row = zero_based / 5;
    offset_in_row = zero_based % 5;
    board_col = offset_in_row * 2 + ((board_row % 2) == 0 ? 1 : 0);

    if (row != NULL) {
        *row = board_row;
    }
    if (col != NULL) {
        *col = board_col;
    }

    return DRAUGHTS_OK;
}

int draughts_coord_to_square(int row, int col, int *square_id)
{
    int offset_in_row;

    if (!draughts_is_playable_coord(row, col)) {
        return DRAUGHTS_ERR_INVALID_COORD;
    }

    offset_in_row = col / 2;
    if (square_id != NULL) {
        *square_id = row * 5 + offset_in_row + 1;
    }

    return DRAUGHTS_OK;
}

int draughts_clear(DraughtsState *state)
{
    if (state == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    memset(state->squares, DRAUGHTS_EMPTY, sizeof(state->squares));
    state->side_to_move = DRAUGHTS_BLACK;

    return DRAUGHTS_OK;
}

int draughts_init(DraughtsState *state)
{
    int i;

    if (state == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    for (i = 0; i < DRAUGHTS_SQUARE_COUNT; ++i) {
        if (i < 20) {
            state->squares[i] = DRAUGHTS_BLACK_MAN;
        } else if (i < 30) {
            state->squares[i] = DRAUGHTS_EMPTY;
        } else {
            state->squares[i] = DRAUGHTS_WHITE_MAN;
        }
    }
    state->side_to_move = DRAUGHTS_BLACK;

    return DRAUGHTS_OK;
}

int draughts_copy(DraughtsState *dst, const DraughtsState *src)
{
    if (dst == NULL || src == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    *dst = *src;

    return DRAUGHTS_OK;
}

int draughts_get_piece(const DraughtsState *state, int square_id, DraughtsPiece *piece)
{
    if (state == NULL || piece == NULL) {
        return DRAUGHTS_ERR_NULL;
    }
    if (!is_valid_square_id(square_id)) {
        return DRAUGHTS_ERR_INVALID_SQUARE;
    }

    *piece = (DraughtsPiece)state->squares[square_id - 1];

    return DRAUGHTS_OK;
}

int draughts_set_piece(DraughtsState *state, int square_id, DraughtsPiece piece)
{
    if (state == NULL) {
        return DRAUGHTS_ERR_NULL;
    }
    if (!is_valid_square_id(square_id)) {
        return DRAUGHTS_ERR_INVALID_SQUARE;
    }
    if (!is_valid_piece(piece)) {
        return DRAUGHTS_ERR_INVALID_PIECE;
    }

    state->squares[square_id - 1] = (unsigned char)piece;

    return DRAUGHTS_OK;
}

int draughts_get_side_to_move(const DraughtsState *state, DraughtsSide *side)
{
    if (state == NULL || side == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    *side = (DraughtsSide)state->side_to_move;

    return DRAUGHTS_OK;
}

int draughts_set_side_to_move(DraughtsState *state, DraughtsSide side)
{
    if (state == NULL) {
        return DRAUGHTS_ERR_NULL;
    }
    if (!is_valid_side(side)) {
        return DRAUGHTS_ERR_INVALID_SIDE;
    }

    state->side_to_move = (unsigned char)side;

    return DRAUGHTS_OK;
}

char draughts_piece_to_char(DraughtsPiece piece)
{
    switch (piece) {
    case DRAUGHTS_EMPTY:
        return '.';
    case DRAUGHTS_BLACK_MAN:
        return 'b';
    case DRAUGHTS_BLACK_KING:
        return 'B';
    case DRAUGHTS_WHITE_MAN:
        return 'w';
    case DRAUGHTS_WHITE_KING:
        return 'W';
    default:
        return '?';
    }
}
