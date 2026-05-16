#include "draughts.h"

#include <stdio.h>
#include <string.h>

static void print_usage(const char *program)
{
    printf("Usage: %s --ids|--initial\n", program);
}

static void print_id_board(void)
{
    int row;
    int col;
    int square;

    for (row = 0; row < DRAUGHTS_BOARD_ROWS; ++row) {
        for (col = 0; col < DRAUGHTS_BOARD_COLS; ++col) {
            if (draughts_coord_to_square(row, col, &square) == DRAUGHTS_OK) {
                printf("%3d", square);
            } else {
                printf("%3s", ".");
            }
        }
        printf("\n");
    }
}

static void print_state_board(const DraughtsState *state)
{
    int row;
    int col;
    int square;
    DraughtsPiece piece;

    printf("    1   2   3   4   5   6   7   8   9  10\n");
    printf("  +---+---+---+---+---+---+---+---+---+---+\n");
    for (row = 0; row < DRAUGHTS_BOARD_ROWS; ++row) {
        printf("%2d|", row + 1);
        for (col = 0; col < DRAUGHTS_BOARD_COLS; ++col) {
            if (draughts_coord_to_square(row, col, &square) == DRAUGHTS_OK) {
                if (draughts_get_piece(state, square, &piece) == DRAUGHTS_OK) {
                    printf(" %c |", draughts_piece_to_char(piece));
                } else {
                    printf(" ? |");
                }
            } else {
                printf("   |");
            }
        }
        printf("\n");
        printf("  +---+---+---+---+---+---+---+---+---+---+\n");
    }

    printf("\n");
    printf("b = black man, B = black king\n");
    printf("w = white man, W = white king\n");
    printf(". = empty playable square, blank = non-playable square\n");
}

int main(int argc, char **argv)
{
    DraughtsState state;

    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--ids") == 0) {
        print_id_board();
        return 0;
    }

    if (strcmp(argv[1], "--initial") == 0) {
        if (draughts_init(&state) != DRAUGHTS_OK) {
            printf("Failed to initialize board\n");
            return 1;
        }
        print_state_board(&state);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
