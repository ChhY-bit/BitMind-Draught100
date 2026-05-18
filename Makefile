CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -Werror -Iengine/include
PIC_CFLAGS := $(CFLAGS) -fPIC

BUILD_DIR := build
ENGINE_OBJS := $(BUILD_DIR)/board.o
MOVE_OBJS := $(BUILD_DIR)/move.o
ALL_ENGINE_OBJS := $(ENGINE_OBJS) $(MOVE_OBJS)
PIC_ENGINE_OBJS := $(BUILD_DIR)/board.pic.o $(BUILD_DIR)/move.pic.o
INTERFACE_OBJ := $(BUILD_DIR)/draughts_interface.pic.o
LIBDRAUGHTS := $(BUILD_DIR)/libdraughts.so
TEST_BOARD := $(BUILD_DIR)/test_board
TEST_MOVES := $(BUILD_DIR)/test_moves
RANDOM_PLAY := $(BUILD_DIR)/random_play
BOARD_VIEWER := $(BUILD_DIR)/board_viewer

.PHONY: all test random-test tools interface interface-test trainer-test clean

all: test tools

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/board.o: engine/src/board.c engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c engine/src/board.c -o $@

$(BUILD_DIR)/move.o: engine/src/move.c engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c engine/src/move.c -o $@

$(BUILD_DIR)/board.pic.o: engine/src/board.c engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(PIC_CFLAGS) -c engine/src/board.c -o $@

$(BUILD_DIR)/move.pic.o: engine/src/move.c engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(PIC_CFLAGS) -c engine/src/move.c -o $@

$(INTERFACE_OBJ): interface/draughts_interface.c interface/draughts_interface.h engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(PIC_CFLAGS) -Iinterface -c interface/draughts_interface.c -o $@

$(LIBDRAUGHTS): $(PIC_ENGINE_OBJS) $(INTERFACE_OBJ) | $(BUILD_DIR)
	$(CC) -shared $(PIC_ENGINE_OBJS) $(INTERFACE_OBJ) -o $@

$(TEST_BOARD): engine/tests/test_board.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tests/test_board.c $(ALL_ENGINE_OBJS) -o $@

$(TEST_MOVES): engine/tests/test_moves.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tests/test_moves.c $(ALL_ENGINE_OBJS) -o $@

$(RANDOM_PLAY): engine/tests/random_play.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tests/random_play.c $(ALL_ENGINE_OBJS) -o $@

$(BOARD_VIEWER): engine/tools/board_viewer.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tools/board_viewer.c $(ALL_ENGINE_OBJS) -o $@

test: $(TEST_BOARD) $(TEST_MOVES)
	$(TEST_BOARD)
	$(TEST_MOVES)

random-test: $(RANDOM_PLAY)
	$(RANDOM_PLAY)

tools: $(BOARD_VIEWER)

interface: $(LIBDRAUGHTS)

interface-test: interface
	PYTHONPATH=interface python3 -m unittest interface/test_interface.py

trainer-test: interface
	PYTHONPATH=interface:trainer python3 -m unittest trainer/test_env.py trainer/test_agents.py trainer/test_arena.py trainer/test_selfplay.py trainer/test_inspect_dataset.py trainer/test_dataset.py

clean:
	rm -rf $(BUILD_DIR)
